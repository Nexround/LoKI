"""
Utility functions for creating, modifying, and restoring LoKI models.

Provides high-level APIs for the LoKI workflow:
- create_loki_model: Generate LoKI model from pretrained base model
- restore_loki_model: Merge LoKI weights back into standard model format
- set_zero_weights: Zero out specific neuron weights for baseline comparison
"""

import importlib.util
import json
import logging
import sys
from glob import glob
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import (
    AutoModel,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
)

from ..core import LoKILinear
from ..models import ArchitectureSpec, get_architecture_spec, register_architecture

logger = logging.getLogger(__name__)


def _build_loki_module_source(spec: ArchitectureSpec) -> tuple[str, str, str]:
    """
    Build the Python source code for a file-backed LoKI wrapper module.

    Returns the source string plus the generated model and config class names.
    """
    model_class_name = f"LoKI{spec.name}ForCausalLM"
    config_class_name = f"LoKI{spec.name}Config"
    base_model_module = spec.base_model_cls.__module__
    base_model_name = spec.base_model_cls.__name__
    base_config_module = spec.base_config_cls.__module__
    base_config_name = spec.base_config_cls.__name__
    source = f'''"""Auto-generated LoKI wrapper for {spec.name}.

This file is created at model-conversion time so that Hugging Face's
custom_object_save sees a real module on disk instead of a dynamically
constructed class defined in-memory. The code is self-contained so the
saved model can be loaded without installing the LoKI package.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from importlib import import_module

import torch
import torch.nn as nn
from transformers import PreTrainedModel

logger = logging.getLogger(__name__)

_BASE_MODEL_MODULE = "{base_model_module}"
_BASE_MODEL_NAME = "{base_model_name}"
_BASE_CONFIG_MODULE = "{base_config_module}"
_BASE_CONFIG_NAME = "{base_config_name}"

_BASE_MODEL_CLS = getattr(import_module(_BASE_MODEL_MODULE), _BASE_MODEL_NAME)
_BASE_CONFIG_CLS = getattr(import_module(_BASE_CONFIG_MODULE), _BASE_CONFIG_NAME)


def _get_layers(model: PreTrainedModel):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    for attr in ("language_model", "text_model", "encoder"):
        if hasattr(model, attr):
            sub = getattr(model, attr)
            if hasattr(sub, "layers"):
                return sub.layers
            if hasattr(sub, "model") and hasattr(sub.model, "layers"):
                return sub.model.layers
    if hasattr(model, "layers"):
        return model.layers
    raise AttributeError(
        "Cannot locate transformer layers on model (model/model.language_model/text_model/encoder)"
    )


def _infer_mlp_attr(layer) -> str:
    for candidate in ("mlp", "ffn"):
        if hasattr(layer, candidate):
            return candidate
    for attr, value in vars(layer).items():
        if hasattr(value, "down_proj"):
            return attr
    raise AttributeError("Cannot infer MLP attribute on layer")


class LoKILinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, target_pos):
        super().__init__()
        self.out_features = original_linear.out_features
        self.in_features = original_linear.in_features
        self.active_pos = sorted(target_pos)
        self.frozen_pos = [
            i for i in range(self.out_features) if i not in self.active_pos
        ]
        if not all(0 <= idx < self.out_features for idx in self.active_pos):
            raise ValueError(
                f"Target neuron indices must be within [0, {{self.out_features - 1}}]"
            )
        if len(self.active_pos) != len(set(self.active_pos)):
            raise ValueError("Target neuron indices contain duplicate values")
        self.active = nn.Linear(self.in_features, len(self.active_pos), bias=False)
        self.frozen = nn.Linear(self.in_features, len(self.frozen_pos), bias=False)
        W = original_linear.weight.data
        self.active.weight = nn.Parameter(W[self.active_pos].clone(), requires_grad=True)
        self.frozen.weight = nn.Parameter(W[self.frozen_pos].clone(), requires_grad=False)
        if original_linear.bias is not None:
            b = original_linear.bias.data
            self.active_bias = nn.Parameter(b[self.active_pos].clone(), requires_grad=True)
            self.frozen_bias = nn.Parameter(b[self.frozen_pos].clone(), requires_grad=False)
        else:
            self.register_parameter("active_bias", None)
            self.register_parameter("frozen_bias", None)
        index_map = torch.empty(self.out_features, dtype=torch.long)
        index_map[self.active_pos] = torch.arange(len(self.active_pos))
        index_map[self.frozen_pos] = torch.arange(len(self.frozen_pos)) + len(self.active_pos)
        self.register_buffer("index_map", index_map)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        active_out = self.active(x)
        frozen_out = self.frozen(x)
        output = torch.cat([active_out, frozen_out], dim=-1)
        if self.active_bias is not None:
            bias = torch.cat([self.active_bias, self.frozen_bias], dim=0)
            output += bias.unsqueeze(0).unsqueeze(0)
        return output.gather(
            dim=-1,
            index=self.index_map.view(1, 1, -1).expand(
                output.size(0), output.size(1), -1
            ),
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={{self.in_features}}, "
            f"out_features={{self.out_features}}, "
            f"active_neurons={{len(self.active_pos)}} "
            f"({{100 * len(self.active_pos) / self.out_features:.1f}}%)"
        )


class BaseLoKIModel(ABC):
    def __init__(self, config):
        if not hasattr(config, "target_pos") or config.target_pos is None:
            raise ValueError(
                f"Config must include `target_pos` attribute, but got: {{config}}"
            )
        self.target_pos = config.target_pos
        if len(self.target_pos) != config.num_hidden_layers:
            raise ValueError(
                f"Length of target_pos ({{len(self.target_pos)}}) must equal "
                f"num_hidden_layers ({{config.num_hidden_layers}})"
            )
        # Replace target layers up front so weight loading matches LoKILinear
        self.apply_loki_linear()
        # Freeze everything, then unfreeze the LoKI active heads only
        for param in self.parameters():
            param.requires_grad = False
        for module in self.modules():
            if isinstance(module, LoKILinear):
                module.active.weight.requires_grad = True
                if module.active_bias is not None:
                    module.active_bias.requires_grad = True

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *args, **kwargs):
        config = kwargs.pop("config", None)
        if config is None:
            config = cls.config_class.from_pretrained(pretrained_model_name_or_path)
        model = super().from_pretrained(
            pretrained_model_name_or_path,
            *args,
            config=config,
            **kwargs,
        )
        logger.info("Freezing base model parameters")
        if hasattr(model, "model"):
            for param in model.model.parameters():
                param.requires_grad = False
        if hasattr(model, "lm_head"):
            for param in model.lm_head.parameters():
                param.requires_grad = False
        return model

    def apply_loki_linear(self) -> None:
        if not hasattr(self, "config"):
            raise ValueError("Model must have 'config' before applying LoKI")
        layers = _get_layers(self)
        logger.info(f"Replacing down_proj layers in {{self.config.num_hidden_layers}} layers")
        for layer_idx in range(self.config.num_hidden_layers):
            mlp_layer = self._get_mlp_layer(layer_idx, layers)
            original_layer = mlp_layer.down_proj
            target_pos = self.target_pos[layer_idx]
            if len(target_pos) == 0:
                logger.debug(f"Skipping layer {{layer_idx}} (no target neurons)")
                continue
            loki_linear = LoKILinear(
                original_linear=original_layer,
                target_pos=target_pos,
            )
            mlp_layer.down_proj = loki_linear
            logger.info(
                "Replaced down_proj in layer %s (%s/%s neurons trainable)",
                layer_idx,
                len(target_pos),
                original_layer.out_features,
            )

    @abstractmethod
    def _get_mlp_layer(self, layer_idx: int, layers):
        raise NotImplementedError


class {config_class_name}(_BASE_CONFIG_CLS):
    def __init__(self, target_pos=None, **kwargs):
        if target_pos is not None and "num_hidden_layers" not in kwargs:
            kwargs["num_hidden_layers"] = len(target_pos)
        super().__init__(**kwargs)
        self.target_pos = target_pos if target_pos is not None else getattr(self, "target_pos", None)


class {model_class_name}(_BASE_MODEL_CLS, BaseLoKIModel):
    config_class = {config_class_name}

    def __init__(self, config):
        _BASE_MODEL_CLS.__init__(self, config)
        BaseLoKIModel.__init__(self, config)

    def _get_mlp_layer(self, layer_idx: int, layers=None):
        layers = layers or _get_layers(self)
        mlp_attr = _infer_mlp_attr(layers[layer_idx])
        return getattr(layers[layer_idx], mlp_attr)


__all__ = ["{config_class_name}", "{model_class_name}"]
'''
    return source, model_class_name, config_class_name


def _write_module_file(module_path: Path, source: str) -> None:
    """Write generated source code to disk, ensuring the parent directory exists."""
    module_path.parent.mkdir(parents=True, exist_ok=True)
    module_path.write_text(source, encoding="utf-8")


def _load_module_from_path(module_name: str, module_path: Path):
    """Dynamically load a module from the given file path."""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")
    sys.modules.pop(module_name, None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore[call-arg]
    return module


def _materialize_loki_classes(
    spec: ArchitectureSpec, save_dir: Path, module_name: str = "loki_modeling"
) -> tuple[type[PreTrainedModel], type[PretrainedConfig]]:
    """
    Generate a real Python module for the LoKI wrapper and return its classes.

    This follows the required flow to satisfy custom_object_save:
    1) emit a .py file, 2) load it as a module, 3) use the loaded classes.
    """
    register_architecture(spec)
    # Keep the generated source outside the final save_dir root so
    # transformers can copy it into place without hitting SameFileError.
    module_path = save_dir / "_generated" / f"{module_name}.py"
    source, model_class_name, config_class_name = _build_loki_module_source(spec)
    _write_module_file(module_path, source)
    module = _load_module_from_path(module_name, module_path)
    loki_model_class = getattr(module, model_class_name)
    loki_config_cls = getattr(module, config_class_name)
    return loki_model_class, loki_config_cls


def create_loki_model(
    loki_model_class: type[PreTrainedModel] | None = None,
    loki_config_cls: type[PretrainedConfig] | None = None,
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    target_pos_path: str | Path = "",
    save_dir: str | Path = "./loki_model",
    torch_dtype: torch.dtype = torch.bfloat16,
    trust_remote_code: bool = False,
) -> None:
    """
    Create and save a LoKI model from a pretrained base model.

    This now follows the disk-first flow required to make custom_object_save
    treat the wrapper as a normal, file-backed model:
    1) generate a module file that defines the LoKI config/model classes
    2) import that module dynamically
    3) instantiate the classes from the imported module
    4) save via save_pretrained

    If model classes are omitted, they are resolved automatically from the
    registry using the provided model_name.

    Args:
        loki_model_class: Optional LoKI model class override
        loki_config_cls: Optional LoKI config class override
        model_name: Pretrained model identifier or path
        target_pos_path: Path to JSON file with neuron position indices per layer
        save_dir: Output directory for LoKI model and tokenizer
        torch_dtype: Data type for model weights (default: bfloat16)
        trust_remote_code: Allow loading models with custom code

    Note:
        Saves both LoKI model configuration and original weights, as LoKILinear
        is reconstructed on each model load from the original weights.
    """
    logger.info(f"Creating LoKI model from {model_name}")
    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)

    # Resolve architecture spec and materialize file-backed classes
    base_spec = get_architecture_spec(model_name)
    spec = ArchitectureSpec(
        name=base_spec.name,
        model_type=base_spec.model_type,
        base_model_cls=base_spec.base_model_cls,
        base_config_cls=base_spec.base_config_cls,
        mlp_getter=base_spec.mlp_getter,
        down_proj_getter=base_spec.down_proj_getter,
        loki_model_cls=loki_model_class or base_spec.loki_model_cls,
        loki_config_cls=loki_config_cls or base_spec.loki_config_cls,
        kva_model_cls=base_spec.kva_model_cls,
    )
    loki_model_class, loki_config_cls = _materialize_loki_classes(
        spec, save_dir_path
    )
    logger.info(
        f"Generated wrapper module at {save_dir_path / '_generated' / 'loki_modeling.py'}"
    )

    # Load position indices from JSON file
    with open(target_pos_path, encoding="utf-8") as f:
        target_pos = json.load(f)

    logger.info(f"Loaded target positions for {len(target_pos)} layers")

    # Load the original pretrained model
    logger.info("Loading original pretrained model...")
    original_model = AutoModel.from_pretrained(
        model_name, dtype=torch_dtype, trust_remote_code=trust_remote_code
    )

    # Initialize LoKI configuration with specified position indices
    loki_config = loki_config_cls.from_pretrained(model_name, target_pos=target_pos)
    loki_model_class.register_for_auto_class("AutoModel")
    loki_config_cls.register_for_auto_class()

    # Load the LoKI model using the original pretrained weights and new config
    logger.info("Creating LoKI model with selective trainable neurons...")
    loki_model = loki_model_class.from_pretrained(
        pretrained_model_name_or_path=model_name,
        config=loki_config,
        dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )
    # Save LoKI model configuration for Transformers compatibility
    logger.info(f"Saving LoKI model to {save_dir_path}")
    loki_model.save_pretrained(save_dir_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=trust_remote_code
    )
    tokenizer.save_pretrained(save_dir_path)

    # Save original model weights (required for LoKILinear reconstruction)
    logger.info("Saving original model weights...")
    original_model.save_pretrained(save_dir_path, is_main_process=False)

    logger.info("Successfully created and saved LoKI model")


def _merge_loki_weights(
    loki_layer: LoKILinear,
    original_linear: torch.nn.Linear,
) -> None:
    """
    Merge weights from LoKILinear back into standard Linear layer.

    Combines active (trainable) and frozen weights from LoKILinear
    into the original neuron positions in a standard Linear layer.

    Args:
        loki_layer: LoKILinear with split active/frozen weights
        original_linear: Target Linear layer to receive merged weights
    """
    # Allocate space for the merged weight matrix
    merged_weight = torch.zeros_like(original_linear.weight.data)

    # Insert active and frozen weights into correct positions
    merged_weight[loki_layer.active_pos] = loki_layer.active.weight.data
    merged_weight[loki_layer.frozen_pos] = loki_layer.frozen.weight.data

    # If bias exists, merge it similarly
    if original_linear.bias is not None:
        merged_bias = torch.zeros_like(original_linear.bias.data)
        merged_bias[loki_layer.active_pos] = loki_layer.active_bias.data
        merged_bias[loki_layer.frozen_pos] = loki_layer.frozen_bias.data

    # Overwrite the original layer's parameters
    original_linear.weight.data.copy_(merged_weight)
    if original_linear.bias is not None:
        original_linear.bias.data.copy_(merged_bias)


def set_zero_weights(
    target_pos_path: str | Path,
    output_path: str | Path,
    model_name: str | Path,
    torch_dtype: torch.dtype = torch.bfloat16,
) -> None:
    """
    Zero out specific neuron weights in down_proj layers.

    Creates a baseline model by zeroing specified neurons in down_proj layers,
    useful for ablation studies and measuring knowledge impact.

    Args:
        target_pos_path: JSON path with neuron indices to zero per layer
        output_path: Directory to save modified model
        model_name: Base model identifier to load
        torch_dtype: Data type for model weights
    """
    logger.info(f"Creating zero-weight baseline from {model_name}")

    # Load position indices
    with open(target_pos_path, encoding="utf-8") as f:
        target_pos = json.load(f)

    # Load the original pretrained model
    original_model = AutoModel.from_pretrained(model_name, dtype=torch_dtype)

    # Iterate over all transformer layers
    for layer_idx in range(original_model.config.num_hidden_layers):
        down_proj = original_model.model.layers[layer_idx].mlp.down_proj

        # Zero out weights for each specified position index
        with torch.no_grad():
            for idx in target_pos.get(str(layer_idx), []):
                if 0 <= idx < down_proj.weight.size(0):
                    down_proj.weight[idx, :] = 0.0

        logger.debug(f"Zeroed {len(target_pos.get(str(layer_idx), []))} neurons in layer {layer_idx}")

    # Save the modified model and tokenizer
    logger.info(f"Saving zero-weight model to {output_path}")
    original_model.save_pretrained(output_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(output_path)


def restore_loki_model(
    target_pos_path: str | Path,
    model_path: str | Path,
    model_name: str | Path,
    output_path: str | Path,
    torch_dtype: torch.dtype = torch.bfloat16,
) -> None:
    """
    Restore LoKI model to standard Transformers format.

    Merges LoKI-split weights (active/frozen) back into standard Linear layers,
    producing a model compatible with standard Transformers library.

    Args:
        target_pos_path: JSON file with target neuron positions
        model_path: Directory with LoKI model weights (safetensors)
        model_name: Base model identifier for architecture
        output_path: Output directory for restored model
        torch_dtype: Data type for model loading

    Note:
        Handles both single and sharded safetensors files.
    """
    logger.info(f"Restoring LoKI model from {model_path} to standard format")

    # Load target position indices from JSON file
    with open(target_pos_path, encoding="utf-8") as f:
        target_pos = json.load(f)

    # Load base transformer model
    logger.info("Loading base model architecture...")
    original_model = AutoModel.from_pretrained(model_name, dtype=torch_dtype)

    # Initialize tensor dictionary for modified weights
    tensor_dict = {}

    # Handle single safetensors file case
    safe_tensor_path = Path(model_path) / "model.safetensors"
    if safe_tensor_path.is_file():
        logger.info("Loading from single safetensors file")
        with safe_open(safe_tensor_path, framework="pt") as f:
            tensor_dict.update({key: f.get_tensor(key) for key in f.keys()})
    else:
        # Load from sharded safetensors files
        shard_files = sorted(glob(f"{model_path}/model-*-of-*.safetensors"))
        logger.info(f"Loading from {len(shard_files)} sharded safetensors files")
        for shard_file in shard_files:
            with safe_open(shard_file, framework="pt") as f:
                tensor_dict.update({key: f.get_tensor(key) for key in f.keys()})

    # Process each transformer layer
    for layer_idx in range(original_model.config.num_hidden_layers):
        # Skip layers without modifications
        if not target_pos[layer_idx]:
            continue

        # Get original down projection layer
        original_down_proj = original_model.model.layers[layer_idx].mlp.down_proj

        # Initialize LoKI modified layer
        loki_layer = LoKILinear(
            original_down_proj, target_pos=target_pos[layer_idx]
        )

        # Prepare state dictionary for weight loading
        state_dict = {}

        # Load weight parameters
        weight_keys = {
            "active.weight": f"model.layers.{layer_idx}.mlp.down_proj.active.weight",
            "frozen.weight": f"model.layers.{layer_idx}.mlp.down_proj.frozen.weight",
        }
        for param_key, tensor_key in weight_keys.items():
            if tensor_key in tensor_dict:
                state_dict[param_key] = tensor_dict[tensor_key]

        # Load bias parameters if present in original layer
        if original_down_proj.bias is not None:
            bias_keys = {
                "active_bias": f"model.layers.{layer_idx}.mlp.down_proj.active_bias",
                "frozen_bias": f"model.layers.{layer_idx}.mlp.down_proj.frozen_bias"
            }
            for param_key, tensor_key in bias_keys.items():
                if tensor_key in tensor_dict:
                    state_dict[param_key] = tensor_dict[tensor_key]

        # Load parameters into LoKI layer
        loki_layer.load_state_dict(state_dict, strict=False)

        # Integrate LoKI parameters back into base model
        _merge_loki_weights(loki_layer, original_down_proj)

        logger.debug(f"Restored layer {layer_idx}")

    # Save reconstructed model and tokenizer
    logger.info(f"Saving restored model to {output_path}")
    original_model.save_pretrained(output_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(output_path)

    logger.info(f"Model successfully restored and saved to {output_path}")
