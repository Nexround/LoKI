"""
Utility functions for creating, modifying, and restoring LoKI models.

Provides high-level APIs for the LoKI workflow:
- create_loki_model: Generate LoKI model from pretrained base model
- restore_loki_model: Merge LoKI weights back into standard model format
- set_zero_weights: Zero out specific neuron weights for baseline comparison
"""

import json
import logging
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
from ..models import get_loki_config_class, get_loki_model_class

logger = logging.getLogger(__name__)


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

    Loads the base model, applies LoKI transformations based on target_pos
    configuration, and saves both the LoKI model and original weights. If
    model classes are omitted, they are resolved automatically from the
    registry using the provided model_name.

    Args:
        loki_model_class: LoKI model class (optional; auto-resolved from registry)
        loki_config_cls: Corresponding config class (optional; auto-resolved from registry)
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

    # Auto-resolve classes if not provided
    if loki_model_class is None or loki_config_cls is None:
        loki_model_class = loki_model_class or get_loki_model_class(model_name)
        loki_config_cls = loki_config_cls or get_loki_config_class(model_name)

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

    # Load the LoKI model using the original pretrained weights and new config
    logger.info("Creating LoKI model with selective trainable neurons...")
    loki_model = loki_model_class.from_pretrained(
        pretrained_model_name_or_path=model_name,
        config=loki_config,
        dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )
    loki_model_class.register_for_auto_class("AutoModel")
    loki_config_cls.register_for_auto_class()
    # Save LoKI model configuration for Transformers compatibility
    logger.info(f"Saving LoKI model to {save_dir}")
    loki_model.save_pretrained(save_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=trust_remote_code
    )
    tokenizer.save_pretrained(save_dir)

    # Save original model weights (required for LoKILinear reconstruction)
    logger.info("Saving original model weights...")
    original_model.save_pretrained(save_dir, is_main_process=False)

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
