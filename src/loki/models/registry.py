"""Architecture registry and dynamic wrapper generation for LoKI/KVA models.

This registry lets us support additional Transformer architectures without
hand-writing a pair of LoKI/KVA classes for each new base model.  Register
the base model class, (LoKI) config class, and how to access MLP/down_proj
layers; we generate the wrapper classes on demand.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable, Union

import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModel,
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedModel,
    PretrainedConfig,
    Qwen2Config,
    Qwen2ForCausalLM,
)

from ..core import BaseKVAModel, BaseLoKIModel

# Type aliases for the registry
LayerAccessor = Callable[[PreTrainedModel, int], nn.Module]
TargetPosType = Union[
    list[list[int] | tuple[int, ...]],
    tuple[list[int] | tuple[int, ...], ...],
]


@dataclass(frozen=True)
class ArchitectureSpec:
    """
    Specification for wiring LoKI/KVA wrappers around a base model.

    Args:
        name: Human-readable architecture name (e.g., "Llama")
        model_type: Hugging Face config.model_type string (e.g., "llama")
        base_model_cls: Hugging Face AutoModel-compatible class
        base_config_cls: Hugging Face config class
        loki_config_cls: Config class that includes target_pos (optional; auto-generated if missing)
        mlp_getter: Callback returning the MLP module for a layer
        down_proj_getter: Callback returning the down_proj Linear for a layer
        loki_model_cls: Optional pre-defined LoKI wrapper (useful for existing code)
        kva_model_cls: Optional pre-defined KVA wrapper (useful for existing code)
    """

    name: str
    model_type: str
    base_model_cls: type[PreTrainedModel]
    base_config_cls: type[PretrainedConfig]
    mlp_getter: LayerAccessor
    down_proj_getter: LayerAccessor
    loki_config_cls: type[PretrainedConfig] | None = None
    loki_model_cls: type[PreTrainedModel] | None = None
    kva_model_cls: type[PreTrainedModel] | None = None


_REGISTRY: dict[str, ArchitectureSpec] = {}


def register_architecture(spec: ArchitectureSpec) -> None:
    """Register an architecture spec keyed by its model_type."""
    _REGISTRY[spec.model_type] = spec


def list_registered_architectures() -> Iterable[str]:
    """Return iterable of registered model_type keys."""
    return _REGISTRY.keys()


def _normalize_model_type(
    identifier: str | Path | PretrainedConfig | PreTrainedModel,
) -> str:
    """Resolve model_type from a string path/name, config, or model instance."""
    if isinstance(identifier, (str, Path)):
        return AutoConfig.from_pretrained(identifier).model_type
    if isinstance(identifier, PretrainedConfig):
        return identifier.model_type
    if isinstance(identifier, PreTrainedModel):
        return identifier.config.model_type
    raise TypeError(
        "identifier must be a model name/path, PretrainedConfig, or PreTrainedModel"
    )


def get_architecture_spec(
    identifier: str | Path | PretrainedConfig | PreTrainedModel,
) -> ArchitectureSpec:
    """Fetch the ArchitectureSpec for a given model identifier."""
    model_type = (
        identifier if isinstance(identifier, str) and identifier in _REGISTRY else None
    )
    if model_type is None:
        model_type = _normalize_model_type(identifier)
    if model_type not in _REGISTRY:
        _auto_register_architecture(identifier, model_type=model_type)
    return _REGISTRY[model_type]


def get_loki_config_class(
    identifier: str | Path | PretrainedConfig | PreTrainedModel,
) -> type[PretrainedConfig]:
    """Return the registered LoKI config class for the given architecture."""
    spec = get_architecture_spec(identifier)
    if spec.loki_config_cls is not None:
        return spec.loki_config_cls
    return _build_loki_config(spec.model_type)


@lru_cache(maxsize=16)
def _build_loki_config(model_type: str) -> type[PretrainedConfig]:
    """Create (and cache) a LoKI config subclass with target_pos support."""
    spec = get_architecture_spec(model_type)
    base_config_cls = spec.base_config_cls

    class LoKIConfig(base_config_cls):  # type: ignore[misc]
        def __init__(self, target_pos: TargetPosType | None = None, **kwargs):
            self.target_pos = target_pos
            super().__init__(**kwargs)

    LoKIConfig.__name__ = f"LoKI{spec.name}Config"
    LoKIConfig.__qualname__ = LoKIConfig.__name__
    LoKIConfig.__module__ = __name__
    return LoKIConfig


@lru_cache(maxsize=16)
def _build_loki_wrapper(model_type: str) -> type[PreTrainedModel]:
    """Create (and cache) a LoKI wrapper class for the given model_type."""
    spec = get_architecture_spec(model_type)

    if spec.loki_model_cls is not None:
        return spec.loki_model_cls

    class LoKIWrapper(BaseLoKIModel, spec.base_model_cls):  # type: ignore[misc]
        config_class = get_loki_config_class(spec.model_type)

        def __init__(self, config):
            spec.base_model_cls.__init__(self, config)
            BaseLoKIModel.__init__(self, config)

        def _get_mlp_layer(self, layer_idx: int):
            return spec.mlp_getter(self, layer_idx)

    LoKIWrapper.__name__ = f"LoKI{spec.name}ForCausalLM"
    LoKIWrapper.__qualname__ = LoKIWrapper.__name__
    return LoKIWrapper


@lru_cache(maxsize=16)
def _build_kva_wrapper(model_type: str) -> type[PreTrainedModel]:
    """Create (and cache) a KVA wrapper class for the given model_type."""
    spec = get_architecture_spec(model_type)

    if spec.kva_model_cls is not None:
        return spec.kva_model_cls

    class KVAWrapper(BaseKVAModel, spec.base_model_cls):  # type: ignore[misc]
        def __init__(self, config):
            spec.base_model_cls.__init__(self, config)
            BaseKVAModel.__init__(self, config)

        def _get_down_proj_layer(self, layer_idx: int):
            return spec.down_proj_getter(self, layer_idx)

    KVAWrapper.__name__ = f"KVA{spec.name}ForCausalLM"
    KVAWrapper.__qualname__ = KVAWrapper.__name__
    return KVAWrapper


def get_loki_model_class(
    identifier: str | Path | PretrainedConfig | PreTrainedModel,
) -> type[PreTrainedModel]:
    """Resolve the LoKI wrapper class for the given architecture/model."""
    spec = get_architecture_spec(identifier)
    return _build_loki_wrapper(spec.model_type)


def get_kva_model_class(
    identifier: str | Path | PretrainedConfig | PreTrainedModel,
) -> type[PreTrainedModel]:
    """Resolve the KVA wrapper class for the given architecture/model."""
    spec = get_architecture_spec(identifier)
    return _build_kva_wrapper(spec.model_type)


# ---------- Built-in registrations ----------

mget = lambda model, idx: model.model.layers[idx].mlp  # noqa: E731
dpget = lambda model, idx: model.model.layers[idx].mlp.down_proj  # noqa: E731

register_architecture(
    ArchitectureSpec(
        name="Llama",
        model_type="llama",
        base_model_cls=LlamaForCausalLM,
        base_config_cls=LlamaConfig,
        mlp_getter=mget,
        down_proj_getter=dpget,
    )
)

register_architecture(
    ArchitectureSpec(
        name="Qwen2",
        model_type="qwen2",
        base_model_cls=Qwen2ForCausalLM,
        base_config_cls=Qwen2Config,
        mlp_getter=lambda model, idx: model.model.layers[idx].mlp,
        down_proj_getter=lambda model, idx: model.model.layers[idx].mlp.down_proj,
    )
)


def _get_layers(model: PreTrainedModel):
    """Return the transformer layers sequence from common attributes."""
    # Standard HF causal LM
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    # Some encoder-only or VLM text towers
    for attr in ("language_model", "text_model", "encoder"):
        if hasattr(model, attr):
            sub = getattr(model, attr)
            if hasattr(sub, "layers"):
                return sub.layers
            if hasattr(sub, "model") and hasattr(sub.model, "layers"):
                return sub.model.layers
    if hasattr(model, "layers"):
        return model.layers
    raise AttributeError("Cannot locate transformer layers on model (model/model.language_model/text_model/encoder)")


def _infer_mlp_attr(layer) -> str:
    """Infer the name of the MLP module on a transformer layer."""
    for candidate in ("mlp", "ffn"):
        if hasattr(layer, candidate):
            return candidate
    for attr, value in vars(layer).items():
        if hasattr(value, "down_proj"):
            return attr
    raise AttributeError("Cannot infer MLP attribute on layer")


def _auto_register_architecture(
    identifier: str | Path | PretrainedConfig | PreTrainedModel,
    model_type: str,
) -> None:
    """Attempt to infer architecture spec dynamically for unseen models."""
    # Load config and instantiate a model from config to avoid weight download
    if isinstance(identifier, PretrainedConfig):
        config = identifier
    elif isinstance(identifier, PreTrainedModel):
        config = identifier.config
    else:
        config = AutoConfig.from_pretrained(identifier)

    base_config_cls = config.__class__
    # Use the causal LM head version so we always have `.model` and `lm_head`
    # attributes that BaseKVAModel expects.
    model = AutoModel.from_config(config)
    base_model_cls = model.__class__

    layers = _get_layers(model)
    if len(layers) == 0:
        raise ValueError(f"Cannot auto-register '{model_type}': no layers found")

    mlp_attr = _infer_mlp_attr(layers[0])

    def mlp_getter(m: PreTrainedModel, idx: int):
        return getattr(_get_layers(m)[idx], mlp_attr)

    def down_proj_getter(m: PreTrainedModel, idx: int):
        return getattr(mlp_getter(m, idx), "down_proj")

    spec = ArchitectureSpec(
        name=config.model_type.capitalize(),
        model_type=model_type,
        base_model_cls=base_model_cls,
        base_config_cls=base_config_cls,
        mlp_getter=mlp_getter,
        down_proj_getter=down_proj_getter,
    )
    register_architecture(spec)
