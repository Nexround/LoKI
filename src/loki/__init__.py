"""LoKI: Low-damage Knowledge Implanting.

Selective fine-tuning method that identifies and trains only specific
knowledge-bearing neurons in MLP down-projection layers.
"""

from .constants import INTEGRATION_METHODS, MMLU_ALL_SETS
from .core import BaseKVAModel, BaseLoKIModel, LoKILinear
from .models import (
    ArchitectureSpec,
    get_architecture_spec,
    get_kva_model_class,
    get_loki_config_class,
    get_loki_model_class,
    list_registered_architectures,
    register_architecture,
)
from .utils import (
    configure_root_logger,
    create_loki_model,
    restore_loki_model,
    set_zero_weights,
    setup_logger,
)

__all__ = [
    # Core classes
    "BaseLoKIModel",
    "BaseKVAModel",
    "LoKILinear",
    # Registry helpers
    "ArchitectureSpec",
    "register_architecture",
    "get_architecture_spec",
    "list_registered_architectures",
    "get_loki_model_class",
    "get_loki_config_class",
    "get_kva_model_class",
    # Utilities
    "setup_logger",
    "configure_root_logger",
    "create_loki_model",
    "set_zero_weights",
    "restore_loki_model",
    # Constants
    "MMLU_ALL_SETS",
    "INTEGRATION_METHODS",
]

__version__ = "0.2.0"
