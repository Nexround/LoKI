"""
Core abstractions for LoKI models.

This module provides base classes that unify common functionality across
different model architectures (Llama, Qwen, etc.), eliminating code duplication.
"""

from .base_kva_model import BaseKVAModel
from .base_loki_model import BaseLoKIModel
from .loki_linear import LoKILinear

__all__ = ["BaseLoKIModel", "BaseKVAModel", "LoKILinear"]
