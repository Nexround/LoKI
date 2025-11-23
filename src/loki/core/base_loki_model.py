"""Base class for LoKI models that apply selective fine-tuning to MLP layers.

This abstract base class provides common functionality for all LoKI model variants,
eliminating ~95% code duplication between architecture-specific implementations.
"""

import logging
from abc import ABC, abstractmethod

import torch.nn as nn
from transformers import PreTrainedModel

from .loki_linear import LoKILinear

logger = logging.getLogger(__name__)


class BaseLoKIModel(ABC):
    """Abstract base class for LoKI models.

    Provides common initialization, validation, parameter freezing, and layer
    replacement logic. Subclasses only need to implement architecture-specific
    methods for accessing MLP layers.

    Attributes:
        target_pos: List of neuron indices to train per layer
    """

    def __init__(self, config):
        """Initialize LoKI model with target neuron positions.

        Args:
            config: Model configuration with target_pos attribute

        Raises:
            ValueError: If config lacks target_pos or length mismatches num_hidden_layers
        """
        if not hasattr(config, "target_pos") or config.target_pos is None:
            raise ValueError(f"Config must include `target_pos` attribute, but got: {config}")

        # Let the parent PreTrainedModel initialize first
        # (This is called via super().__init__(config) in the child class)

        self.target_pos = config.target_pos

        # Validate neuron configuration matches number of layers
        if len(self.target_pos) != config.num_hidden_layers:
            raise ValueError(
                f"Length of target_pos ({len(self.target_pos)}) must equal "
                f"num_hidden_layers ({config.num_hidden_layers})"
            )

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str, *args, **kwargs
    ) -> PreTrainedModel:
        """Load pretrained model and apply LoKI transformations.

        This method:
        1. Loads the model configuration
        2. Loads pretrained weights
        3. Freezes all base model parameters
        4. Replaces target down_proj layers with LoKILinear

        Args:
            pretrained_model_name_or_path: Path or model identifier
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            LoKI model with selective trainable parameters
        """
        # Automatically load corresponding config file
        config = kwargs.pop("config", None)
        if config is None:
            config = cls.config_class.from_pretrained(pretrained_model_name_or_path)  # type: ignore[attr-defined]

        # Call parent class to load pretrained model
        model = super().from_pretrained(  # type: ignore[misc]
            pretrained_model_name_or_path, *args, config=config, **kwargs
        )

        # Freeze all base model parameters
        logger.info("Freezing base model parameters")
        for param in model.model.parameters():
            param.requires_grad = False
        for param in model.lm_head.parameters():
            param.requires_grad = False

        # Replace target linear layers with LoKILinear
        model.apply_loki_linear()

        return model  # type: ignore[return-value,no-any-return]

    def apply_loki_linear(self) -> None:
        """Replace all target MLP down_proj layers with LoKILinear.

        For each layer specified in self.target_pos, replaces the down_proj
        Linear layer with a LoKILinear layer that splits weights into trainable
        (active) and frozen portions.

        Raises:
            ValueError: If config or model attributes are missing
        """
        if not hasattr(self, "config") or not hasattr(self, "model"):
            raise ValueError("Model must have 'config' and 'model' attributes before applying LoKI")

        logger.info(f"Replacing down_proj layers in {self.config.num_hidden_layers} layers")

        for layer_idx in range(self.config.num_hidden_layers):
            mlp_layer = self._get_mlp_layer(layer_idx)
            original_layer = mlp_layer.down_proj
            target_pos = self.target_pos[layer_idx]

            if len(target_pos) == 0:
                logger.debug(f"Skipping layer {layer_idx} (no target neurons)")
                continue

            # Initialize LoKI layer and replace
            if not isinstance(original_layer, nn.Linear):
                raise TypeError(f"Expected nn.Linear, got {type(original_layer)}")
            loki_linear = LoKILinear(original_linear=original_layer, target_pos=target_pos)

            mlp_layer.down_proj = loki_linear
            logger.info(
                f"Replaced down_proj in layer {layer_idx} "
                f"({len(target_pos)}/{original_layer.out_features} neurons trainable)"
            )

    @abstractmethod
    def _get_mlp_layer(self, layer_idx: int) -> nn.Module:
        """Get the MLP layer for a given layer index.

        This method must be implemented by subclasses to handle architecture-specific
        layer access patterns (e.g., model.layers[idx].mlp for Llama/Qwen).

        Args:
            layer_idx: Index of the transformer layer

        Returns:
            MLP module containing down_proj layer

        Raises:
            NotImplementedError: If subclass doesn't implement this method
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement _get_mlp_layer()")
