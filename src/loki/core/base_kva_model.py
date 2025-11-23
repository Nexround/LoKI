"""Base class for KVA (Knowledge-Value Attribution) analysis models.

Provides common functionality for computing integrated gradients using Captum's
LayerIntegratedGradients to identify knowledge-bearing neurons in MLP layers.
"""

import logging
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.attr import LayerIntegratedGradients

logger = logging.getLogger(__name__)


class BaseKVAModel(ABC):
    """Abstract base class for KVA analysis using Captum.

    Uses LayerIntegratedGradients to compute attributions for each down_proj layer,
    identifying which neurons are most critical for specific knowledge domains.

    Attributes:
        integrated_gradients: List of attribution tensors per layer
    """

    def __init__(self, config):
        """Initialize KVA model and setup for integrated gradients computation.

        Args:
            config: Model configuration with num_hidden_layers attribute
        """
        # Parent class (PreTrainedModel) initializes first via super().__init__(config)
        self.integrated_gradients = [None] * config.num_hidden_layers

        # Freeze all parameters except down_proj weights
        logger.info("Freezing all parameters except down_proj weights")
        for param in self.model.parameters():
            param.requires_grad = False

        # Enable gradients only for down_proj layers
        for layer in self.model.layers:
            layer.mlp.down_proj.weight.requires_grad = True

    def compute_integrated_gradients(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_token_idx: int,
        predicted_label: int,
        steps: int = 10,
        method: str = "riemann_trapezoid",
    ) -> list[torch.Tensor]:
        """Compute integrated gradients for all down_proj layers using Captum.

        Uses Captum's LayerIntegratedGradients with zero baselines to identify
        which neurons in each layer contribute most to the predicted label.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            target_token_idx: Token position to analyze (-1 for last token)
            predicted_label: Target class for attribution
            steps: Number of integration steps (higher = more accurate)
            method: Integration method - one of:
                - 'riemann_trapezoid' (default): Trapezoidal rule
                - 'gausslegendre': Gauss-Legendre quadrature
                - 'riemann_left': Left Riemann sum
                - 'riemann_right': Right Riemann sum
                - 'riemann_middle': Middle Riemann sum

        Returns:
            List of attribution tensors, one per layer [hidden_dim]
        """
        logger.debug(f"Computing integrated gradients with method={method}, steps={steps}")

        # Store context for forward functions
        self._context = {
            "target_token_idx": target_token_idx,
            "predicted_label": predicted_label,
        }

        # Compute IG for each layer independently
        for layer_idx in range(self.config.num_hidden_layers):  # type: ignore[attr-defined]
            # Create forward function that outputs target probability
            def forward_func(input_ids, attention_mask):
                outputs = self.model(input_ids, attention_mask)  # type: ignore[attr-defined]
                logits = self._get_lm_head_output(outputs, self._context["target_token_idx"])
                probs = F.softmax(logits, dim=-1)
                return probs[:, self._context["predicted_label"]]

            # Get the down_proj layer for this transformer layer
            down_proj_layer = self._get_down_proj_layer(layer_idx)

            # Use Captum's LayerIntegratedGradients
            lig = LayerIntegratedGradients(
                forward_func=forward_func,
                layer=down_proj_layer,
            )

            # Compute attributions - Captum handles all integration logic
            attributions = lig.attribute(
                inputs=input_ids,
                baselines=torch.zeros_like(input_ids),
                additional_forward_args=(attention_mask,),
                n_steps=steps,
                method=method,
                attribute_to_layer_input=False,  # Attribute to layer output
            )

            # Extract attribution at target token: [batch, seq_len, hidden] -> [hidden]
            self.integrated_gradients[layer_idx] = (
                attributions[0, target_token_idx, :].detach().cpu()
            )

            if layer_idx % 5 == 0:
                logger.debug(f"Completed layer {layer_idx}/{self.config.num_hidden_layers}")  # type: ignore[attr-defined]

        logger.debug("Integrated gradients computation complete")
        return self.integrated_gradients  # type: ignore[return-value,no-any-return]

    def clean(self) -> None:
        """Clean up stored attribution data to free memory."""
        self.integrated_gradients = [None] * self.config.num_hidden_layers  # type: ignore[attr-defined]
        if hasattr(self, "_context"):
            del self._context

    @abstractmethod
    def _get_down_proj_layer(self, layer_idx: int) -> nn.Module:
        """Get the down_proj layer for a given transformer layer index.

        This method must be implemented by subclasses to handle architecture-specific
        layer access patterns.

        Args:
            layer_idx: Index of the transformer layer

        Returns:
            down_proj Linear layer for that transformer layer

        Raises:
            NotImplementedError: If subclass doesn't implement this method
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _get_down_proj_layer()"
        )

    def _get_lm_head_output(self, outputs, target_token_idx: int) -> torch.Tensor:
        """Extract logits from model output at target token position.

        Default implementation works for most models. Override if needed.

        Args:
            outputs: Model output object
            target_token_idx: Token position to extract

        Returns:
            Logits tensor [batch_size, vocab_size]
        """
        return self.lm_head(outputs.last_hidden_state[:, target_token_idx, :])  # type: ignore[attr-defined,no-any-return]
