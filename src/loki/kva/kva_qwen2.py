"""
KVA Qwen2 Model with Captum-based Integrated Gradients.

Uses Captum's LayerIntegratedGradients to compute attributions for MLP down_proj
layer outputs, identifying knowledge-bearing neurons.
"""

from transformers import Qwen2ForCausalLM
import torch
import torch.nn.functional as F
from captum.attr import LayerIntegratedGradients
from typing import List


class KVAQwen2ForCausalLM(Qwen2ForCausalLM):
    """
    Qwen2 model for Knowledge-Value Attribution using Captum.

    Uses LayerIntegratedGradients to compute attributions for each down_proj layer.
    """

    def __init__(self, config):
        super().__init__(config)
        self.integrated_gradients = [None] * self.config.num_hidden_layers

        # Freeze all parameters except down_proj weights
        for param in self.model.parameters():
            param.requires_grad = False
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
    ) -> List[torch.Tensor]:
        """
        Compute integrated gradients for all down_proj layers using Captum.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            target_token_idx: Token position to analyze (-1 for last token)
            predicted_label: Target class for attribution
            steps: Number of integration steps
            method: Integration method ('riemann_trapezoid', 'gausslegendre', etc.)

        Returns:
            List of attribution tensors, one per layer [hidden_dim]
        """
        # Store context for forward functions
        self._context = {
            "target_token_idx": target_token_idx,
            "predicted_label": predicted_label,
        }

        # Compute IG for each layer independently
        for layer_idx in range(self.config.num_hidden_layers):
            # Create forward function that outputs target probability
            def forward_func(input_ids, attention_mask):
                outputs = self.model(input_ids, attention_mask)
                logits = self.lm_head(
                    outputs.last_hidden_state[:, self._context["target_token_idx"], :]
                )
                probs = F.softmax(logits, dim=-1)
                return probs[:, self._context["predicted_label"]]

            # Use Captum's LayerIntegratedGradients
            lig = LayerIntegratedGradients(
                forward_func=forward_func,
                layer=self.model.layers[layer_idx].mlp.down_proj,
            )

            # Compute attributions - Captum handles everything
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

        return self.integrated_gradients

    def clean(self):
        """Clean up stored data."""
        self.integrated_gradients = [None] * self.config.num_hidden_layers
        if hasattr(self, "_context"):
            del self._context
