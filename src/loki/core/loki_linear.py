"""LoKILinear: Custom Linear layer with selective trainable parameters.

Splits a Linear layer's weights into 'active' (trainable) and 'frozen' portions
based on neuron indices, enabling selective fine-tuning of specific neurons while
preserving others.
"""


import torch
import torch.nn as nn


class LoKILinear(nn.Module):
    """Linear layer with selective neuron training.

    Splits the output dimension into trainable (active) and frozen portions based
    on target_pos indices. During forward pass, computes both portions separately,
    then reorders to restore original neuron positions.

    Attributes:
        out_features: Total number of output features
        in_features: Number of input features
        active_pos: Sorted list of trainable neuron indices
        frozen_pos: Sorted list of frozen neuron indices
        active: Linear layer for trainable neurons
        frozen: Linear layer for frozen neurons
        index_map: Buffer for efficient output reordering
    """

    def __init__(self, original_linear: nn.Linear, target_pos: list[int]) -> None:
        """Initialize LoKILinear by splitting an original Linear layer.

        Args:
            original_linear: Pretrained Linear layer to split
            target_pos: Indices of neurons to make trainable (0-indexed)

        Raises:
            ValueError: If target_pos contains invalid or duplicate indices
        """
        super().__init__()
        self.out_features = original_linear.out_features
        self.in_features = original_linear.in_features
        self.active_pos = sorted(target_pos)
        self.frozen_pos = [i for i in range(self.out_features) if i not in self.active_pos]

        # Parameter validation
        if not all(0 <= idx < self.out_features for idx in self.active_pos):
            raise ValueError(f"Target neuron indices must be within [0, {self.out_features - 1}]")
        if len(self.active_pos) != len(set(self.active_pos)):
            raise ValueError("Target neuron indices contain duplicate values")

        # Create separate Linear layers for active and frozen portions
        self.active = nn.Linear(self.in_features, len(self.active_pos), bias=False)
        self.frozen = nn.Linear(self.in_features, len(self.frozen_pos), bias=False)

        # Split the weight matrix
        W = original_linear.weight.data
        self.active.weight = nn.Parameter(W[self.active_pos].clone(), requires_grad=True)
        self.frozen.weight = nn.Parameter(W[self.frozen_pos].clone(), requires_grad=False)

        # Handle bias if present
        if original_linear.bias is not None:
            b = original_linear.bias.data
            self.active_bias = nn.Parameter(b[self.active_pos].clone(), requires_grad=True)
            self.frozen_bias = nn.Parameter(b[self.frozen_pos].clone(), requires_grad=False)
        else:
            self.register_parameter("active_bias", None)
            self.register_parameter("frozen_bias", None)

        # Pre-compute index mapping for efficient reordering
        # Maps original position -> position in concatenated [active, frozen] tensor
        index_map = torch.empty(self.out_features, dtype=torch.long)
        index_map[self.active_pos] = torch.arange(len(self.active_pos))
        index_map[self.frozen_pos] = torch.arange(len(self.frozen_pos)) + len(self.active_pos)
        self.register_buffer("index_map", index_map)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with weight splitting and reordering.

        Computes active and frozen portions separately, concatenates them,
        optionally adds bias, then reorders to restore original neuron positions.

        Args:
            x: Input tensor [..., in_features]

        Returns:
            Output tensor [..., out_features] with neurons in original order
        """
        # Compute active and frozen portions
        active_out = self.active(x)
        frozen_out = self.frozen(x)

        # Concatenate: [active neurons | frozen neurons]
        output = torch.cat([active_out, frozen_out], dim=-1)

        # Add bias if present
        if self.active_bias is not None:
            bias = torch.cat([self.active_bias, self.frozen_bias], dim=0)
            # Broadcast bias across batch and sequence dimensions
            output += bias.unsqueeze(0).unsqueeze(0)

        # Reorder output using pre-computed index map to restore original positions
        return output.gather(
            dim=-1,
            index=self.index_map.view(1, 1, -1).expand(output.size(0), output.size(1), -1),
        )

    def extra_repr(self) -> str:
        """String representation for debugging."""
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"active_neurons={len(self.active_pos)} "
            f"({100 * len(self.active_pos) / self.out_features:.1f}%)"
        )
