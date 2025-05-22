import torch
import torch.nn as nn


class LoKILinear(nn.Module):
    def __init__(self, original_linear, target_pos):
        super().__init__()
        self.out_features = original_linear.out_features
        self.in_features = original_linear.in_features
        self.active_pos = sorted(target_pos)
        self.frozen_pos = [
            i for i in range(self.out_features) if i not in self.active_pos
        ]

        # Parameter validation
        if not all(0 <= idx < self.out_features for idx in self.active_pos):
            raise ValueError(f"Activation indices must be within [0, {self.out_features - 1}]")
        if len(self.active_pos) != len(set(self.active_pos)):
            raise ValueError("Activation indices contain duplicate values")
        self.active = nn.Linear(self.in_features, len(self.active_pos), bias=False)
        self.frozen = nn.Linear(self.in_features, len(self.frozen_pos), bias=False)
        # Split the weight matrix
        W = original_linear.weight.data
        self.active.weight = nn.Parameter(W[self.active_pos].clone(), requires_grad=True)
        self.frozen.weight = nn.Parameter(W[self.frozen_pos].clone(), requires_grad=False)

        # Handle bias
        if original_linear.bias is not None:
            b = original_linear.bias.data
            self.active_bias = nn.Parameter(
                b[self.active_pos].clone(), requires_grad=True
            )
            self.frozen_bias = nn.Parameter(
                b[self.frozen_pos].clone(), requires_grad=False
            )
        else:
            self.register_parameter("active_bias", None)
            self.register_parameter("frozen_bias", None)

        # Pre-generate index mapping
        index_map = torch.empty(self.out_features, dtype=torch.long)
        index_map[self.active_pos] = torch.arange(len(self.active_pos))
        index_map[self.frozen_pos] = torch.arange(len(self.frozen_pos)) + len(
            self.active_pos
        )
        self.register_buffer("index_map", index_map)

    def forward(self, x):
        active_out = self.active(x)  # Compute active part via submodule
        frozen_out = self.frozen(x)    # Fixed part
        output = torch.cat([active_out, frozen_out], dim=-1)

        # Add combined bias
        if self.active_bias is not None:
            bias = torch.cat([self.active_bias, self.frozen_bias], dim=0)
            output += bias.unsqueeze(0).unsqueeze(0)  # Broadcast bias to all batches and sequence positions

        # Reorder output using pre-generated indices
        return output.gather(
            dim=-1,
            index=self.index_map.view(1, 1, -1).expand(
                output.size(0), output.size(1), -1
            ),
        )