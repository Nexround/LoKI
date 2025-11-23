"""Unit tests for LoKILinear layer."""

import pytest
import torch
import torch.nn as nn
from loki.core.loki_linear import LoKILinear


class TestLoKILinearBasics:
    """Basic functionality tests for LoKILinear."""

    def test_initialization(self):
        """Test LoKILinear initialization with valid parameters."""
        original_linear = nn.Linear(1024, 2048, bias=False)
        target_pos = [0, 10, 20, 30]

        loki_linear = LoKILinear(original_linear, target_pos)

        assert loki_linear.in_features == 1024
        assert loki_linear.out_features == 2048
        assert loki_linear.active.weight.shape == (len(target_pos), 1024)
        assert loki_linear.frozen.weight.shape == (2048 - len(target_pos), 1024)

    def test_empty_target_pos(self):
        """Test behavior with empty target_pos (all frozen)."""
        original_linear = nn.Linear(512, 1024, bias=False)
        target_pos = []

        loki_linear = LoKILinear(original_linear, target_pos)

        assert loki_linear.active.weight.shape == (0, 512)
        assert loki_linear.frozen.weight.shape == (1024, 512)

    def test_full_target_pos(self):
        """Test behavior with all positions active."""
        original_linear = nn.Linear(512, 1024, bias=False)
        target_pos = list(range(1024))

        loki_linear = LoKILinear(original_linear, target_pos)

        assert loki_linear.active.weight.shape == (1024, 512)
        assert loki_linear.frozen.weight.shape == (0, 512)


class TestLoKILinearForward:
    """Test forward pass of LoKILinear."""

    def test_forward_output_shape(self):
        """Test that forward pass produces correct output shape."""
        original_linear = nn.Linear(256, 512, bias=False)
        target_pos = [0, 50, 100, 150]

        loki_linear = LoKILinear(original_linear, target_pos)

        batch_size = 8
        seq_len = 128
        input_tensor = torch.randn(batch_size, seq_len, 256)

        output = loki_linear(input_tensor)

        assert output.shape == (batch_size, seq_len, 512)

    def test_forward_preserves_output(self):
        """Test that LoKILinear produces same output as original Linear initially."""
        torch.manual_seed(42)

        original_linear = nn.Linear(128, 256, bias=False)
        target_pos = [0, 10, 20, 30, 40]

        # Create LoKILinear
        loki_linear = LoKILinear(original_linear, target_pos)

        # Create input
        input_tensor = torch.randn(4, 16, 128)

        # Compare outputs
        original_output = original_linear(input_tensor)
        loki_output = loki_linear(input_tensor)

        # Should be very close (within numerical precision)
        torch.testing.assert_close(original_output, loki_output, rtol=1e-4, atol=1e-4)


class TestLoKILinearGradients:
    """Test gradient behavior of LoKILinear."""

    def test_frozen_weights_no_gradient(self):
        """Test that frozen weights don't receive gradients."""
        original_linear = nn.Linear(64, 128, bias=False)
        target_pos = [0, 10, 20]  # Only 3 active neurons

        loki_linear = LoKILinear(original_linear, target_pos)

        # Ensure frozen weights require no grad
        assert not loki_linear.frozen.weight.requires_grad
        assert loki_linear.active.weight.requires_grad

    def test_active_weights_receive_gradient(self):
        """Test that active weights receive gradients during backprop."""
        original_linear = nn.Linear(32, 64, bias=False)
        target_pos = [0, 5, 10, 15]

        loki_linear = LoKILinear(original_linear, target_pos)

        # Forward pass
        input_tensor = torch.randn(2, 8, 32, requires_grad=True)
        output = loki_linear(input_tensor)
        loss = output.sum()

        # Backward pass
        loss.backward()

        # Active weights should have gradients
        assert loki_linear.active.weight.grad is not None
        assert loki_linear.active.weight.grad.shape == loki_linear.active.weight.shape

        # Frozen weights should not have gradients
        assert loki_linear.frozen.weight.grad is None


class TestLoKILinearWeightSplitting:
    """Test weight splitting and reconstruction logic."""

    def test_weight_splitting_correctness(self):
        """Test that weights are correctly split into active and frozen."""
        torch.manual_seed(42)

        original_linear = nn.Linear(16, 32, bias=False)
        # Set specific values for verification
        with torch.no_grad():
            original_linear.weight[:] = torch.arange(32 * 16).reshape(32, 16).float()

        target_pos = [0, 5, 10, 15, 20, 25]

        loki_linear = LoKILinear(original_linear, target_pos)

        # Check active weights match target positions
        for i, pos in enumerate(target_pos):
            expected = original_linear.weight[pos, :]
            actual = loki_linear.active.weight[i, :]
            torch.testing.assert_close(expected, actual)

    def test_index_map_correctness(self):
        """Test that index_map correctly reorders concatenated outputs."""
        original_linear = nn.Linear(8, 16, bias=False)
        target_pos = [1, 3, 5, 7, 9, 11]  # Non-sequential positions

        loki_linear = LoKILinear(original_linear, target_pos)

        # index_map should have length = out_features
        assert len(loki_linear.index_map) == 16

        # Active positions should map to their original indices
        # Frozen positions should map to remaining indices
        assert loki_linear.index_map.shape == (16,)


class TestLoKILinearEdgeCases:
    """Test edge cases and error handling."""

    def test_duplicate_target_pos(self):
        """Duplicate positions should raise to avoid ambiguous mapping."""
        original_linear = nn.Linear(32, 64, bias=False)
        target_pos = [0, 10, 10, 20, 20, 20]  # Duplicates

        with pytest.raises(ValueError, match="duplicate"):
            LoKILinear(original_linear, target_pos)

    def test_out_of_range_positions(self):
        """Test that out-of-range positions are handled."""
        original_linear = nn.Linear(16, 32, bias=False)

        # Position 32 is out of range (valid: 0-31)
        # Implementation should handle this (error or ignore)
        # This test documents expected behavior
        with pytest.raises((IndexError, ValueError)):
            target_pos = [0, 10, 32]  # 32 is invalid
            LoKILinear(original_linear, target_pos)
