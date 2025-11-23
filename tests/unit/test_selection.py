"""Unit tests for node selection strategies."""

import numpy as np
import pytest
from loki.selection import (
    select_trainable_nodes_global_highest,
    select_trainable_nodes_global_lowest,
    select_trainable_nodes_layer_balanced,
)


class TestLayerBalancedSelection:
    """Test layer-balanced selection strategy."""

    def test_basic_selection(self):
        """Test basic selection with simple attribution scores."""
        # Create simple test data: 10 inferences, 4 layers, 8 nodes each
        np.random.seed(42)
        attribution_scores = np.random.rand(10, 4, 8)
        quota = 50.0  # 50% of nodes

        positions = select_trainable_nodes_layer_balanced(attribution_scores, quota)

        # Should return list with 4 layers
        assert len(positions) == 4

        # Each layer should have approximately 50% of 8 = 4 nodes
        for layer_pos in positions:
            assert len(layer_pos) == 4

    def test_quota_scaling(self):
        """Test that different quotas produce different numbers of nodes."""
        np.random.seed(42)
        attribution_scores = np.random.rand(20, 8, 16)

        positions_10 = select_trainable_nodes_layer_balanced(attribution_scores, 10.0)
        positions_50 = select_trainable_nodes_layer_balanced(attribution_scores, 50.0)

        total_10 = sum(len(layer) for layer in positions_10)
        total_50 = sum(len(layer) for layer in positions_50)

        # Expected per-layer counts: int(num_nodes * quota / 100)
        num_layers, num_nodes = attribution_scores.shape[1], attribution_scores.shape[2]
        expected_10 = int(num_nodes * 0.10) * num_layers
        expected_50 = int(num_nodes * 0.50) * num_layers

        assert total_10 == expected_10
        assert total_50 == expected_50
        assert total_50 > total_10

    def test_equal_distribution_across_layers(self):
        """Test that layer-balanced strategy distributes equally."""
        np.random.seed(42)
        attribution_scores = np.random.rand(50, 10, 100)
        quota = 20.0

        positions = select_trainable_nodes_layer_balanced(attribution_scores, quota)

        # All layers should have same number of selected nodes
        node_counts = [len(layer_pos) for layer_pos in positions]
        assert len(set(node_counts)) == 1  # All counts should be identical


class TestGlobalLowestSelection:
    """Test global lowest selection strategy."""

    def test_basic_selection(self):
        """Test basic global lowest selection."""
        np.random.seed(42)
        attribution_scores = np.random.rand(10, 4, 8)
        quota = 25.0  # 25% of total nodes

        positions = select_trainable_nodes_global_lowest(attribution_scores, quota)

        # Should return list with 4 layers
        assert len(positions) == 4

        # Total selected should be approximately 25% of 32 (4*8) = 8
        total_selected = sum(len(layer_pos) for layer_pos in positions)
        expected = int(4 * 8 * 0.25)
        assert total_selected == expected

    def test_unequal_distribution(self):
        """Test that global strategy can produce unequal distribution."""
        np.random.seed(42)

        # Create biased attribution scores
        # Layer 0: very low scores, Layer 1: very high scores
        attribution_scores = np.zeros((20, 2, 10))
        attribution_scores[:, 0, :] = 0.1  # Layer 0: low values
        attribution_scores[:, 1, :] = 0.9  # Layer 1: high values

        positions = select_trainable_nodes_global_lowest(attribution_scores, 50.0)

        # Layer 0 (low scores) should have more selected nodes
        # Layer 1 (high scores) should have fewer selected nodes
        assert len(positions[0]) >= len(positions[1])


class TestGlobalHighestSelection:
    """Test global highest selection strategy."""

    def test_basic_selection(self):
        """Test basic global highest selection."""
        np.random.seed(42)
        attribution_scores = np.random.rand(10, 4, 8)
        quota = 25.0

        positions = select_trainable_nodes_global_highest(attribution_scores, quota)

        # Should return list with 4 layers
        assert len(positions) == 4

        # Total selected should be approximately 25% of total
        total_selected = sum(len(layer_pos) for layer_pos in positions)
        expected = int(4 * 8 * 0.25)
        assert total_selected == expected

    def test_opposite_of_lowest(self):
        """Test that highest selects opposite nodes from lowest."""
        np.random.seed(42)

        # Create clear gradient: layer 0 has lowest, layer 3 has highest
        attribution_scores = np.zeros((20, 4, 10))
        for layer_idx in range(4):
            attribution_scores[:, layer_idx, :] = layer_idx * 0.3

        positions_low = select_trainable_nodes_global_lowest(attribution_scores, 25.0)
        positions_high = select_trainable_nodes_global_highest(attribution_scores, 25.0)

        # Lowest should favor earlier layers, highest should favor later layers
        total_low_early = len(positions_low[0]) + len(positions_low[1])
        total_high_early = len(positions_high[0]) + len(positions_high[1])

        assert total_low_early > total_high_early


class TestSelectionEdgeCases:
    """Test edge cases for selection strategies."""

    def test_zero_quota(self):
        """Test behavior with very small quota."""
        np.random.seed(42)
        attribution_scores = np.random.rand(10, 4, 8)

        # Very small quota (< 1 node per layer)
        positions = select_trainable_nodes_layer_balanced(attribution_scores, 1.0)

        # Should still return valid structure
        assert len(positions) == 4

    def test_full_quota(self):
        """Test behavior with 100% quota."""
        np.random.seed(42)
        attribution_scores = np.random.rand(10, 4, 8)

        positions = select_trainable_nodes_layer_balanced(attribution_scores, 100.0)

        # Should select all or nearly all nodes
        total_selected = sum(len(layer_pos) for layer_pos in positions)
        total_nodes = 4 * 8

        # Should be close to 100%
        assert total_selected >= total_nodes * 0.95

    def test_single_inference(self):
        """Test with only one inference."""
        np.random.seed(42)
        attribution_scores = np.random.rand(1, 4, 8)  # Only 1 inference

        positions = select_trainable_nodes_layer_balanced(attribution_scores, 50.0)

        # Should still work with single inference
        assert len(positions) == 4

    def test_deterministic_behavior(self):
        """Test that selection is deterministic for same input."""
        np.random.seed(42)
        attribution_scores = np.random.rand(20, 8, 16)

        positions1 = select_trainable_nodes_layer_balanced(attribution_scores, 30.0)
        positions2 = select_trainable_nodes_layer_balanced(attribution_scores, 30.0)

        # Results should be identical
        assert positions1 == positions2
