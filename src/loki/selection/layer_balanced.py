"""Layer-balanced node selection strategy for LoKI.

This module implements the layer-balanced strategy which distributes
trainable nodes equally across all layers based on attribution scores.
"""


import numpy as np


def split_n(num_layers: int) -> list[float]:
    """Split the total quota into equal parts across all layers.

    Args:
        num_layers: Number of layers to distribute quota across

    Returns:
        List of equal fractions (1/num_layers for each layer)
    """
    return [1.0 / num_layers for _ in range(num_layers)]


def select_trainable_nodes_layer_balanced(
    attribution_scores: np.ndarray, quota: float
) -> list[list[int]]:
    """Select trainable nodes using layer-balanced strategy.

    This strategy:
    1. Distributes the quota equally across all layers
    2. For each inference, normalizes attribution scores per layer
    3. Selects nodes with lowest normalized scores in each layer
    4. Counts selection frequency across all inferences
    5. Returns most frequently selected nodes per layer

    Args:
        attribution_scores: 3D array of shape (num_inferences, num_layers, num_nodes)
                          containing attribution scores for each node
        quota: Percentage of parameters to train (value between 0-100)

    Returns:
        List of lists containing indices of selected nodes for each layer.
        Length equals num_layers, each sublist contains selected node indices.

    Example:
        >>> scores = np.random.rand(100, 32, 4096)  # 100 inferences, 32 layers, 4096 nodes
        >>> positions = select_trainable_nodes_layer_balanced(scores, quota=10.0)
        >>> len(positions)  # 32 layers
        32
        >>> len(positions[0])  # ~128 nodes per layer (10% of 4096)
        128
    """
    num_inferences, num_layers, num_nodes = attribution_scores.shape

    # Calculate total number of trainable nodes based on quota
    num_trainable = num_layers * num_nodes * quota / 100

    # Calculate how many nodes to select per layer (equal distribution)
    spindle_parts = split_n(num_layers)
    k_per_layer = [int(x * num_trainable) for x in spindle_parts]
    print(f"Number of nodes to select per layer: {k_per_layer}")

    # Initialize matrix to count node selections across inferences
    node_counts = np.zeros((num_layers, num_nodes), dtype=int)

    # Process each inference and layer
    for infer_idx in range(num_inferences):
        for layer_idx in range(num_layers):
            # Get attribution scores for current layer
            layer_grad = attribution_scores[infer_idx, layer_idx, :]

            # Apply min-max normalization
            min_val = np.min(layer_grad)
            max_val = np.max(layer_grad)
            if max_val == min_val:
                normalized = np.zeros_like(layer_grad)
            else:
                normalized = (layer_grad - min_val) / (max_val - min_val)

            # Select indices of nodes with lowest normalized scores
            smallest_indices = np.argsort(normalized)[: k_per_layer[layer_idx]]
            node_counts[layer_idx, smallest_indices] += 1

    # Select final trainable nodes for each layer
    result = []
    for layer_idx in range(num_layers):
        # Get and sort node selection counts for current layer
        counts = node_counts[layer_idx]
        sorted_indices = np.argsort(counts)[::-1]  # Sort in descending order

        # Select top k_per_layer nodes
        selected_indices = sorted_indices[: k_per_layer[layer_idx]]
        result.append(selected_indices.tolist())

    return result
