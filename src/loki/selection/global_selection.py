"""Global node selection strategies for LoKI.

This module implements global selection strategies that select nodes
across all layers based on global attribution score ranking.
"""


import numpy as np


def select_trainable_nodes_global_lowest(
    attribution_scores: np.ndarray,
    quota: float
) -> list[list[int]]:
    """Select trainable nodes with globally lowest attribution scores.

    This strategy:
    1. Applies global min-max normalization across all layers
    2. Selects nodes with lowest scores globally (not per-layer)
    3. Distributes selected nodes back to their respective layers

    Args:
        attribution_scores: 3D array of shape (num_inferences, num_layers, num_nodes)
                          containing attribution scores for each node
        quota: Percentage of parameters to train (value between 0-100)

    Returns:
        List of lists containing indices of selected nodes for each layer.
        Distribution of nodes per layer is not uniform.

    Example:
        >>> scores = np.random.rand(100, 32, 4096)
        >>> positions = select_trainable_nodes_global_lowest(scores, quota=10.0)
        >>> total_selected = sum(len(layer_pos) for layer_pos in positions)
        >>> total_selected  # Approximately 10% of total nodes
        13107
    """
    num_inferences, num_layers, num_nodes = attribution_scores.shape
    total_nodes = num_layers * num_nodes
    total_trainable = int(total_nodes * quota / 100)

    # Initialize node selection count matrix
    node_counts = np.zeros((num_layers, num_nodes), dtype=int)

    # Process each inference
    for infer_idx in range(num_inferences):
        # Get and flatten all layer gradients for current inference
        all_layers_grad = attribution_scores[infer_idx, :, :]
        flattened_grad = all_layers_grad.flatten()

        # Apply global min-max normalization
        min_val = np.min(flattened_grad)
        max_val = np.max(flattened_grad)
        if max_val == min_val:
            normalized = np.zeros_like(flattened_grad)
        else:
            normalized = (flattened_grad - min_val) / (max_val - min_val)

        # Select indices of nodes with lowest global normalized scores
        smallest_global_indices = np.argsort(normalized)[:total_trainable]

        # Update node selection counts
        for global_idx in smallest_global_indices:
            layer_idx = global_idx // num_nodes
            node_idx = global_idx % num_nodes
            node_counts[layer_idx, node_idx] += 1

    # Select nodes with highest selection counts
    flattened_counts = node_counts.flatten()
    sorted_global_indices = np.argsort(flattened_counts, kind='stable')[::-1][:total_trainable]

    # Organize results by layer
    result = [[] for _ in range(num_layers)]
    for global_idx in sorted_global_indices:
        layer_idx = global_idx // num_nodes
        node_idx = global_idx % num_nodes
        result[layer_idx].append(int(node_idx))

    return result


def select_trainable_nodes_global_highest(
    attribution_scores: np.ndarray,
    quota: float
) -> list[list[int]]:
    """Select trainable nodes with globally highest attribution scores.

    This strategy is identical to global_lowest except it selects nodes
    with the HIGHEST attribution scores instead of lowest.

    Args:
        attribution_scores: 3D array of shape (num_inferences, num_layers, num_nodes)
                          containing attribution scores for each node
        quota: Percentage of parameters to train (value between 0-100)

    Returns:
        List of lists containing indices of selected nodes for each layer.
        Distribution of nodes per layer is not uniform.

    Example:
        >>> scores = np.random.rand(100, 32, 4096)
        >>> positions = select_trainable_nodes_global_highest(scores, quota=10.0)
        >>> total_selected = sum(len(layer_pos) for layer_pos in positions)
        >>> total_selected  # Approximately 10% of total nodes
        13107
    """
    num_inferences, num_layers, num_nodes = attribution_scores.shape
    total_nodes = num_layers * num_nodes
    total_trainable = int(total_nodes * quota / 100)

    # Initialize node selection count matrix
    node_counts = np.zeros((num_layers, num_nodes), dtype=int)

    # Process each inference
    for infer_idx in range(num_inferences):
        # Get and flatten all layer gradients for current inference
        all_layers_grad = attribution_scores[infer_idx, :, :]
        flattened_grad = all_layers_grad.flatten()

        # Apply global min-max normalization
        min_val = np.min(flattened_grad)
        max_val = np.max(flattened_grad)
        if max_val == min_val:
            normalized = np.zeros_like(flattened_grad)
        else:
            normalized = (flattened_grad - min_val) / (max_val - min_val)

        # Select indices of nodes with HIGHEST normalized scores
        largest_global_indices = np.argsort(-normalized)[:total_trainable]  # Negative for descending

        # Update node selection counts
        for global_idx in largest_global_indices:
            layer_idx = global_idx // num_nodes
            node_idx = global_idx % num_nodes
            node_counts[layer_idx, node_idx] += 1

    # Select nodes with highest selection counts
    flattened_counts = node_counts.flatten()
    sorted_global_indices = np.argsort(flattened_counts, kind='stable')[::-1][:total_trainable]

    # Organize results by layer
    result = [[] for _ in range(num_layers)]
    for global_idx in sorted_global_indices:
        layer_idx = global_idx // num_nodes
        node_idx = global_idx % num_nodes
        result[layer_idx].append(int(node_idx))

    return result
