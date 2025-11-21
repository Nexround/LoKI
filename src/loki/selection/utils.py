"""Utility functions for node selection."""

import json
from pathlib import Path

import numpy as np

from loki.utils.hdf5_manager import HDF5Manager


def load_attributions_from_hdf5(hdf5_path: Path) -> np.ndarray:
    """Load attribution scores from HDF5 file.

    Args:
        hdf5_path: Path to HDF5 file containing attribution scores

    Returns:
        3D numpy array of shape (num_inferences, num_layers, num_nodes)

    Raises:
        FileNotFoundError: If HDF5 file does not exist
        KeyError: If 'dataset' key not found in HDF5 file

    Example:
        >>> scores = load_attributions_from_hdf5("kva_result/hdf5/model/kva_mmlu.h5")
        >>> print(scores.shape)  # (100, 32, 4096)
    """
    manager = HDF5Manager(hdf5_path, mode='r')
    return manager.read_dataset()


def save_positions_to_json(
    positions: list[list[int]], output_path: Path, create_dirs: bool = True
) -> None:
    """Save selected node positions to JSON file.

    Args:
        positions: List of lists containing node indices for each layer
        output_path: Path to save JSON file
        create_dirs: If True, create parent directories if they don't exist

    Raises:
        ValueError: If positions list is empty
    """
    if not positions:
        raise ValueError("Positions list cannot be empty")

    output_path = Path(output_path)

    if create_dirs:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(positions, f, ensure_ascii=False, indent=4)

    print(f"JSON file saved to: {output_path}")


def merge_hdf5_files(input_dir: Path, output_file: Path) -> None:
    """Merge multiple HDF5 files into a single file.

    This function is useful for combining KVA results from parallel GPU execution.
    All input files must have the same dataset structure (shape[1:] and dtype).

    Args:
        input_dir: Directory containing HDF5 files to merge
        output_file: Path to output merged HDF5 file

    Raises:
        ValueError: If no HDF5 files found, missing 'dataset' key, or shape/dtype mismatch

    Example:
        >>> merge_hdf5_files(
        ...     Path("kva_result/hdf5/partial/"),
        ...     Path("kva_result/hdf5/merged.h5")
        ... )
        Successfully merged 4 files into kva_result/hdf5/merged.h5
    """
    # Use HDF5Manager's static method
    HDF5Manager.merge_files(input_dir, output_file)
