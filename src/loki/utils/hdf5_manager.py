"""HDF5 Manager for LoKI project - Encapsulated HDF5 access with metadata support.

This module provides a unified interface for all HDF5 operations in LoKI,
ensuring consistency and preventing direct h5py usage throughout the codebase.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch


class HDF5Manager:
    """Manager class for HDF5 file operations with metadata support.

    This class encapsulates all HDF5 operations for the LoKI project, providing:
    - Safe read/write operations with proper resource management
    - Metadata storage for analysis provenance (model info, parameters, timestamps)
    - Incremental writing for large-scale KVA analysis
    - File merging capabilities for parallel computation results

    Key Design Principles:
    - All h5py operations are encapsulated within this class
    - Metadata is stored in HDF5 attributes for full traceability
    - File handles are properly managed (context managers)
    - Type hints and validation for robustness

    Example:
        >>> # Create and write attribution data
        >>> manager = HDF5Manager("kva_results.h5", mode='w')
        >>> metadata = {
        ...     "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        ...     "model_type": "llama",
        ...     "ig_steps": 7,
        ...     "ig_method": "riemann_trapezoid"
        ... }
        >>> manager.create_dataset_with_metadata(
        ...     shape=(0, 32, 4096),
        ...     dtype=np.float16,
        ...     metadata=metadata
        ... )
        >>>
        >>> # Append data incrementally
        >>> for sample_data in samples:
        ...     manager.append_data(sample_data)
        >>>
        >>> # Read data and metadata
        >>> data = manager.read_dataset()
        >>> meta = manager.read_metadata()
    """

    # Constants
    DATASET_NAME = "dataset"
    METADATA_PREFIX = "metadata_"

    def __init__(self, filepath: Path | str, mode: str = "r"):
        """Initialize HDF5 manager.

        Args:
            filepath: Path to HDF5 file
            mode: File access mode ('r', 'w', 'a')
                - 'r': Read-only (default)
                - 'w': Write (create new, overwrite if exists)
                - 'a': Append (create if not exists, otherwise append)

        Raises:
            ValueError: If mode is invalid
            FileNotFoundError: If mode is 'r' and file doesn't exist
        """
        self.filepath = Path(filepath)

        if mode not in ["r", "w", "a"]:
            raise ValueError(f"Invalid mode: {mode}. Must be 'r', 'w', or 'a'")

        if mode == "r" and not self.filepath.exists():
            raise FileNotFoundError(f"File not found: {self.filepath}")

        self.mode = mode

    def create_dataset_with_metadata(
        self,
        shape: tuple[int, ...],
        dtype: np.dtype | type = np.float16,
        metadata: dict[str, Any] | None = None,
        compression: str = "gzip",
        compression_opts: int = 4,
    ) -> None:
        """Create a new dataset with metadata.

        This method creates a resizable dataset suitable for incremental writing
        during KVA analysis. Metadata is stored as HDF5 attributes for full
        traceability of analysis parameters.

        Args:
            shape: Initial shape (first dimension should be 0 for empty dataset)
            dtype: Data type (default: float16 for space efficiency)
            metadata: Dictionary of metadata to store (model info, parameters, etc.)
            compression: Compression algorithm (default: gzip)
            compression_opts: Compression level (default: 4)

        Raises:
            ValueError: If dataset already exists in write mode
            RuntimeError: If file is opened in read-only mode

        Example:
            >>> metadata = {
            ...     "model_name": "meta-llama/Llama-3.1-8B-Instruct",
            ...     "model_type": "llama",
            ...     "num_layers": 32,
            ...     "hidden_size": 4096,
            ...     "ig_steps": 7,
            ...     "ig_method": "riemann_trapezoid",
            ...     "created_at": "2025-11-20T10:30:00"
            ... }
            >>> manager.create_dataset_with_metadata(
            ...     shape=(0, 32, 4096),
            ...     metadata=metadata
            ... )
        """
        if self.mode == "r":
            raise RuntimeError("Cannot create dataset in read-only mode")

        # Ensure parent directory exists
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

        # Add creation timestamp to metadata
        if metadata is None:
            metadata = {}

        if "created_at" not in metadata:
            metadata["created_at"] = datetime.now().isoformat()

        with h5py.File(self.filepath, self.mode) as f:
            if self.DATASET_NAME in f:
                if self.mode == "w":
                    raise ValueError(
                        f"Dataset '{self.DATASET_NAME}' already exists. "
                        "Use mode='a' to append or delete the file first."
                    )
                # In append mode, don't recreate
                return

            # Create resizable dataset
            maxshape = tuple(None if i == 0 else s for i, s in enumerate(shape))
            dset = f.create_dataset(
                self.DATASET_NAME,
                shape=shape,
                maxshape=maxshape,
                dtype=dtype,
                chunks=True,
                compression=compression,
                compression_opts=compression_opts,
            )

            # Store metadata as attributes
            for key, value in metadata.items():
                attr_key = f"{self.METADATA_PREFIX}{key}"
                # Convert complex types to JSON strings
                if isinstance(value, dict | list):
                    value = json.dumps(value)
                dset.attrs[attr_key] = value

    def append_data(self, data: np.ndarray | torch.Tensor | list[torch.Tensor]) -> None:
        """Append data to the dataset.

        This method handles incremental writing during KVA analysis. It automatically
        resizes the dataset and appends new data. Supports both NumPy arrays and
        PyTorch tensors.

        Args:
            data: Data to append. Can be:
                - NumPy array with shape matching dataset[1:]
                - PyTorch tensor (will be converted to NumPy)
                - List of PyTorch tensors (will be stacked)

        Raises:
            ValueError: If data shape doesn't match existing dataset
            RuntimeError: If dataset doesn't exist or file is read-only

        Example:
            >>> # Append single sample (list of layer tensors)
            >>> layer_tensors = [torch.randn(4096) for _ in range(32)]
            >>> manager.append_data(layer_tensors)
            >>>
            >>> # Append pre-stacked array
            >>> array = np.random.rand(32, 4096).astype(np.float16)
            >>> manager.append_data(array)
        """
        if self.mode == "r":
            raise RuntimeError("Cannot append data in read-only mode")

        # Convert PyTorch tensors to NumPy
        if isinstance(data, list):
            # Stack list of tensors
            data = torch.stack(data).cpu().numpy()
        elif isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        elif not isinstance(data, np.ndarray):
            raise TypeError(f"Unsupported data type: {type(data)}")

        # Convert to float16 for space efficiency
        if data.dtype != np.float16:
            data = data.astype(np.float16)

        with h5py.File(self.filepath, "a") as f:
            if self.DATASET_NAME not in f:
                raise RuntimeError(
                    f"Dataset '{self.DATASET_NAME}' does not exist. "
                    "Call create_dataset_with_metadata() first."
                )

            dset = f[self.DATASET_NAME]

            # Validate shape
            if data.shape != dset.shape[1:]:
                raise ValueError(
                    f"Data shape mismatch: expected {dset.shape[1:]}, got {data.shape}"
                )

            # Resize and append
            current_size = dset.shape[0]
            dset.resize(current_size + 1, axis=0)
            dset[current_size] = data

    def read_dataset(
        self,
        start: int | None = None,
        end: int | None = None,
        layer_idx: int | None = None,
    ) -> np.ndarray:
        """Read data from the dataset.

        Supports full or partial reading for memory efficiency with large files.

        Args:
            start: Starting sample index (None = from beginning)
            end: Ending sample index (None = to end)
            layer_idx: If specified, read only this layer (returns 2D array)

        Returns:
            NumPy array with requested data

        Raises:
            FileNotFoundError: If file doesn't exist
            KeyError: If dataset not found in file

        Example:
            >>> # Read all data
            >>> all_data = manager.read_dataset()  # shape: (N, L, H)
            >>>
            >>> # Read first 10 samples
            >>> subset = manager.read_dataset(start=0, end=10)
            >>>
            >>> # Read specific layer across all samples
            >>> layer_5 = manager.read_dataset(layer_idx=5)  # shape: (N, H)
        """
        with h5py.File(self.filepath, "r") as f:
            if self.DATASET_NAME not in f:
                raise KeyError(f"Dataset '{self.DATASET_NAME}' not found in {self.filepath}")

            dset = f[self.DATASET_NAME]

            if layer_idx is not None:
                # Read specific layer
                data = dset[start:end, layer_idx, :]
            else:
                # Read full dataset or slice
                data = dset[start:end]

            return np.array(data)

    def read_metadata(self) -> dict[str, Any]:
        """Read metadata from the dataset.

        Returns:
            Dictionary containing all metadata attributes

        Example:
            >>> metadata = manager.read_metadata()
            >>> print(f"Model: {metadata['model_name']}")
            >>> print(f"Steps: {metadata['ig_steps']}")
        """
        with h5py.File(self.filepath, "r") as f:
            if self.DATASET_NAME not in f:
                return {}

            dset = f[self.DATASET_NAME]
            metadata = {}

            for key in dset.attrs.keys():
                if key.startswith(self.METADATA_PREFIX):
                    # Remove prefix
                    clean_key = key[len(self.METADATA_PREFIX) :]
                    value = dset.attrs[key]

                    # Try to parse JSON strings
                    if isinstance(value, str):
                        try:
                            value = json.loads(value)
                        except (json.JSONDecodeError, TypeError):
                            pass  # Keep as string

                    metadata[clean_key] = value

            return metadata

    def get_shape(self) -> tuple[int, ...]:
        """Get dataset shape without loading data.

        Returns:
            Tuple representing dataset shape

        Example:
            >>> shape = manager.get_shape()  # (100, 32, 4096)
            >>> num_samples, num_layers, hidden_size = shape
        """
        with h5py.File(self.filepath, "r") as f:
            if self.DATASET_NAME not in f:
                raise KeyError(f"Dataset '{self.DATASET_NAME}' not found")
            return tuple(f[self.DATASET_NAME].shape)

    def get_dtype(self) -> np.dtype:
        """Get dataset data type.

        Returns:
            NumPy dtype of the dataset
        """
        with h5py.File(self.filepath, "r") as f:
            if self.DATASET_NAME not in f:
                raise KeyError(f"Dataset '{self.DATASET_NAME}' not found")
            return np.dtype(f[self.DATASET_NAME].dtype)

    def update_metadata(self, metadata: dict[str, Any]) -> None:
        """Update or add metadata attributes.

        Args:
            metadata: Dictionary of metadata to update/add

        Raises:
            RuntimeError: If file is read-only

        Example:
            >>> manager.update_metadata({
            ...     "num_samples_processed": 100,
            ...     "processing_completed_at": "2025-11-20T12:00:00"
            ... })
        """
        if self.mode == "r":
            raise RuntimeError("Cannot update metadata in read-only mode")

        with h5py.File(self.filepath, "a") as f:
            if self.DATASET_NAME not in f:
                raise RuntimeError(f"Dataset '{self.DATASET_NAME}' does not exist")

            dset = f[self.DATASET_NAME]

            for key, value in metadata.items():
                attr_key = f"{self.METADATA_PREFIX}{key}"
                if isinstance(value, dict | list):
                    value = json.dumps(value)
                dset.attrs[attr_key] = value

    @staticmethod
    def merge_files(
        input_dir: Path | str,
        output_file: Path | str,
        pattern: str = "*.h5",
    ) -> None:
        """Merge multiple HDF5 files into a single file.

        Useful for combining KVA results from parallel GPU execution.
        All input files must have compatible dataset structures.

        Args:
            input_dir: Directory containing HDF5 files to merge
            output_file: Path to output merged HDF5 file
            pattern: Glob pattern for input files (default: "*.h5")

        Raises:
            ValueError: If no files found, missing dataset, or incompatible structures

        Example:
            >>> HDF5Manager.merge_files(
            ...     input_dir="kva_result/hdf5/partial/",
            ...     output_file="kva_result/hdf5/merged.h5"
            ... )
        """
        input_dir = Path(input_dir)
        output_file = Path(output_file)

        # Collect input files
        input_files = sorted(input_dir.glob(pattern))

        # Exclude output file if it's in the same directory
        output_abspath = output_file.resolve()
        input_files = [f for f in input_files if f.resolve() != output_abspath]

        if not input_files:
            raise ValueError(f"No files matching '{pattern}' found in {input_dir}")

        # Validate all files and get reference shape/dtype
        ref_shape, ref_dtype = None, None
        total_samples = 0

        for fpath in input_files:
            with h5py.File(fpath, "r") as f:
                if HDF5Manager.DATASET_NAME not in f:
                    raise ValueError(f"File {fpath} missing '{HDF5Manager.DATASET_NAME}' dataset")

                dset = f[HDF5Manager.DATASET_NAME]

                if ref_shape is None:
                    ref_shape = dset.shape[1:]
                    ref_dtype = dset.dtype
                elif dset.shape[1:] != ref_shape or dset.dtype != ref_dtype:
                    raise ValueError(
                        f"Incompatible dataset in {fpath}: "
                        f"expected shape[1:]={ref_shape}, dtype={ref_dtype}, "
                        f"got shape[1:]={dset.shape[1:]}, dtype={dset.dtype}"
                    )

                total_samples += dset.shape[0]

        # Create or open output file
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(output_file, "a") as f_out:
            if HDF5Manager.DATASET_NAME in f_out:
                # Validate existing dataset
                existing_dset = f_out[HDF5Manager.DATASET_NAME]
                if existing_dset.shape[1:] != ref_shape or existing_dset.dtype != ref_dtype:
                    raise ValueError(
                        f"Output file has incompatible dataset: "
                        f"expected shape[1:]={ref_shape}, dtype={ref_dtype}"
                    )
                current_pos = existing_dset.shape[0]
                new_size = current_pos + total_samples
                existing_dset.resize((new_size,) + ref_shape)
            else:
                # Create new dataset
                ref_shape_tuple = ref_shape if ref_shape is not None else ()
                existing_dset = f_out.create_dataset(
                    HDF5Manager.DATASET_NAME,
                    shape=(total_samples,) + ref_shape_tuple,
                    maxshape=(None,) + ref_shape_tuple,
                    dtype=ref_dtype,
                    chunks=True,
                    compression="gzip",
                    compression_opts=4,
                )
                current_pos = 0

            # Copy data from each input file
            for fpath in input_files:
                with h5py.File(fpath, "r") as f_in:
                    dset_in = f_in[HDF5Manager.DATASET_NAME]
                    num_samples = dset_in.shape[0]
                    existing_dset[current_pos : current_pos + num_samples] = dset_in[:]
                    current_pos += num_samples

        print(f"Successfully merged {len(input_files)} files into {output_file}")
        print(f"Total samples: {total_samples}")

    def __repr__(self) -> str:
        """String representation of the manager."""
        exists = "exists" if self.filepath.exists() else "not found"
        return f"HDF5Manager(filepath={self.filepath}, mode={self.mode}, {exists})"
