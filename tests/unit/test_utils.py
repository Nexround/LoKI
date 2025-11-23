"""Unit tests for HDF5 and JSON utilities."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from loki.selection.utils import (
    load_attributions_from_hdf5,
    merge_hdf5_files,
    save_positions_to_json,
)
from loki.utils.hdf5_manager import HDF5Manager


class TestHDF5Loading:
    """Test HDF5 loading functionality."""

    def test_load_valid_hdf5(self):
        """Test loading from valid HDF5 file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            hdf5_path = Path(tmpdir) / "test.h5"

            # Create test HDF5 file using HDF5Manager
            test_data = np.random.rand(10, 4, 8)
            manager = HDF5Manager(hdf5_path, mode="w")
            manager.create_dataset_with_metadata(shape=(0, 4, 8))
            for i in range(10):
                manager.append_data(test_data[i])

            # Load and verify
            loaded_data = load_attributions_from_hdf5(hdf5_path)
            np.testing.assert_array_almost_equal(loaded_data, test_data, decimal=3)

    def test_load_nonexistent_file(self):
        """Test error handling for nonexistent file."""
        with pytest.raises(FileNotFoundError):
            load_attributions_from_hdf5(Path("/nonexistent/path.h5"))

    def test_load_missing_dataset_key(self):
        """Test error handling when 'dataset' key is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            hdf5_path = Path(tmpdir) / "test.h5"

            # Create HDF5 without proper dataset using raw h5py
            # This test verifies error handling for corrupted files
            import h5py

            with h5py.File(hdf5_path, "w") as f:
                f.create_dataset("wrong_key", data=np.random.rand(10, 4, 8))

            # Should raise KeyError
            with pytest.raises(KeyError):
                load_attributions_from_hdf5(hdf5_path)


class TestJSONSaving:
    """Test JSON saving functionality."""

    def test_save_positions(self):
        """Test saving positions to JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "positions.json"
            positions = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

            save_positions_to_json(positions, output_path)

            # Verify file was created
            assert output_path.exists()

            # Verify content
            with open(output_path) as f:
                loaded = json.load(f)
            assert loaded == positions

    def test_save_with_directory_creation(self):
        """Test that parent directories are created automatically."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "subdir1" / "subdir2" / "positions.json"
            positions = [[0, 1], [2, 3]]

            save_positions_to_json(positions, output_path, create_dirs=True)

            assert output_path.exists()

    def test_save_empty_positions_error(self):
        """Test that saving empty positions raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "positions.json"

            with pytest.raises(ValueError):
                save_positions_to_json([], output_path)


class TestHDF5Merging:
    """Test HDF5 file merging functionality."""

    def test_merge_multiple_files(self):
        """Test merging multiple HDF5 files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            input_dir.mkdir()
            output_file = Path(tmpdir) / "merged.h5"

            # Create multiple HDF5 files using HDF5Manager
            data1 = np.random.rand(5, 4, 8).astype(np.float16)
            data2 = np.random.rand(7, 4, 8).astype(np.float16)
            data3 = np.random.rand(3, 4, 8).astype(np.float16)

            for i, data in enumerate([data1, data2, data3], 1):
                filepath = input_dir / f"file{i}.h5"
                manager = HDF5Manager(filepath, mode="w")
                manager.create_dataset_with_metadata(shape=(0, 4, 8))
                for sample in data:
                    manager.append_data(sample)

            # Merge files
            merge_hdf5_files(input_dir, output_file)

            # Verify merged file
            manager = HDF5Manager(output_file, mode="r")
            merged_data = manager.read_dataset()
            assert merged_data.shape == (15, 4, 8)  # 5+7+3 = 15

    def test_merge_with_existing_output(self):
        """Test appending to existing output file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            input_dir.mkdir()
            output_file = Path(tmpdir) / "merged.h5"

            # Create initial output file
            initial_data = np.random.rand(2, 4, 8).astype(np.float16)
            manager = HDF5Manager(output_file, mode="w")
            manager.create_dataset_with_metadata(shape=(0, 4, 8))
            for sample in initial_data:
                manager.append_data(sample)

            # Create input files
            data1 = np.random.rand(3, 4, 8).astype(np.float16)
            filepath = input_dir / "file1.h5"
            manager = HDF5Manager(filepath, mode="w")
            manager.create_dataset_with_metadata(shape=(0, 4, 8))
            for sample in data1:
                manager.append_data(sample)

            # Merge
            merge_hdf5_files(input_dir, output_file)

            # Verify
            manager = HDF5Manager(output_file, mode="r")
            merged_data = manager.read_dataset()
            assert merged_data.shape == (5, 4, 8)  # 2+3 = 5

    def test_merge_shape_mismatch_error(self):
        """Test error handling for shape mismatch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            input_dir.mkdir()
            output_file = Path(tmpdir) / "merged.h5"

            # Create files with different shapes
            data1 = np.random.rand(5, 4, 8).astype(np.float16)
            data2 = np.random.rand(5, 4, 16).astype(np.float16)  # Different!

            for i, (data, shape) in enumerate([(data1, (0, 4, 8)), (data2, (0, 4, 16))], 1):
                filepath = input_dir / f"file{i}.h5"
                manager = HDF5Manager(filepath, mode="w")
                manager.create_dataset_with_metadata(shape=shape)
                for sample in data:
                    manager.append_data(sample)

            # Should raise ValueError
            with pytest.raises(ValueError, match="Incompatible dataset"):
                merge_hdf5_files(input_dir, output_file)

    def test_merge_no_files_error(self):
        """Test error when no HDF5 files found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "empty"
            input_dir.mkdir()
            output_file = Path(tmpdir) / "merged.h5"

            with pytest.raises(ValueError, match="No files matching"):
                merge_hdf5_files(input_dir, output_file)
