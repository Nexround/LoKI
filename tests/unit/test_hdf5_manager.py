"""Unit tests for HDF5Manager class."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from loki.utils.hdf5_manager import HDF5Manager


class TestHDF5ManagerInit:
    """Test HDF5Manager initialization."""

    def test_init_valid_modes(self):
        """Test initialization with valid modes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.h5"

            # Write mode
            manager = HDF5Manager(filepath, mode='w')
            assert manager.mode == 'w'
            assert manager.filepath == filepath

            # Append mode
            manager = HDF5Manager(filepath, mode='a')
            assert manager.mode == 'a'

    def test_init_invalid_mode(self):
        """Test initialization with invalid mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.h5"
            with pytest.raises(ValueError, match="Invalid mode"):
                HDF5Manager(filepath, mode='x')

    def test_init_read_nonexistent_file(self):
        """Test read mode with nonexistent file."""
        with pytest.raises(FileNotFoundError):
            HDF5Manager("/nonexistent/file.h5", mode='r')


class TestHDF5ManagerCreateDataset:
    """Test dataset creation with metadata."""

    def test_create_dataset_basic(self):
        """Test basic dataset creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.h5"
            manager = HDF5Manager(filepath, mode='w')

            manager.create_dataset_with_metadata(
                shape=(0, 32, 4096),
                dtype=np.float16
            )

            # Verify dataset was created
            shape = manager.get_shape()
            assert shape == (0, 32, 4096)
            assert manager.get_dtype() == np.float16

    def test_create_dataset_with_metadata(self):
        """Test dataset creation with metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.h5"
            manager = HDF5Manager(filepath, mode='w')

            metadata = {
                "model_name": "example/model",
                "model_type": "example_arch",
                "num_layers": 32,
                "hidden_size": 4096,
                "ig_steps": 7,
                "ig_method": "riemann_trapezoid",
            }

            manager.create_dataset_with_metadata(
                shape=(0, 32, 4096),
                metadata=metadata
            )

            # Verify metadata was stored
            loaded_metadata = manager.read_metadata()
            assert loaded_metadata["model_name"] == metadata["model_name"]
            assert loaded_metadata["model_type"] == metadata["model_type"]
            assert loaded_metadata["num_layers"] == metadata["num_layers"]
            assert loaded_metadata["ig_steps"] == metadata["ig_steps"]
            assert "created_at" in loaded_metadata  # Auto-added timestamp

    def test_create_dataset_readonly_error(self):
        """Test that creating dataset in read-only mode raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.h5"

            # Create file first
            manager = HDF5Manager(filepath, mode='w')
            manager.create_dataset_with_metadata(shape=(0, 10, 20))

            # Try to create in read mode
            manager_ro = HDF5Manager(filepath, mode='r')
            with pytest.raises(RuntimeError, match="read-only mode"):
                manager_ro.create_dataset_with_metadata(shape=(0, 10, 20))


class TestHDF5ManagerAppendData:
    """Test data appending functionality."""

    def test_append_numpy_array(self):
        """Test appending NumPy arrays."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.h5"
            manager = HDF5Manager(filepath, mode='w')
            manager.create_dataset_with_metadata(shape=(0, 4, 8))

            # Append data
            data1 = np.random.rand(4, 8).astype(np.float16)
            data2 = np.random.rand(4, 8).astype(np.float16)

            manager.append_data(data1)
            manager.append_data(data2)

            # Verify
            loaded = manager.read_dataset()
            assert loaded.shape == (2, 4, 8)
            np.testing.assert_array_almost_equal(loaded[0], data1, decimal=3)
            np.testing.assert_array_almost_equal(loaded[1], data2, decimal=3)

    def test_append_torch_tensor(self):
        """Test appending PyTorch tensors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.h5"
            manager = HDF5Manager(filepath, mode='w')
            manager.create_dataset_with_metadata(shape=(0, 4, 8))

            # Append tensor
            tensor = torch.randn(4, 8)
            manager.append_data(tensor)

            # Verify
            loaded = manager.read_dataset()
            assert loaded.shape == (1, 4, 8)

    def test_append_list_of_tensors(self):
        """Test appending list of tensors (stacked)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.h5"
            manager = HDF5Manager(filepath, mode='w')
            manager.create_dataset_with_metadata(shape=(0, 4, 8))

            # Append list of tensors
            tensors = [torch.randn(8) for _ in range(4)]
            manager.append_data(tensors)

            # Verify shape
            loaded = manager.read_dataset()
            assert loaded.shape == (1, 4, 8)

    def test_append_shape_mismatch_error(self):
        """Test error when appending data with wrong shape."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.h5"
            manager = HDF5Manager(filepath, mode='w')
            manager.create_dataset_with_metadata(shape=(0, 4, 8))

            # Try to append wrong shape
            wrong_data = np.random.rand(4, 16)  # Wrong size
            with pytest.raises(ValueError, match="shape mismatch"):
                manager.append_data(wrong_data)

    def test_append_without_dataset_error(self):
        """Test error when appending without creating dataset first."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.h5"
            manager = HDF5Manager(filepath, mode='w')

            # Try to append without creating dataset
            data = np.random.rand(4, 8)
            with pytest.raises(RuntimeError, match="does not exist"):
                manager.append_data(data)


class TestHDF5ManagerReadData:
    """Test data reading functionality."""

    def test_read_full_dataset(self):
        """Test reading complete dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.h5"
            manager = HDF5Manager(filepath, mode='w')
            manager.create_dataset_with_metadata(shape=(0, 4, 8))

            # Write data
            data = np.random.rand(10, 4, 8).astype(np.float16)
            for i in range(10):
                manager.append_data(data[i])

            # Read all
            loaded = manager.read_dataset()
            assert loaded.shape == (10, 4, 8)

    def test_read_dataset_slice(self):
        """Test reading dataset slice."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.h5"
            manager = HDF5Manager(filepath, mode='w')
            manager.create_dataset_with_metadata(shape=(0, 4, 8))

            # Write data
            for _i in range(10):
                manager.append_data(np.random.rand(4, 8))

            # Read slice
            subset = manager.read_dataset(start=2, end=5)
            assert subset.shape == (3, 4, 8)

    def test_read_specific_layer(self):
        """Test reading specific layer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.h5"
            manager = HDF5Manager(filepath, mode='w')
            manager.create_dataset_with_metadata(shape=(0, 4, 8))

            # Write data
            for _i in range(10):
                manager.append_data(np.random.rand(4, 8))

            # Read specific layer
            layer_2 = manager.read_dataset(layer_idx=2)
            assert layer_2.shape == (10, 8)

    def test_read_nonexistent_dataset_error(self):
        """Test error when reading from file without dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.h5"
            manager = HDF5Manager(filepath, mode='w')
            # Don't create dataset

            # Create empty file
            import h5py
            with h5py.File(filepath, 'w'):
                pass

            with pytest.raises(KeyError, match="not found"):
                manager.read_dataset()


class TestHDF5ManagerMetadata:
    """Test metadata operations."""

    def test_update_metadata(self):
        """Test updating metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.h5"
            manager = HDF5Manager(filepath, mode='w')

            # Create with initial metadata
            initial_meta = {"version": "1.0", "author": "test"}
            manager.create_dataset_with_metadata(
                shape=(0, 4, 8),
                metadata=initial_meta
            )

            # Update metadata
            manager.update_metadata({
                "version": 2.0,  # Will be stored as number, not string
                "processed_samples": 100
            })

            # Verify
            meta = manager.read_metadata()
            assert meta["version"] == 2.0
            assert meta["processed_samples"] == 100
            assert meta["author"] == "test"  # Still there

    def test_metadata_with_complex_types(self):
        """Test metadata with dict and list values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.h5"
            manager = HDF5Manager(filepath, mode='w')

            metadata = {
                "config": {"lr": 0.001, "epochs": 10},
                "layers": [1, 2, 3, 4, 5],
                "name": "test_model"
            }

            manager.create_dataset_with_metadata(
                shape=(0, 4, 8),
                metadata=metadata
            )

            # Verify complex types are preserved
            loaded_meta = manager.read_metadata()
            assert loaded_meta["config"] == metadata["config"]
            assert loaded_meta["layers"] == metadata["layers"]
            assert loaded_meta["name"] == metadata["name"]


class TestHDF5ManagerMerge:
    """Test file merging functionality."""

    def test_merge_basic(self):
        """Test basic file merging."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            input_dir.mkdir()
            output_file = Path(tmpdir) / "merged.h5"

            # Create input files
            for i in range(3):
                filepath = input_dir / f"file{i}.h5"
                manager = HDF5Manager(filepath, mode='w')
                manager.create_dataset_with_metadata(shape=(0, 4, 8))
                for _ in range(5):
                    manager.append_data(np.random.rand(4, 8))

            # Merge
            HDF5Manager.merge_files(input_dir, output_file)

            # Verify
            manager = HDF5Manager(output_file, mode='r')
            assert manager.get_shape() == (15, 4, 8)  # 3 files * 5 samples

    def test_merge_custom_pattern(self):
        """Test merging with custom file pattern."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            input_dir.mkdir()
            output_file = Path(tmpdir) / "merged.h5"

            # Create .hdf5 files
            for i in range(2):
                filepath = input_dir / f"data{i}.hdf5"
                manager = HDF5Manager(filepath, mode='w')
                manager.create_dataset_with_metadata(shape=(0, 4, 8))
                manager.append_data(np.random.rand(4, 8))

            # Merge with pattern
            HDF5Manager.merge_files(input_dir, output_file, pattern="*.hdf5")

            # Verify
            manager = HDF5Manager(output_file, mode='r')
            assert manager.get_shape()[0] == 2


class TestHDF5ManagerRepr:
    """Test string representation."""

    def test_repr(self):
        """Test __repr__ method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.h5"
            manager = HDF5Manager(filepath, mode='w')

            repr_str = repr(manager)
            assert "HDF5Manager" in repr_str
            assert str(filepath) in repr_str
            assert "mode=w" in repr_str
