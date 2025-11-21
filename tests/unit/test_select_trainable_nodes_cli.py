"""Tests for scripts/select_trainable_nodes CLI wrapper."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from loki.utils.hdf5_manager import HDF5Manager


@pytest.fixture
def simple_hdf5(tmp_path: Path) -> Path:
    """Create a tiny HDF5 file with deterministic scores."""
    hdf5_path = tmp_path / "kva_result" / "hdf5" / "DemoModel" / "kva.h5"
    hdf5_path.parent.mkdir(parents=True, exist_ok=True)

    manager = HDF5Manager(hdf5_path, mode="w")
    manager.create_dataset_with_metadata(shape=(0, 2, 4))

    # Single inference, two layers.
    scores = np.array(
        [
            [
                [0.1, 0.2, 0.3, 0.4],
                [0.4, 0.3, 0.2, 0.1],
            ]
        ],
        dtype=np.float32,
    )
    manager.append_data(scores[0])
    return hdf5_path


def test_select_trainable_nodes_default_output(monkeypatch, tmp_path, simple_hdf5):
    """End-to-end call of CLI with auto output dir/name and layer-balanced strategy."""
    from scripts import select_trainable_nodes as cli

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "select_trainable_nodes",
            "--hdf5_path",
            str(simple_hdf5),
            "--quota",
            "50",
        ],
    )

    cli.main()

    output_path = tmp_path / "kva_result" / "pos_json" / "DemoModel" / "50.json"
    with open(output_path) as f:
        positions = json.load(f)

    # Layer-balanced with deterministic scores should pick lowest two per layer.
    assert len(positions) == 2
    assert sorted(positions[0]) == [0, 1]
    assert sorted(positions[1]) == [2, 3]


def test_select_trainable_nodes_custom_output(monkeypatch, tmp_path):
    """CLI should honor explicit strategy, output dir, and output filename."""
    from scripts import select_trainable_nodes as cli

    hdf5_path = tmp_path / "inputs" / "scores.h5"
    hdf5_path.parent.mkdir(parents=True, exist_ok=True)
    manager = HDF5Manager(hdf5_path, mode="w")
    manager.create_dataset_with_metadata(shape=(0, 1, 5))
    manager.append_data(np.array([[0.05, 0.2, 0.4, 0.6, 0.8]], dtype=np.float32))

    output_dir = tmp_path / "outputs"
    output_name = "custom.json"

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "select_trainable_nodes",
            "--hdf5_path",
            str(hdf5_path),
            "--quota",
            "40",
            "--strategy",
            "global_lowest",
            "--output_dir",
            str(output_dir),
            "--output_name",
            output_name,
        ],
    )

    cli.main()

    with open(output_dir / output_name) as f:
        positions = json.load(f)

    # 40% of 5 nodes -> 2 lowest indices should be selected.
    assert len(positions) == 1
    assert sorted(positions[0]) == [0, 1]
