"""Integration test for LoKI model creation and restoration workflow."""

import json
import os
import tempfile
from pathlib import Path

import pytest
import torch
from loki.models import (
    get_loki_config_class,
    get_loki_model_class,
    list_registered_architectures,
)

# Mark as integration test (requires model loading)
pytestmark = pytest.mark.integration


class TestLoKIModelWorkflow:
    """Integration tests for complete LoKI workflow."""

    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            yield {
                "pos_json": tmpdir / "positions.json",
                "loki_model": tmpdir / "loki_model",
                "restored_model": tmpdir / "restored_model",
            }

    @pytest.fixture
    def sample_positions(self):
        """Create sample position JSON for testing."""
        # For a small model with 12 layers, select a few nodes per layer
        return [[0, 10, 20, 30] for _ in range(12)]

    def test_position_json_format(self, temp_dirs, sample_positions):
        """Test that position JSON is correctly formatted."""
        pos_path = temp_dirs["pos_json"]

        # Save positions
        with open(pos_path, "w") as f:
            json.dump(sample_positions, f)

        # Load and verify
        with open(pos_path) as f:
            loaded = json.load(f)

        assert isinstance(loaded, list)
        assert len(loaded) == 12
        assert all(isinstance(layer, list) for layer in loaded)
        assert all(isinstance(idx, int) for layer in loaded for idx in layer)

    @pytest.mark.skipif(
        not os.environ.get("LOKI_TEST_MODEL_NAME")
        or not os.environ.get("LOKI_TEST_MODEL_TYPE"),
        reason="Requires actual model download - set LOKI_TEST_MODEL_NAME/TYPE",
    )
    def test_create_loki_model(self, temp_dirs, sample_positions):
        """Test creating LoKI model from a registered architecture (manual)."""
        from loki import create_loki_model

        model_name = os.environ.get("LOKI_TEST_MODEL_NAME")
        model_type = os.environ.get("LOKI_TEST_MODEL_TYPE")
        if not model_name or not model_type:
            pytest.skip("Set LOKI_TEST_MODEL_NAME and LOKI_TEST_MODEL_TYPE to run")
        if model_type not in list_registered_architectures():
            pytest.skip(f"Model type {model_type} not registered for tests")

        loki_model_class = get_loki_model_class(model_type)
        loki_config_class = get_loki_config_class(model_type)

        # Save positions
        pos_path = temp_dirs["pos_json"]
        with open(pos_path, "w") as f:
            json.dump(sample_positions, f)

        # Create LoKI model (this will download the model)
        create_loki_model(
            loki_model_class=loki_model_class,
            loki_config_cls=loki_config_class,
            model_name=model_name,
            target_pos_path=str(pos_path),
            save_dir=str(temp_dirs["loki_model"]),
            torch_dtype=torch.float16,
        )

        # Verify model was created
        assert temp_dirs["loki_model"].exists()
        assert (temp_dirs["loki_model"] / "config.json").exists()
        assert (temp_dirs["loki_model"] / "pytorch_model.bin").exists() or \
               (temp_dirs["loki_model"] / "model.safetensors").exists()

    @pytest.mark.skipif(
        not os.environ.get("LOKI_TEST_MODEL_NAME")
        or not os.environ.get("LOKI_TEST_MODEL_TYPE"),
        reason="Requires actual model - set LOKI_TEST_MODEL_NAME/TYPE",
    )
    def test_restore_loki_model(self, temp_dirs, sample_positions):
        """Test restoring LoKI model to original format."""
        from loki import create_loki_model, restore_loki_model

        model_name = os.environ.get("LOKI_TEST_MODEL_NAME")
        model_type = os.environ.get("LOKI_TEST_MODEL_TYPE")
        if not model_name or not model_type:
            pytest.skip("Set LOKI_TEST_MODEL_NAME and LOKI_TEST_MODEL_TYPE to run")
        if model_type not in list_registered_architectures():
            pytest.skip(f"Model type {model_type} not registered for tests")

        loki_model_class = get_loki_model_class(model_type)
        loki_config_class = get_loki_config_class(model_type)

        # Save positions
        pos_path = temp_dirs["pos_json"]
        with open(pos_path, "w") as f:
            json.dump(sample_positions, f)

        # Create LoKI model
        create_loki_model(
            loki_model_class=loki_model_class,
            loki_config_cls=loki_config_class,
            model_name=model_name,
            target_pos_path=str(pos_path),
            save_dir=str(temp_dirs["loki_model"]),
            torch_dtype=torch.bfloat16,
        )

        # Restore model
        restore_loki_model(
            model_path=str(temp_dirs["loki_model"]),
            target_pos_path=str(pos_path),
            output_path=str(temp_dirs["restored_model"]),
            model_name=model_name,
        )

        # Verify restored model
        assert temp_dirs["restored_model"].exists()
        assert (temp_dirs["restored_model"] / "config.json").exists()


class TestLoKIModelInference:
    """Test inference with LoKI models."""

    @pytest.mark.skip(reason="Requires actual model - run manually")
    def test_loki_model_forward_pass(self):
        """Test that LoKI model can perform forward pass."""
        from transformers import AutoModel, AutoTokenizer

        # Load a pre-created LoKI model (assumes it exists)
        model_path = "models/loki_test"  # Should be created in setup

        if not Path(model_path).exists():
            pytest.skip(f"Test model not found at {model_path}")

        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Simple forward pass
        text = "Hello, world!"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)

        # Verify output structure
        assert hasattr(outputs, "logits")
        assert outputs.logits.shape[0] == 1  # batch size
        assert outputs.logits.shape[1] == inputs.input_ids.shape[1]  # seq len
