"""Unit tests for helper routines in scripts/analyse_mmlu.py."""

import types

import torch

from scripts.analyse_mmlu import build_conversation, collate_fn, process_batch
from loki.utils.hdf5_manager import HDF5Manager


def test_build_conversation_includes_subject_and_choices():
    conversation = build_conversation(
        "college_biology",
        {"question": "What is DNA?", "choices": ["A", "B", "C", "D"]},
    )

    assert conversation[0]["role"] == "user"
    content = conversation[0]["content"]
    assert "College Biology" in content
    assert "Choices:" in content
    assert "\nA. A" in content
    assert content.strip().endswith("Answer:")


def test_collate_fn_pads_batches():
    batch = [
        {
            "input_ids": torch.tensor([1, 2]),
            "attention_mask": torch.tensor([1, 1]),
            "subset": "set_a",
            "answer": "A",
        },
        {
            "input_ids": torch.tensor([3, 4, 5]),
            "attention_mask": torch.tensor([1, 1, 1]),
            "subset": "set_b",
            "answer": "B",
        },
    ]

    collated = collate_fn(batch)

    assert collated["input_ids"].shape == (2, 3)
    assert collated["attention_mask"].shape == (2, 3)
    # Pads with zeros on the shorter entry.
    assert collated["input_ids"][0, -1].item() == 0
    assert collated["subsets"] == ["set_a", "set_b"]
    assert collated["answers"] == ["A", "B"]


def test_process_batch_writes_hdf5_and_cleans(tmp_path):
    class FakeOutputs:
        def __init__(self, logits):
            self.logits = logits

    class FakeModel:
        def __init__(self):
            self.integrated_gradients = []
            self.clean_calls = 0

        def __call__(self, input_ids, attention_mask):
            batch_size = input_ids.shape[0]
            vocab = 4
            logits = torch.stack(
                [torch.arange(vocab, dtype=torch.float32) for _ in range(batch_size)]
            ).unsqueeze(1)
            return FakeOutputs(logits)

        def compute_integrated_gradients(
            self,
            input_ids,
            attention_mask,
            target_token_idx,
            predicted_label,
            steps,
            method,
        ):
            # Two layers, three nodes each.
            self.integrated_gradients = [
                torch.tensor([0.0, 1.0, 2.0]),
                torch.tensor([3.0, 4.0, 5.0]),
            ]

        def clean(self):
            self.clean_calls += 1
            self.integrated_gradients = []

    batch = {
        "input_ids": torch.tensor([[1, 2], [3, 4]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1], [1, 1]], dtype=torch.long),
    }

    manager = HDF5Manager(tmp_path / "out.h5", mode="w")
    manager.create_dataset_with_metadata(shape=(0, 2, 3))

    args = types.SimpleNamespace()

    model = FakeModel()
    process_batch(
        model=model,
        batch=batch,
        device=torch.device("cpu"),
        args=args,
        hdf5_manager=manager,
        ig_steps=3,
        ig_method="trapezoid",
        is_model_parallel=False,
    )

    data = manager.read_dataset()
    assert data.shape == (2, 2, 3)  # Two samples appended.
    assert str(manager.get_dtype()) == "float16"
    # Clean should have been called after each sample in the batch.
    assert model.clean_calls == 2
