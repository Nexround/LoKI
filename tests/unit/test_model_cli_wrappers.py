"""Lightweight tests for CLI wrappers around model creation/restoration scripts."""

import sys
from pathlib import Path

import torch


class DummyModel:
    """Placeholder class returned by registry stubs."""


def test_create_loki_model_cli_invokes_library(monkeypatch, tmp_path):
    from scripts import create_loki_model as cli

    target_pos = tmp_path / "positions.json"
    target_pos.write_text("[[0, 1]]")

    called = {}

    def fake_create_loki_model(**kwargs):
        called.update(kwargs)

    monkeypatch.setattr(cli, "create_loki_model", fake_create_loki_model)
    monkeypatch.setattr(cli, "get_loki_model_class", lambda ident: DummyModel)
    monkeypatch.setattr(cli, "get_loki_config_class", lambda ident: DummyModel)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "create_loki_model",
            "--model_type",
            "dummy-arch",
            "--model_name",
            "dummy/name",
            "--target_pos_path",
            str(target_pos),
            "--save_dir",
            str(tmp_path / "out"),
            "--torch_dtype",
            "bfloat16",
            "--trust_remote_code",
        ],
    )

    cli.main()

    assert called["loki_model_class"] is DummyModel
    assert called["loki_config_cls"] is DummyModel
    assert called["model_name"] == "dummy/name"
    assert called["target_pos_path"] == str(target_pos)
    assert called["save_dir"] == str(tmp_path / "out")
    assert called["torch_dtype"] == torch.bfloat16
    assert called["trust_remote_code"] is True


def test_restore_loki_model_cli_invokes_library(monkeypatch, tmp_path):
    from scripts import restore_loki_model as cli

    model_path = tmp_path / "loki_model"
    model_path.mkdir()
    (model_path / "config.json").write_text("{}")

    target_pos = tmp_path / "positions.json"
    target_pos.write_text("[[0]]")

    called = {}

    def fake_restore_loki_model(**kwargs):
        called.update(kwargs)

    monkeypatch.setattr(cli, "restore_loki_model", fake_restore_loki_model)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "restore_loki_model",
            "--model_path",
            str(model_path),
            "--target_pos_path",
            str(target_pos),
            "--output_path",
            str(tmp_path / "restored"),
            "--model_name",
            "base/name",
        ],
    )

    cli.main()

    assert called["model_path"] == str(model_path)
    assert called["target_pos_path"] == str(target_pos)
    assert called["output_path"] == str(tmp_path / "restored")
    assert called["model_name"] == "base/name"
