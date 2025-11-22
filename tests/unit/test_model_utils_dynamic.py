import importlib
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig

from loki.models import ArchitectureSpec
from loki.utils import model_utils


class DummyConfig(PretrainedConfig):
    model_type = "dummy_test"

    def __init__(self, target_pos=None, num_hidden_layers=2, **kwargs):
        self.target_pos = target_pos or [[] for _ in range(num_hidden_layers)]
        self.num_hidden_layers = num_hidden_layers
        super().__init__(model_type=self.model_type, **kwargs)

    @classmethod
    def from_pretrained(cls, *args, target_pos=None, **kwargs):
        target_pos = target_pos or kwargs.get(
            "target_pos", [[] for _ in range(kwargs.get("num_hidden_layers", 2))]
        )
        return cls(target_pos=target_pos, num_hidden_layers=len(target_pos))


class FakeMLP(nn.Module):
    def __init__(self, hidden_size=8):
        super().__init__()
        self.down_proj = nn.Linear(hidden_size, hidden_size, bias=False)


class FakeLayer(nn.Module):
    def __init__(self, hidden_size=8):
        super().__init__()
        self.mlp = FakeMLP(hidden_size=hidden_size)


class FakeModel(PreTrainedModel):
    config_class = DummyConfig

    def __init__(self, config):
        super().__init__(config)
        hidden = 8
        self.config = config
        self.model = types.SimpleNamespace(
            layers=nn.ModuleList([FakeLayer(hidden) for _ in range(config.num_hidden_layers)])
        )
        self.lm_head = nn.Linear(hidden, 10, bias=False)

    @classmethod
    def from_pretrained(cls, *args, config=None, **kwargs):
        cfg = config or DummyConfig()
        return cls(cfg)

    def save_pretrained(self, save_directory, *args, **kwargs):  # type: ignore[override]
        Path(save_directory).mkdir(parents=True, exist_ok=True)
        (Path(save_directory) / "fake.bin").write_text("weights")
        mod = sys.modules.get(self.__class__.__module__)
        if mod is not None and getattr(mod, "__file__", None):
            src = Path(mod.__file__)
            dst = Path(save_directory) / src.name
            if src.exists() and src.resolve() != dst.resolve():
                dst.write_text(src.read_text())


def _make_fake_module():
    mod_name = "fake_base_module_for_tests"
    mod = types.ModuleType(mod_name)
    DummyConfig.__module__ = mod_name
    FakeModel.__module__ = mod_name
    mod.DummyConfig = DummyConfig
    mod.FakeModel = FakeModel
    sys.modules[mod_name] = mod
    return mod_name


def _build_spec(mod_name: str):
    return ArchitectureSpec(
        name="Dummy",
        model_type="dummy_test_dynamic",
        base_model_cls=FakeModel,
        base_config_cls=DummyConfig,
        mlp_getter=lambda model, idx: model.model.layers[idx].mlp,
        down_proj_getter=lambda model, idx: model.model.layers[idx].mlp.down_proj,
    )


def test_generated_source_is_self_contained():
    mod_name = _make_fake_module()
    spec = _build_spec(mod_name)
    source, model_cls_name, config_cls_name = model_utils._build_loki_module_source(spec)
    assert "from loki" not in source
    assert model_cls_name in source
    assert config_cls_name in source
    assert spec.base_model_cls.__module__ in source


def test_materialize_builds_loki_linear_and_freezes(tmp_path):
    mod_name = _make_fake_module()
    spec = _build_spec(mod_name)
    loki_model_cls, loki_config_cls = model_utils._materialize_loki_classes(spec, tmp_path)

    config = loki_config_cls(target_pos=[[0, 1], [1, 2]])
    model = loki_model_cls(config)

    layers = model.model.layers
    assert type(layers[0].mlp.down_proj).__name__ == "LoKILinear"
    assert layers[0].mlp.down_proj.active.weight.requires_grad is True
    assert layers[0].mlp.down_proj.frozen.weight.requires_grad is False
    # Base params like lm_head should be frozen
    assert model.lm_head.weight.requires_grad is False
    # Generated module file exists for custom_object_save copying
    generated_path = tmp_path / "_generated" / "loki_modeling.py"
    assert generated_path.exists()


def test_create_loki_model_uses_generated_module(tmp_path, monkeypatch):
    mod_name = _make_fake_module()
    spec = _build_spec(mod_name)
    monkeypatch.setattr(model_utils, "get_architecture_spec", lambda identifier: spec)

    # Stub AutoModel/AutoTokenizer
    class _DummyTokenizer:
        def save_pretrained(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "tokenizer.json").write_text("tok")

    monkeypatch.setattr(model_utils.AutoModel, "from_pretrained", FakeModel.from_pretrained)
    monkeypatch.setattr(
        model_utils, "AutoTokenizer", types.SimpleNamespace(from_pretrained=lambda *a, **k: _DummyTokenizer())
    )

    target_pos = [[0, 1], [1, 2]]
    pos_path = tmp_path / "pos.json"
    import json

    pos_path.write_text(json.dumps(target_pos))

    model_utils.create_loki_model(
        model_name="dummy-id",
        target_pos_path=pos_path,
        save_dir=tmp_path / "out",
        torch_dtype=torch.float32,
    )

    gen_module = tmp_path / "out" / "_generated" / "loki_modeling.py"
    saved_module = tmp_path / "out" / "loki_modeling.py"
    assert gen_module.exists()
    assert saved_module.exists()

    spec_saved = importlib.util.spec_from_file_location("loaded_loki_module", saved_module)
    module = importlib.util.module_from_spec(spec_saved)
    spec_saved.loader.exec_module(module)  # type: ignore[attr-defined]
    LokiConfig = getattr(module, "LoKIDummyConfig")
    LokiModel = getattr(module, "LoKIDummyForCausalLM")
    cfg = LokiConfig(target_pos=target_pos)
    model = LokiModel(cfg)
    assert type(model.model.layers[0].mlp.down_proj).__name__ == "LoKILinear"
    assert model.model.layers[0].mlp.down_proj.active.weight.requires_grad
    assert not model.lm_head.weight.requires_grad


def test_lokilinear_copies_original_weights(tmp_path):
    # Build a tiny model with deterministic weights/biases to verify cloning.
    import torch

    class BiasConfig(PretrainedConfig):
        model_type = "dummy_copy_check"

        def __init__(self, target_pos=None, num_hidden_layers=1, **kwargs):
            self.target_pos = target_pos or [[] for _ in range(num_hidden_layers)]
            self.num_hidden_layers = num_hidden_layers
            super().__init__(model_type=self.model_type, **kwargs)

        @classmethod
        def from_pretrained(cls, *args, target_pos=None, **kwargs):
            target_pos = target_pos or kwargs.get(
                "target_pos", [[] for _ in range(kwargs.get("num_hidden_layers", 1))]
            )
            return cls(target_pos=target_pos, num_hidden_layers=len(target_pos))

    class BiasMLP(nn.Module):
        def __init__(self, hidden_size=4):
            super().__init__()
            self.down_proj = nn.Linear(hidden_size, hidden_size, bias=True)
            # deterministic weights/bias for checking copies
            w = torch.arange(hidden_size * hidden_size).float().view(hidden_size, hidden_size)
            b = torch.arange(hidden_size).float()
            with torch.no_grad():
                self.down_proj.weight.copy_(w)
                self.down_proj.bias.copy_(b)
            self._init_weight = w.clone()
            self._init_bias = b.clone()

    class BiasLayer(nn.Module):
        def __init__(self, hidden_size=4):
            super().__init__()
            self.mlp = BiasMLP(hidden_size=hidden_size)

    class BiasModel(PreTrainedModel):
        config_class = BiasConfig

        def __init__(self, config):
            super().__init__(config)
            hidden = 4
            self.config = config
            self.model = types.SimpleNamespace(
                layers=nn.ModuleList([BiasLayer(hidden) for _ in range(config.num_hidden_layers)])
            )
            self.lm_head = nn.Linear(hidden, 2, bias=False)

        @classmethod
        def from_pretrained(cls, *args, config=None, **kwargs):
            cfg = config or BiasConfig()
            return cls(cfg)

    mod_name = "fake_bias_module_for_tests"
    mod = types.ModuleType(mod_name)
    BiasConfig.__module__ = mod_name
    BiasModel.__module__ = mod_name
    mod.BiasConfig = BiasConfig
    mod.BiasModel = BiasModel
    sys.modules[mod_name] = mod

    spec = ArchitectureSpec(
        name="Bias",
        model_type="dummy_copy_check",
        base_model_cls=BiasModel,
        base_config_cls=BiasConfig,
        mlp_getter=lambda model, idx: model.model.layers[idx].mlp,
        down_proj_getter=lambda model, idx: model.model.layers[idx].mlp.down_proj,
    )

    loki_model_cls, loki_config_cls = model_utils._materialize_loki_classes(spec, tmp_path)
    target_pos = [[0, 2]]  # active rows
    config = loki_config_cls(target_pos=target_pos)
    model = loki_model_cls(config)

    mlp = model.model.layers[0].mlp
    loki_layer = mlp.down_proj
    expected_w = mlp._init_weight
    expected_b = mlp._init_bias

    torch.testing.assert_close(loki_layer.active.weight, expected_w[target_pos[0]])
    torch.testing.assert_close(loki_layer.frozen.weight, expected_w[[1, 3]])
    torch.testing.assert_close(loki_layer.active_bias, expected_b[target_pos[0]])
    torch.testing.assert_close(loki_layer.frozen_bias, expected_b[[1, 3]])
