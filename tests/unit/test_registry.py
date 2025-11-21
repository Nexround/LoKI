"""Tests for dynamic architecture registry and auto-registration."""

from types import SimpleNamespace
import importlib

import pytest
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
from loki import models as registry
from loki.models import registry as registry_module


class DummyConfig(PretrainedConfig):
    """Minimal config for dummy model."""

    model_type = "dummy"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = kwargs.get("vocab_size", 10)
        self.hidden_size = kwargs.get("hidden_size", 8)
        self.num_hidden_layers = kwargs.get("num_hidden_layers", 2)


class DummyModel(PreTrainedModel):
    """Minimal CausalLM-like model with mlp.down_proj structure."""

    config_class = DummyConfig

    def __init__(self, config: DummyConfig):
        super().__init__(config)
        self.model = SimpleNamespace(
            layers=[self._build_layer() for _ in range(config.num_hidden_layers)]
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

    def _build_layer(self):
        down_proj = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        mlp = SimpleNamespace(down_proj=down_proj)
        layer = SimpleNamespace(mlp=mlp)
        return layer

    def forward(self, *args, **kwargs):  # pragma: no cover - not used
        # Return a simple namespace mimicking HF output
        batch = kwargs.get("input_ids", torch.zeros(1, 1, dtype=torch.long)).shape[0]
        seq_len = kwargs.get("input_ids", torch.zeros(1, 1, dtype=torch.long)).shape[1]
        hidden = torch.zeros(batch, seq_len, self.config.hidden_size)
        return SimpleNamespace(last_hidden_state=hidden)


def test_auto_register_architecture(monkeypatch):
    """Auto-registration should create specs for unseen architectures."""

    # Isolate registry state
    monkeypatch.setattr(registry_module, "_REGISTRY", {}, raising=False)

    # Stub AutoConfig and AutoModel to return dummy classes
    monkeypatch.setattr(
        registry_module,
        "AutoConfig",
        SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: DummyConfig()),
    )
    monkeypatch.setattr(
        registry_module,
        "AutoModel",
        SimpleNamespace(from_config=lambda config: DummyModel(config)),
    )

    # Trigger auto-registration via get_loki_model_class
    loki_cls = registry_module.get_loki_model_class("dummy-checkpoint")
    kva_cls = registry_module.get_kva_model_class("dummy-checkpoint")

    # The classes should be generated and cached
    assert loki_cls.__name__.startswith("LoKIDummy")
    assert kva_cls.__name__.startswith("KVADummy")

    # Their configs should carry target_pos attribute
    cfg_cls = registry_module.get_loki_config_class("dummy-checkpoint")
    cfg = cfg_cls(target_pos=[[], []], hidden_size=8, num_hidden_layers=2)
    assert hasattr(cfg, "target_pos")
    assert cfg.target_pos == [[], []]


def test_register_architecture_manual(monkeypatch):
    """Manual registration should be honored and cached."""

    monkeypatch.setattr(registry_module, "_REGISTRY", {}, raising=False)

    spec = registry.ArchitectureSpec(
        name="Dummy",
        model_type="dummy",
        base_model_cls=DummyModel,
        base_config_cls=DummyConfig,
        mlp_getter=lambda m, i: m.model.layers[i].mlp,
        down_proj_getter=lambda m, i: m.model.layers[i].mlp.down_proj,
    )
    registry.register_architecture(spec)

    loki_cls = registry_module.get_loki_model_class("dummy")
    kva_cls = registry_module.get_kva_model_class("dummy")
    cfg_cls = registry_module.get_loki_config_class("dummy")

    assert loki_cls.__name__ == "LoKIDummyForCausalLM"
    assert kva_cls.__name__ == "KVADummyForCausalLM"
    assert cfg_cls.__name__ == "LoKIDummyConfig"
