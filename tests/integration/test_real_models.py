"""
Integration sanity checks against real HF model configs to ensure registry
resolves/auto-registers architectures from real-world identifiers.
Uses accelerate.init_empty_weights to avoid downloading model weights.
"""

import pytest
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModel

from loki.core import BaseKVAModel, BaseLoKIModel
from loki.models import get_architecture_spec, get_kva_model_class, get_loki_model_class

REAL_MODEL_IDS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen3-VL-2B-Instruct",
    "Qwen/Qwen3-0.6B",
    "meta-llama/Llama-3.1-8B-Instruct",
]


@pytest.mark.integration
@pytest.mark.parametrize("model_id", REAL_MODEL_IDS)
def test_registry_resolves_real_models(model_id):
    """Ensure registry can resolve/auto-register using real HF configs."""
    config = AutoConfig.from_pretrained(model_id)

    # Instantiate model weights-free to ensure module structure is accessible
    with init_empty_weights():
        AutoModel.from_config(config)

    spec = get_architecture_spec(config)  # triggers auto registration if needed
    assert spec.model_type == config.model_type

    loki_cls = get_loki_model_class(config.model_type)
    kva_cls = get_kva_model_class(config.model_type)

    assert issubclass(loki_cls, BaseLoKIModel)
    assert issubclass(kva_cls, BaseKVAModel)
