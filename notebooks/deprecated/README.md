# Deprecated Notebooks

⚠️ **These notebooks are deprecated and kept only for reference.**

## Migration Guide

### `analysing_utils.ipynb` → CLI Scripts

**Old workflow** (notebook cells):
```python
# Load HDF5
with h5py.File(hdf5_path, "r") as f:
    attribution_scores = np.array(f["dataset"])

# Select nodes
result = select_trainable_nodes(attribution_scores, QUOTA)

# Save JSON
with open(result_path, "w") as f:
    json.dump(result, f)
```

**New workflow** (command line):
```bash
# Layer-balanced strategy
python scripts/select_trainable_nodes.py \
    --hdf5_path kva_result/hdf5/model/kva_mmlu.h5 \
    --quota 10 \
    --output_dir kva_result/pos_json/model

# Global lowest strategy
python scripts/select_trainable_nodes.py \
    --hdf5_path kva_result/hdf5/model/kva_mmlu.h5 \
    --quota 30 \
    --strategy global_lowest

# Global highest strategy
python scripts/select_trainable_nodes.py \
    --hdf5_path kva_result/hdf5/model/kva_mmlu.h5 \
    --quota 30 \
    --strategy global_highest
```

**Python API**:
```python
from src.loki.selection import (
    select_trainable_nodes_layer_balanced,
    select_trainable_nodes_global_lowest,
    select_trainable_nodes_global_highest,
    load_attributions_from_hdf5,
    save_positions_to_json,
)

# Load scores
scores = load_attributions_from_hdf5("kva_result/hdf5/model/kva_mmlu.h5")

# Select nodes
positions = select_trainable_nodes_layer_balanced(scores, quota=10.0)

# Save results
save_positions_to_json(positions, "kva_result/pos_json/model/10.json")
```

---

### `utils.ipynb` → CLI Scripts

**Old workflow** (notebook cells):
```python
from src.loki import create_loki_model, LoKILlamaForCausalLM, LoKILlamaConfig

create_loki_model(
    loki_model_class=LoKILlamaForCausalLM,
    loki_config_cls=LoKILlamaConfig,
    target_pos_path="./kva_result/model/10.json",
    save_dir="./models/loki",
    model_name="meta-llama/Llama-3.1-8B-Instruct",
)
```

**New workflow** (command line):
```bash
# Create LoKI model
python scripts/create_loki_model.py \
    --model_type llama \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --target_pos_path kva_result/pos_json/model/10.json \
    --save_dir models/loki_model

# Restore model
python scripts/restore_loki_model.py \
    --model_path models/loki_model \
    --target_pos_path kva_result/pos_json/model/10.json \
    --output_path models/restored \
    --model_name meta-llama/Llama-3.1-8B-Instruct
```

**Python API** (still available):
```python
from src.loki import create_loki_model, restore_loki_model
from src.loki.models.llama import LoKILlamaForCausalLM, LoKILlamaConfig

# Create model
create_loki_model(
    loki_model_class=LoKILlamaForCausalLM,
    loki_config_cls=LoKILlamaConfig,
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    target_pos_path="kva_result/pos_json/model/10.json",
    save_dir="models/loki_model",
)

# Restore model
restore_loki_model(
    model_path="models/loki_model",
    target_pos_path="kva_result/pos_json/model/10.json",
    output_path="models/restored",
    model_name="meta-llama/Llama-3.1-8B-Instruct",
)
```

---

## See Also

- **Selection strategies**: `src/loki/selection/`
- **Model utilities**: `src/loki/utils/model_utils.py`
- **CLI tools**: `scripts/`
- **Full guide**: `REFACTORING_SUMMARY_CN.md`
