# LoKI: Low-damage Knowledge Implanting - AI Coding Agent Guide

## Project Overview
LoKI implements a selective fine-tuning method for LLMs that identifies and trains only specific "knowledge-bearing" neurons in MLP down-projection layers, minimizing damage to existing knowledge while implanting new capabilities. The workflow: **Analyse (KVA)** → **Select (Layer-Balanced Strategy)** → **Implant (Training)**.

## ⚠️ Critical: UV Environment Requirement
**ALL operations in this repository MUST be performed within the UV environment.** This project uses `uv` for dependency management to ensure reproducible builds and avoid conflicts.

### Initial Setup
```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies (run this in project root)
cd /workspace/LoKI
uv sync

# Run any Python command within UV environment
uv run python <script.py>

# Or activate the virtual environment
source .venv/bin/activate  # Then run commands normally
```

### Running Commands with UV
- **KVA Analysis**: `uv run accelerate launch analyse_mmlu_llama.py --model_path ...`
- **Python Scripts**: Always prefix with `uv run` or activate `.venv` first
- **Jupyter Notebooks**: Use `uv run jupyter lab` to ensure correct kernel
- **LlamaFactory Training**: Install LlamaFactory in UV env, then use `uv run` for training commands

## Architecture & Key Components

### Three-Phase Pipeline
1. **KVA (Knowledge-Value Attribution)**: Uses integrated gradients to identify which neurons in MLP down-projection layers are most critical for specific knowledge domains
2. **Selection**: Applies Layer-Balanced Strategy to choose trainable neurons from KVA results, generating position JSON files
3. **Implanting**: Fine-tunes only selected neurons using LlamaFactory while freezing everything else

### Core Model Classes
- **`LoKILinear`** (`src/loki/loki_linear.py`): Custom Linear layer that splits weights into `active` (trainable) and `frozen` (fixed) portions based on neuron positions. Uses pre-computed index mapping for efficient reordering during forward pass.
- **`LoKILlamaForCausalLM`/`LoKIQwenForCausalLM`** (`src/loki/loki_llama_model.py`, `loki_qwen_model.py`): Modified Transformers models that replace MLP `down_proj` layers with LoKILinear. Config requires `target_pos` list matching layer count.
- **`KVALlamaForCausalLM`/`KVAQwenForCausalLM`** (`src/loki/kva/kva_llama.py`, `kva_qwen2.py`): Analysis models for computing integrated gradients. Only `down_proj.weight` requires gradients. Implements `forward_with_partitioning()` (default, memory-intensive) and `forward_with_partitioning_single()` (fallback for 80GB+ models).

### Configuration Pattern
LoKI configs extend base model configs (e.g., `LlamaConfig`) with `target_pos` field:
```python
target_pos: List[List[int]]  # One list of neuron indices per layer
```

## Critical Workflows

### Running KVA Analysis
```bash
# Ensure UV environment is active first
source .venv/bin/activate

# Qwen models
make analysing_mmlu_qwen2

# Llama models  
make analysing_mmlu_llama

# Or run directly with uv:
uv run accelerate launch --mixed_precision bf16 analyse_mmlu_llama.py --model_path meta-llama/Llama-3.1-8B-Instruct ...
```
- **Single GPU only** - no multi-GPU support
- **Memory**: 80GB GPU recommended for 8B+ models; use `forward_with_partitioning_single()` fallback if OOM
- Outputs: HDF5 file in `kva_result/hdf5/<model_name>/kva_mmlu.h5` containing integrated gradient tensors (shape: `[num_samples, num_layers, hidden_dim]`)
- Edit `analyse_mmlu_llama.py` or `analyse_mmlu_qwen2.py` to switch methods if memory constrained

### Generating LoKI Models
Use `utils.ipynb` with `create_loki_model()`:
```python
from src.loki import LoKILlamaForCausalLM, LoKILlamaConfig, create_loki_model

create_loki_model(
    loki_model_class=LoKILlamaForCausalLM,
    loki_config_cls=LoKILlamaConfig,
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    target_pos_path="kva_result/pos_json/Llama-3.1-8B-Instruct/10.json",
    save_dir="./loki_model_llama_10",
    torch_dtype=torch.bfloat16
)
```
- Saves **both** LoKI model and original weights (LoKILinear reconstructs split on each load)
- Pre-generated position JSONs available in `kva_result/pos_json/`

### Training with LlamaFactory
1. Modify `lf_config/<dataset>/train.sh` to set LlamaFactory path
2. Update `lf_config/<dataset>/<dataset>.yaml`:
   - Set `model_name_or_path` to LoKI model directory
   - Set `deepspeed` config path
   - Set `output_dir` for checkpoints
   - Fields marked with `()` require manual configuration
3. Run: `cd lf_config/<dataset> && bash train.sh`

Training shell script dynamically writes dataset_info.json to LlamaFactory's data directory before launching.

## Project Conventions

### File Organization
- `src/loki/`: Core LoKI implementations (model classes, config, linear layer)
- `src/loki/kva/`: KVA analysis models and HDF5 utilities
- `analyse_mmlu_*.py`: Entry points for KVA on MMLU benchmark (Llama/Qwen variants)
- `kva_result/pos_json/`: Pre-computed neuron position JSONs organized by `<model>/<percentage>.json`
- `lf_config/`: Training configurations per dataset (ToolACE, Reranker)

### Model Support
Currently supports **Llama** and **Qwen2.5** architectures. Adding new architectures requires:
1. Create `LoKI<Model>Config` extending base config with `target_pos`
2. Create `LoKI<Model>ForCausalLM` with `apply_loki_linear()` method
3. Create `KVA<Model>ForCausalLM` for analysis phase

### Dependency Management
**Must use UV environment for all operations.** Install LlamaFactory within UV env first, then run `uv sync` to install LoKI dependencies. No strict version pinning in `pyproject.toml`.

```bash
# Recommended installation order
uv pip install llama-factory  # Install LlamaFactory in UV env
uv sync                        # Install LoKI dependencies
```

### Memory Optimization
When encountering OOM during KVA:
1. Use 80GB+ GPU if available
2. Switch to `forward_with_partitioning_single()` in analysis scripts (single-layer gradient computation vs. parallel)

### Data Flow
MMLU → KVA Analysis (HDF5) → `analysing_utils.ipynb` (Layer-Balanced Selection) → Position JSON → `utils.ipynb` (LoKI Model Creation) → LlamaFactory Training

## Key Implementation Details

### LoKILinear Weight Splitting
Splits `out_features` dimension into active/frozen based on `target_pos`. Forward pass: compute both portions, concatenate, then reorder using buffered `index_map` to restore original neuron ordering.

### Integrated Gradients Implementation
`generate_partitioning()` creates linear interpolation between zero baseline and actual activation across `steps` steps. Gradients computed w.r.t. intermediate activations, then summed and scaled by step size.

### Freezing Strategy
All model parameters frozen except `down_proj.weight` in target layers. During training, only `active` portion of LoKILinear updates.
