# LoKI: Low-damage Knowledge Implanting of Large Language Models


## Usage

This project uses **uv** for dependency management. The training workflow is based on **Llama-Factory**. 

### Setup

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
cd LoKI
uv sync

# Install Llama-Factory in the UV environment (for training)
uv pip install llama-factory
```

We do not provide detailed code explanations for the entire process—please refer to the workflow described in the paper.

### Running Analysing

The current code uses **Captum's LayerIntegratedGradients** for the KVA process and supports Qwen2.5 and Llama series models. Execution commands are stored in the Makefile.

**Note:** The current code does not support multi-GPU processing. All commands should be run within the UV environment.

For Qwen2.5:
```bash
make analysing_mmlu_qwen2
```

For Llama:
```bash
make analysing_mmlu_llama
```

You can customize integration parameters (steps, method) by modifying the Makefile or running directly:
```bash
uv run accelerate launch --mixed_precision bf16 analyse_mmlu_llama.py \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --output_dir kva_result/hdf5/Llama-3.1-8B-Instruct \
    --steps 7 \
    --result_file kva_mmlu.h5 \
    --write_mode w \
    --ig_method riemann_trapezoid
```

Supported integration methods: `riemann_trapezoid` (default), `gausslegendre`, `riemann_left`, `riemann_right`, `riemann_middle`.

After execution, an HDF5 file recording the KVA results will be generated in the corresponding directory.

## Running Selecting

Use the generated HDF5 file in `selecting_utils.ipynb` to perform the **Layer-Balanced Strategy** and generate trainable node JSON files for training. Note that we have already provided these JSON files in the `kva_result/pos_json` directory.

## Running Implanting

We use **Llama-Factory** to perform the training. Before training, please prepare the following:

1. Use `utils.ipynb` with the trainable node JSON files to generate the **LoKI model**, i.e., a model where target Linear layers are replaced with LoKI Linear layers.
2. Modify the training configuration files under the `lf_config` folder for the corresponding dataset (bash scripts and YAML files), setting the paths and training parameters accordingly. Fields that need manual modification are marked with `()` in the YAML file. In the bash script, you need to modify the path to Llama-Factory.

After completing the steps above, run the following commands to start training:

```bash
cd lf_config/(dataset_name)
bash train.sh
```
