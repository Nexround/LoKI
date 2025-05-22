# LoKI: Low-damage Knowledge Implanting of Large Language Models

This Supplementary Material includes the following contents:
- **LoKI code**, which supports reproducing the experiments in the paper
- **ToolACE dataset**, which is a reformatted version of the [ToolACE dataset](https://huggingface.co/datasets/Team-ACE/ToolACE) publicly available on Hugging Face
- **Trainable node JSON files based on KVA results**. Due to file size limitations, the HDF5 files recording the KVA results are not included here
- **Appendix**

## Usage

Currently, our code does not enforce specific library versions. The training workflow is based on **Llama-Factory**. To execute training, we recommend installing Llama-Factory first, and then running the following commands to install the dependencies required by LoKI to avoid dependency conflicts.

```bash
cd LoKI
conda create -n loki python=3.10
pip install -e .
```

We do not provide detailed code explanations for the entire process—please refer to the workflow described in the paper.

### Running Analysing

The current code supports performing the KVA process on the Qwen2.5 and Llama series models. Execution commands are stored in the Makefile. You can run the KVA process with the following commands. To change related parameters, please check the Makefile.

**Note:** The current code does not support multi-GPU processing. For models of 8B and above, we recommend using GPUs with 80GB of memory. If you encounter memory issues, we provide a temporary degradation solution: in `analysing_mmlu_().py`, replace the `forward_with_partitioning` method with `forward_with_partitioning_single`.

For Qwen2.5:
```bash
make analysing_mmlu_qwen2
```

For Llama:
```bash
make analysing_mmlu_llama
```

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
