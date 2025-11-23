# The current KVA implementation uses Captum for Integrated Gradients
# Single GPU only - no multi-GPU support (original script)

analysing_mmlu_qwen:
	CUDA_VISIBLE_DEVICES=0 \
	uv run accelerate launch \
		--mixed_precision bf16 \
		scripts/analyse_mmlu.py \
		--model_path Qwen/Qwen3-0.6B \
		--output_dir kva_result/hdf5/Qwen3-0.6B \
		--result_file kva_mmlu.h5 \
		--write_mode w \
		--batch_size 8 \

analysing_mmlu_llama:
	CUDA_VISIBLE_DEVICES=0 \
	uv run accelerate launch \
		--mixed_precision bf16 \
		scripts/analyse_mmlu.py \
		--model_path meta-llama/Llama-3.1-8B-Instruct \
		--output_dir kva_result/hdf5/Llama-3.1-8B-Instruct \
		--result_file kva_mmlu.h5 \
		--write_mode w \

# Parallel analysis with batch processing and multi-GPU support

# Single GPU with batch processing (4x faster)
analysing_mmlu_qwen_batch:
	CUDA_VISIBLE_DEVICES=0 \
	uv run python scripts/analyse_mmlu.py \
		--model_path Qwen/Qwen2.5-0.5B-Instruct \
		--output_dir kva_result/hdf5/Qwen2.5-0.5B-Instruct \
		--batch_size 16 \
		--result_file kva_mmlu_batch.h5 \
		--write_mode w \
		--use_flash_attention

analysing_mmlu_llama_batch:
	CUDA_VISIBLE_DEVICES=0 \
	uv run python scripts/analyse_mmlu.py \
		--model_path meta-llama/Llama-3.1-8B-Instruct \
		--output_dir kva_result/hdf5/Llama-3.1-8B-Instruct \
		--batch_size 4 \
		--result_file kva_mmlu_batch.h5 \
		--write_mode w \
		--use_flash_attention

# Multi-GPU data parallelism (samples distributed across GPUs)
analysing_mmlu_qwen_data_parallel:
	uv run accelerate launch \
		--num_processes 4 \
		--mixed_precision bf16 \
		scripts/analyse_mmlu.py \
		--model_path Qwen/Qwen2.5-0.5B-Instruct \
		--output_dir kva_result/hdf5/Qwen2.5-0.5B-Instruct \
		--batch_size 8 \
		--parallel_mode data \
		--result_file kva_mmlu_data_parallel.h5 \
		--write_mode w \
		--use_flash_attention

analysing_mmlu_llama_data_parallel:
	uv run accelerate launch \
		--num_processes 4 \
		--mixed_precision bf16 \
		scripts/analyse_mmlu.py \
		--model_path meta-llama/Llama-3.1-8B-Instruct \
		--output_dir kva_result/hdf5/Llama-3.1-8B-Instruct \
		--batch_size 2 \
		--parallel_mode data \
		--result_file kva_mmlu_data_parallel.h5 \
		--write_mode w \
		--use_flash_attention

# Multi-GPU model parallelism (model layers across GPUs)
analysing_mmlu_llama_model_parallel:
	uv run python scripts/analyse_mmlu.py \
		--model_path meta-llama/Llama-3.1-8B-Instruct \
		--output_dir kva_result/hdf5/Llama-3.1-8B-Instruct \
		--batch_size 4 \
		--parallel_mode model \
		--result_file kva_mmlu_model_parallel.h5 \
		--write_mode w \
		--use_flash_attention

# Quick test with limited samples (for debugging)
test_analyse_qwen:
	CUDA_VISIBLE_DEVICES=0 \
	uv run python scripts/analyse_mmlu.py \
		--model_path Qwen/Qwen2.5-0.5B-Instruct \
		--output_dir kva_result/hdf5/test \
		--batch_size 8 \
		--result_file test_qwen.h5 \
		--write_mode w \
		--max_samples_per_subset 10

test_analyse_llama:
	CUDA_VISIBLE_DEVICES=0 \
	uv run python scripts/analyse_mmlu.py \
		--model_type llama \
		--model_path meta-llama/Llama-3.1-8B-Instruct \
		--output_dir kva_result/hdf5/test \
		--steps 7 \
		--batch_size 4 \
		--result_file test_llama.h5 \
		--write_mode w \
		--max_samples_per_subset 10

# Node selection
select_nodes:
	uv run python scripts/select_trainable_nodes.py \
		--hdf5_path $(HDF5_PATH) \
		--quota $(QUOTA) \
		--strategy $(STRATEGY) \
		--output_dir $(OUTPUT_DIR)

# Create LoKI model
create_loki_model:
	uv run python scripts/create_loki_model.py \
		--model_type $(MODEL_TYPE) \
		--model_name $(MODEL_NAME) \
		--target_pos_path $(POS_PATH) \
		--save_dir $(SAVE_DIR)

# Restore model
restore_model:
	uv run python scripts/restore_loki_model.py \
		--model_path $(MODEL_PATH) \
		--target_pos_path $(POS_PATH) \
		--output_path $(OUTPUT_PATH) \
		--model_name $(MODEL_NAME)

# Run tests
test:
	uv run pytest tests/unit/ -v

test_cov:
	uv run pytest tests/unit/ --cov=src/loki --cov-report=html

# Code quality
format:
	uv run black src/ scripts/ tests/

lint:
	uv run ruff check src/ scripts/

typecheck:
	uv run mypy src/

# Pre-commit hooks
pre-commit-install:
	uv run pre-commit install

pre-commit-run:
	uv run pre-commit run --all-files

pre-commit-update:
	uv run pre-commit autoupdate
