# The current KVA implementation code does not support multiple GPUs

# Original implementations
analysing_mmlu_qwen2:
	CUDA_VISIBLE_DEVICES=0 \
	accelerate launch \
		--mixed_precision bf16 \
		mmlu_analyse_hdf5.py \
		--model_path Qwen/Qwen2.5-0.5B-Instruct \
		--output_dir kva_result/hdf5/Qwen2.5-0.5B-Instruct \
		--max_seq_length 32768 \
		--steps 7 \
		--result_file kva_mmlu.h5 \
		--write_mode w

analysing_mmlu_llama:
	CUDA_VISIBLE_DEVICES=0 \
	accelerate launch \
		--mixed_precision bf16 \
		mmlu_analyse_hdf5.py \
		--model_path meta-llama/Llama-3.1-8B-Instruct \
		--output_dir kva_result/hdf5/Llama-3.1-8B-Instruct \
		--max_seq_length 32768 \
		--steps 7 \
		--result_file kva_mmlu.h5 \
		--write_mode w

# Captum-based implementations (recommended)
analysing_mmlu_qwen2_captum:
	CUDA_VISIBLE_DEVICES=0 \
	uv run accelerate launch \
		--mixed_precision bf16 \
		analyse_mmlu_qwen2_captum.py \
		--model_path Qwen/Qwen2.5-0.5B-Instruct \
		--output_dir kva_result/hdf5/Qwen2.5-0.5B-Instruct \
		--steps 7 \
		--result_file kva_mmlu_captum.h5 \
		--write_mode w \
		--ig_method riemann_trapezoid

analysing_mmlu_llama_captum:
	CUDA_VISIBLE_DEVICES=0 \
	uv run accelerate launch \
		--mixed_precision bf16 \
		analyse_mmlu_llama_captum.py \
		--model_path /cache/models/official_models/Meta-Llama-3.1-8B-Instruct \
		--output_dir kva_result/hdf5/Llama-3.1-8B-Instruct \
		--steps 7 \
		--result_file kva_mmlu_captum.h5 \
		--write_mode w \
		--ig_method riemann_trapezoid
