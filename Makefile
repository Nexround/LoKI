# The current KVA implementation uses Captum for Integrated Gradients
# Single GPU only - no multi-GPU support

analysing_mmlu_qwen2:
	CUDA_VISIBLE_DEVICES=0 \
	uv run accelerate launch \
		--mixed_precision bf16 \
		analyse_mmlu_qwen2.py \
		--model_path Qwen/Qwen2.5-0.5B-Instruct \
		--output_dir kva_result/hdf5/Qwen2.5-0.5B-Instruct \
		--steps 7 \
		--result_file kva_mmlu.h5 \
		--write_mode w \
		--ig_method riemann_trapezoid

analysing_mmlu_llama:
	CUDA_VISIBLE_DEVICES=0 \
	uv run accelerate launch \
		--mixed_precision bf16 \
		analyse_mmlu_llama.py \
		--model_path meta-llama/Llama-3.1-8B-Instruct \
		--output_dir kva_result/hdf5/Llama-3.1-8B-Instruct \
		--steps 7 \
		--result_file kva_mmlu.h5 \
		--write_mode w \
		--ig_method riemann_trapezoid
