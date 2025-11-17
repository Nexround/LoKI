"""
MMLU Analysis Script using Captum-based Integrated Gradients.

This implementation uses Captum's LayerIntegratedGradients for robust 
and efficient IG computation.

Usage:
    uv run accelerate launch --mixed_precision bf16 analyse_mmlu_llama.py \
        --model_path meta-llama/Llama-3.1-8B-Instruct \
        --output_dir ./kva_result/hdf5/Llama-3.1-8B-Instruct \
        --result_file kva_mmlu.h5 \
        --write_mode w \
        --steps 10
"""

import random
import time
import argparse
import os

from tqdm import tqdm
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

from src.loki.kva import KVALlamaForCausalLM, save_tensor_to_hdf5


mmlu_all_sets = [
    "college_biology",
    "professional_law",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_physics",
    "electrical_engineering",
    "astronomy",
    "anatomy",
    "abstract_algebra",
    "machine_learning",
    "clinical_knowledge",
    "global_facts",
    "management",
    "nutrition",
    "marketing",
    "professional_accounting",
    "high_school_geography",
    "international_law",
    "moral_scenarios",
    "computer_security",
    "high_school_microeconomics",
    "medical_genetics",
    "professional_psychology",
    "jurisprudence",
    "world_religions",
    "philosophy",
    "virology",
    "high_school_chemistry",
    "public_relations",
    "high_school_macroeconomics",
    "human_sexuality",
    "elementary_mathematics",
    "high_school_physics",
    "high_school_computer_science",
    "high_school_european_history",
    "business_ethics",
    "moral_disputes",
    "high_school_statistics",
    "miscellaneous",
    "formal_logic",
    "high_school_government_and_politics",
    "prehistory",
    "security_studies",
    "high_school_biology",
    "logical_fallacies",
    "high_school_world_history",
    "professional_medicine",
    "high_school_mathematics",
    "college_medicine",
    "high_school_us_history",
    "sociology",
    "econometrics",
    "high_school_psychology",
    "human_aging",
    "us_foreign_policy",
    "conceptual_physics",
]


def build_conversation(subset, test_sample):
    """Build conversation prompt for MMLU question."""
    subject = subset.replace("_", " ").title()
    user_msg = {
        "role": "user",
        "content": (
            f"There is a single choice question about {subject}. "
            "Answer the question by replying A, B, C or D.\n"
            f"Question: {test_sample['question']}\n"
            f"Choices:\n"
            + "\n".join(
                [
                    f"{chr(65 + i)}. {choice}"
                    for i, choice in enumerate(test_sample["choices"])
                ]
            )
            + "\nAnswer: \n"
        ),
    }
    return [user_msg]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze MMLU dataset using Captum-based Integrated Gradients"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to pretrained Llama model"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory for HDF5 files"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Number of integration steps for IG computation",
    )
    parser.add_argument(
        "--result_file", type=str, required=True, help="Name of output HDF5 file"
    )
    parser.add_argument(
        "--write_mode",
        type=str,
        default="w",
        choices=["w", "a"],
        help="HDF5 write mode: 'w' (overwrite) or 'a' (append)",
    )
    parser.add_argument(
        "--ig_method",
        type=str,
        default="riemann_trapezoid",
        choices=[
            "riemann_trapezoid",
            "riemann_left",
            "riemann_right",
            "riemann_middle",
            "gausslegendre",
        ],
        help="Integration method for Captum IG",
    )
    parser.add_argument(
        "--use_flash_attention",
        action="store_true",
        help="Use Flash Attention 2 for faster inference",
    )
    args = parser.parse_args()

    # Set random seeds
    device = torch.device("cuda")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("MMLU Analysis with Captum Integrated Gradients")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Output: {os.path.join(args.output_dir, args.result_file)}")
    print(f"Integration steps: {args.steps}")
    print(f"Integration method: {args.ig_method}")
    print(f"Device: {device}")
    print("=" * 80)

    print("\n***** Clearing CUDA cache *****")
    torch.cuda.empty_cache()

    # Load model with Captum-based KVA
    print(f"\n***** Loading model: {args.model_path} *****")
    model_kwargs = {
        "dtype": torch.bfloat16,
        "device_map": "auto",
    }

    if args.use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"
        print("Using Flash Attention 2")

    model = KVALlamaForCausalLM.from_pretrained(args.model_path, **model_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # Compile model for better performance (optional)
    # model = torch.compile(model)  # Uncomment if using PyTorch 2.0+

    start_time = time.time()

    # Process each MMLU subset
    for subset in tqdm(mmlu_all_sets, desc="📦 MMLU Subsets"):
        dataset = load_dataset("cais/mmlu", subset)
        test_dataset = dataset["test"]

        for idx, test_sample in tqdm(
            enumerate(test_dataset),
            desc=f"🗂️ Analyzing {subset}",
            total=len(test_dataset),
            leave=False,
        ):
            # Build conversation and tokenize
            conversation = build_conversation(subset, test_sample)
            inputs = tokenizer.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            ).to(device)
            inputs["attention_mask"] = inputs["attention_mask"].to(torch.int8)

            tic = time.perf_counter()

            # Forward pass to get prediction
            with torch.no_grad():
                outputs = model(inputs["input_ids"], inputs["attention_mask"])
                logits = outputs.logits[:, -1, :]

            predicted_label = int(torch.argmax(logits, dim=-1))
            predicted_token = tokenizer.decode([predicted_label])

            if idx % 10 == 0:  # Print every 10 samples
                print(f"\nSample {idx}: Predicted token = {predicted_token}")

            # Compute integrated gradients using Captum
            model.compute_integrated_gradients(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                target_token_idx=-1,
                predicted_label=predicted_label,
                steps=args.steps,
                method=args.ig_method,
            )

            # Save results to HDF5
            save_tensor_to_hdf5(
                os.path.join(args.output_dir, args.result_file),
                model.integrated_gradients,
                args.write_mode,
            )

            toc = time.perf_counter()

            print(f"Processing time: {toc - tic:.4f} seconds")

            if idx % 10 == 0:
                print(f"Processing time: {toc - tic:.4f} seconds")

            # Clean up for next sample
            model.clean()

    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"Analysis completed successfully!")
    print(f"Total time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")
    print(f"Results saved to: {os.path.join(args.output_dir, args.result_file)}")
    print("=" * 80)
