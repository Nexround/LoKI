import random
import time
import argparse
import os

from tqdm import tqdm
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

from src.loki.kva import KVAQwen2ForCausalLM, save_tensor_to_hdf5

mmlu_all_sets = [
    "college_biology",
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
    "professional_law",
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


def build_conversation(subset, train_samples, test_sample):
    subject = subset.replace("_", " ").title()
    conversation = []
    for example in train_samples:
        question = f"There is a single choice question about {subject}. Answer the question by replying A, B, C or D.\n"
        question += f"Question: {example['question']}\nChoices:\n"
        question += "\n".join(
            [f"{chr(65 + i)}. {choice}" for i, choice in enumerate(example["choices"])]
        )
        human_msg = {"role": "user", "content": question + "\nAnswer: \n"}
        bot_msg = {"role": "assistant", "content": f"{chr(65 + example['answer'])}\n"}
        conversation.extend([human_msg, bot_msg])
    test_question = f"There is a single choice question about {subject}. Answer the question by replying A, B, C or D.\n"
    test_question += f"Question: {test_sample['question']}\nChoices:\n"
    test_question += "\n".join(
        [f"{chr(65 + i)}. {choice}" for i, choice in enumerate(test_sample["choices"])]
    )
    conversation.append({"role": "user", "content": test_question + "\nAnswer: \n"})
    return conversation


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--result_file", type=str)
    parser.add_argument("--write_mode", type=str)
    args = parser.parse_args()

    device = torch.device("cuda")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.cuda.empty_cache()

    model = KVAQwen2ForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = torch.compile(model)

    start_time = time.time()
    for subset in tqdm(mmlu_all_sets, desc="\U0001f4e6"):
        dataset = load_dataset("cais/mmlu", subset)
        test_dataset = dataset["test"].select(range(50))
        few_shot_samples = dataset["dev"]
        for idx, test_sample in tqdm(
            enumerate(test_dataset),
            desc=f"\U0001f5c2️ Evaluating {subset}",
            total=len(test_dataset),
            leave=False,
        ):
            conversation = build_conversation(subset, few_shot_samples, test_sample)
            inputs = tokenizer.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            ).to(device)
            inputs["attention_mask"] = inputs["attention_mask"].to(torch.int8)

            tic = time.perf_counter()

            logits = model.forward(**inputs, target_token_idx=-1, use_cache=False)
            predicted_label = int(torch.argmax(logits, dim=-1))
            print(tokenizer.decode([predicted_label]))

            model.forward_with_partitioning(
                target_token_idx=-1, steps=args.steps, predicted_label=predicted_label
            )

            save_tensor_to_hdf5(
                os.path.join(args.output_dir, args.result_file),
                model.integrated_gradients,
                args.write_mode,
            )

            print(f"Costing time: {time.perf_counter() - tic:0.4f} seconds")
            model.clean()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total costing time: {elapsed_time:.4f} seconds")
