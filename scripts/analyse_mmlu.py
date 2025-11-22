"""
Unified MMLU Analysis Script with Multi-GPU and Batch Processing Support.

Supports multiple model architectures through a model registry pattern.
Uses Captum's LayerIntegratedGradients for robust and efficient IG computation.

Features:
- Batch inference for efficiency
- Multi-GPU data parallelism (samples distributed across GPUs)
- Multi-GPU model parallelism (model layers distributed across GPUs)

Usage:
    # Single GPU with batch processing
    uv run python scripts/analyse_mmlu.py \
        --model_type <arch> \
        --model_path <hf_model_or_path> \
        --output_dir ./kva_result/hdf5/<Model> \
        --result_file kva_mmlu.h5 \
        --steps 7 \
        --batch_size 4

    # Multi-GPU data parallelism (samples across GPUs)
    uv run accelerate launch --num_processes 4 scripts/analyse_mmlu.py \
        --model_type <arch> \
        --model_path <hf_model_or_path> \
        --output_dir ./kva_result/hdf5/<Model> \
        --result_file kva_mmlu.h5 \
        --steps 7 \
        --batch_size 2 \
        --parallel_mode data

    # Multi-GPU model parallelism (model layers across GPUs)
    uv run python scripts/analyse_mmlu.py \
        --model_type <arch> \
        --model_path <hf_model_or_path> \
        --output_dir ./kva_result/hdf5/<Model> \
        --result_file kva_mmlu.h5 \
        --steps 7 \
        --batch_size 4 \
        --parallel_mode model
"""

import argparse
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedModel

try:
    from accelerate import Accelerator
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False

from loki.constants import DEFAULT_IG_METHOD, DEFAULT_IG_STEPS, MMLU_ALL_SETS
from loki.models import get_architecture_spec, get_kva_model_class
from loki.utils.hdf5_manager import HDF5Manager
from loki.utils.logging_config import configure_root_logger


logger = logging.getLogger(__name__)


class MMLUDataset(Dataset):
    """PyTorch Dataset wrapper for MMLU samples."""

    def __init__(self, samples: list, tokenizer, device):
        """
        Initialize MMLU dataset.

        Args:
            samples: List of (subset_name, sample_dict) tuples
            tokenizer: Tokenizer for processing text
            device: Device to move tensors to
        """
        self.samples = samples
        self.tokenizer = tokenizer
        self.device = device

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        subset, sample = self.samples[idx]
        conversation = build_conversation(subset, sample)
        inputs = self.tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        return {
            "input_ids": inputs["input_ids"][0],
            "attention_mask": inputs["attention_mask"][0],
            "subset": subset,
            "answer": sample["answer"],
        }


def collate_fn(batch):
    """Custom collate function for batching MMLU samples."""
    input_ids = [item["input_ids"] for item in batch]
    attention_masks = [item["attention_mask"] for item in batch]
    
    # Pad sequences to max length in batch
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_masks_padded = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    
    return {
        "input_ids": input_ids_padded,
        "attention_mask": attention_masks_padded,
        "subsets": [item["subset"] for item in batch],
        "answers": [item["answer"] for item in batch],
    }


def build_conversation(subset: str, test_sample: dict) -> list:
    """
    Build conversation prompt for MMLU question.

    Args:
        subset: MMLU subset name (e.g., 'college_biology')
        test_sample: Dictionary containing 'question' and 'choices'

    Returns:
        List with single user message dictionary
    """
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


def setup_model_parallel(model, num_gpus: int):
    """
    Setup model parallelism by distributing layers across GPUs.

    Args:
        model: The model to parallelize
        num_gpus: Number of GPUs to use

    Returns:
        Modified model with layers on different devices
    """
    if num_gpus <= 1:
        return model

    num_layers = model.config.num_hidden_layers
    layers_per_gpu = num_layers // num_gpus
    
    logger.info(f"Setting up model parallelism across {num_gpus} GPUs")
    logger.info(f"Layers per GPU: ~{layers_per_gpu}")
    
    # Move embedding to first GPU
    model.model.embed_tokens.to(f"cuda:0")
    
    # Distribute transformer layers
    for layer_idx in range(num_layers):
        gpu_idx = min(layer_idx // layers_per_gpu, num_gpus - 1)
        model.model.layers[layer_idx].to(f"cuda:{gpu_idx}")
        if layer_idx % 10 == 0:
            logger.debug(f"Layer {layer_idx} -> GPU {gpu_idx}")
    
    # Move final layers to last GPU
    model.model.norm.to(f"cuda:{num_gpus - 1}")
    model.lm_head.to(f"cuda:{num_gpus - 1}")
    
    logger.info("Model parallelism setup complete")
    return model


def process_batch(
    model,
    batch,
    device,
    args,
    hdf5_manager,
    ig_steps: int,
    ig_method: str,
    is_model_parallel: bool = False,
):
    """
    Process a batch of samples and compute integrated gradients.

    Args:
        model: KVA model
        batch: Batch dictionary with input_ids, attention_mask, etc.
        device: Device to use
        args: Command line arguments
        hdf5_manager: HDF5 manager for saving results
        ig_steps: Number of integration steps for IG
        ig_method: Integration method for IG
        is_model_parallel: Whether using model parallelism
    """
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    
    if not is_model_parallel:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
    else:
        # For model parallelism, input goes to first device
        input_ids = input_ids.to("cuda:0")
        attention_mask = attention_mask.to("cuda:0")
    
    attention_mask = attention_mask.to(torch.int8)
    batch_size = input_ids.shape[0]
    
    # Forward pass to get predictions for all samples in batch
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]
    
    predicted_labels = torch.argmax(logits, dim=-1).cpu().numpy()
    
    # Process each sample in batch
    for i in range(batch_size):
        # Extract single sample
        sample_input_ids = input_ids[i:i+1]
        sample_attention_mask = attention_mask[i:i+1]
        predicted_label = int(predicted_labels[i])
        
        # Compute integrated gradients
        model.compute_integrated_gradients(
            input_ids=sample_input_ids,
            attention_mask=sample_attention_mask,
            target_token_idx=-1,
            predicted_label=predicted_label,
            steps=ig_steps,
            method=ig_method,
        )
        
        # Save results to HDF5 (only on main process)
        if hdf5_manager is not None:
            stacked = torch.stack(model.integrated_gradients)
            hdf5_manager.append_data(stacked)
        
        # Clean up
        model.clean()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Unified MMLU analysis with batch and multi-GPU support"
    )

    # Model selection
    parser.add_argument(
        "--model_type",
        type=str,
        required=False,
        default=None,
        help="Optional architecture override; if omitted, inferred from model_path via registry/auto-detect.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier",
    )

    # Output configuration
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for HDF5 files",
    )
    parser.add_argument(
        "--result_file",
        type=str,
        required=True,
        help="Name of output HDF5 file",
    )
    parser.add_argument(
        "--write_mode",
        type=str,
        default="w",
        choices=["w", "a"],
        help="HDF5 write mode: 'w' (overwrite) or 'a' (append)",
    )

    # Batch and parallelism options
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for inference (default: 1)",
    )
    parser.add_argument(
        "--parallel_mode",
        type=str,
        default="none",
        choices=["none", "data", "model"],
        help="Parallelism mode: 'none' (single GPU), 'data' (samples across GPUs), 'model' (layers across GPUs)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of dataloader workers (default: 0)",
    )

    # Other options
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--use_flash_attention",
        action="store_true",
        help="Use Flash Attention 2 for faster inference",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
        help="Print progress every N batches",
    )
    parser.add_argument(
        "--max_samples_per_subset",
        type=int,
        default=50,
        help="Maximum number of samples to process per MMLU subset (default: None, process all)",
    )

    return parser.parse_args()


def main():
    """Main analysis workflow."""
    args = parse_args()

    # Configure logging
    configure_root_logger(logging.INFO)

    # Check for data parallelism requirements
    if args.parallel_mode == "data" and not ACCELERATE_AVAILABLE:
        raise ImportError("Data parallelism requires accelerate library. Install with: pip install accelerate")

    # Initialize accelerator for data parallelism
    accelerator = None
    if args.parallel_mode == "data":
        accelerator = Accelerator()
        device = accelerator.device
        is_main_process = accelerator.is_main_process
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_main_process = True

    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Create output directory (only on main process)
    if is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # Resolve model type using registry/auto-detect
    model_identifier = args.model_type or args.model_path
    resolved_model_type = get_architecture_spec(model_identifier).model_type
    ig_steps = DEFAULT_IG_STEPS
    ig_method = DEFAULT_IG_METHOD

    # Print configuration (only on main process)
    if is_main_process:
        logger.info("=" * 80)
        logger.info("MMLU Analysis with Captum Integrated Gradients")
        logger.info("=" * 80)
        logger.info(f"Model type: {resolved_model_type} (input: {args.model_type or 'auto'})")
        logger.info(f"Model path: {args.model_path}")
        logger.info(f"Output: {os.path.join(args.output_dir, args.result_file)}")
        logger.info(f"Integration steps: {ig_steps}")
        logger.info(f"Integration method: {ig_method}")
        logger.info(f"Batch size: {args.batch_size}")
        logger.info(f"Parallel mode: {args.parallel_mode}")
        logger.info(f"Device: {device}")
        logger.info(f"Write mode: {args.write_mode}")
        if args.max_samples_per_subset is not None:
            logger.info(f"Max samples per subset: {args.max_samples_per_subset}")
        logger.info("=" * 80)

    if is_main_process:
        logger.info("\n***** Clearing CUDA cache *****")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Get KVA model class from registry (auto-register if needed)
    kva_model_class: type[PreTrainedModel] = get_kva_model_class(model_identifier)

    # Load model with Captum-based KVA
    if is_main_process:
        logger.info(f"\n***** Loading {resolved_model_type} model: {args.model_path} *****")
    
    model_kwargs = {
        "dtype": torch.bfloat16,
    }

    if args.use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"
        if is_main_process:
            logger.info("Using Flash Attention 2")

    # Handle device mapping based on parallelism mode
    if args.parallel_mode == "model":
        # Model parallelism: manual device placement
        model_kwargs["device_map"] = None
        num_gpus = torch.cuda.device_count()
        if is_main_process:
            logger.info(f"Using model parallelism across {num_gpus} GPUs")
    elif args.parallel_mode == "data":
        # Data parallelism: accelerate handles it
        model_kwargs["device_map"] = None
    else:
        # Single GPU
        model_kwargs["device_map"] = "auto"

    model = kva_model_class.from_pretrained(args.model_path, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # Setup model parallelism if requested
    if args.parallel_mode == "model":
        num_gpus = torch.cuda.device_count()
        model = setup_model_parallel(model, num_gpus)
    elif args.parallel_mode == "data":
        # Accelerator will handle device placement
        model = accelerator.prepare(model)

    # Prepare metadata for HDF5 file
    analysis_metadata = {
        "model_name": args.model_path,
        "model_type": resolved_model_type,
        "num_layers": model.config.num_hidden_layers if hasattr(model, 'config') else model.module.config.num_hidden_layers,
        "hidden_size": model.config.hidden_size if hasattr(model, 'config') else model.module.config.hidden_size,
        "ig_steps": ig_steps,
        "ig_method": ig_method,
        "seed": args.seed,
        "dataset": "MMLU",
        "use_flash_attention": args.use_flash_attention,
        "batch_size": args.batch_size,
        "parallel_mode": args.parallel_mode,
    }

    # Initialize HDF5 file with metadata if in write mode (only main process)
    output_path = os.path.join(args.output_dir, args.result_file)
    hdf5_manager = None
    
    if is_main_process:
        if args.write_mode == 'w':
            hdf5_manager = HDF5Manager(Path(output_path), mode='w')
            num_layers = model.config.num_hidden_layers if hasattr(model, 'config') else model.module.config.num_hidden_layers
            hidden_size = model.config.hidden_size if hasattr(model, 'config') else model.module.config.hidden_size
            shape = (0, num_layers, hidden_size)
            hdf5_manager.create_dataset_with_metadata(shape=shape, metadata=analysis_metadata)
            logger.info(f"Created HDF5 file with metadata: {output_path}")
        else:
            hdf5_manager = HDF5Manager(Path(output_path), mode='a')

    start_time = time.time()
    total_samples = 0

    # Collect all samples from MMLU subsets
    if is_main_process:
        logger.info("\n***** Collecting MMLU samples *****")
    
    all_samples = []
    for subset in MMLU_ALL_SETS:
        dataset = load_dataset("cais/mmlu", subset)
        test_dataset = dataset["test"]
        
        num_samples = len(test_dataset)
        if args.max_samples_per_subset is not None:
            num_samples = min(args.max_samples_per_subset, num_samples)
        
        for idx in range(num_samples):
            all_samples.append((subset, test_dataset[idx]))
    
    if is_main_process:
        logger.info(f"Total samples to process: {len(all_samples)}")

    # Create dataset and dataloader
    mmlu_dataset = MMLUDataset(all_samples, tokenizer, device)
    
    dataloader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "collate_fn": collate_fn,
    }
    
    dataloader = DataLoader(mmlu_dataset, **dataloader_kwargs)
    
    # Prepare dataloader with accelerator for data parallelism
    if args.parallel_mode == "data":
        dataloader = accelerator.prepare(dataloader)

    # Process batches
    if is_main_process:
        logger.info("\n***** Starting analysis *****")
    
    is_model_parallel = (args.parallel_mode == "model")
    
    for batch_idx, batch in enumerate(tqdm(
        dataloader,
        desc="Processing batches",
        disable=not is_main_process
    )):
        if is_main_process and batch_idx % args.log_interval == 0:
            logger.info(f"Processing batch {batch_idx}/{len(dataloader)}")
        
        process_batch(
            model,
            batch,
            device,
            args,
            hdf5_manager if is_main_process else None,
            ig_steps=ig_steps,
            ig_method=ig_method,
            is_model_parallel=is_model_parallel,
        )
        
        if is_main_process:
            total_samples += len(batch["input_ids"])

    total_time = time.time() - start_time

    # Update metadata with completion info (only main process)
    if is_main_process:
        hdf5_manager = HDF5Manager(Path(output_path), mode='a')
        hdf5_manager.update_metadata({
            "num_samples_processed": total_samples,
            "total_processing_time_seconds": total_time,
            "average_time_per_sample_seconds": total_time / total_samples if total_samples > 0 else 0,
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })

        logger.info("\n" + "=" * 80)
        logger.info("✅ Analysis completed successfully!")
        logger.info(f"Total samples processed: {total_samples}")
        logger.info(f"Total time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")
        logger.info(f"Average time per sample: {total_time / total_samples:.2f} seconds")
        logger.info(f"Results saved to: {output_path}")
        logger.info("=" * 80)


if __name__ == "__main__":
    main()
