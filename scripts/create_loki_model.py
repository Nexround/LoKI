#!/usr/bin/env python3
"""CLI tool for creating LoKI models from pretrained models.

This script converts a standard pretrained model into a LoKI model by replacing
MLP down-projection layers with LoKILinear layers based on position JSON files.

Example:
    python scripts/create_loki_model.py \\
        --model_type <arch> \\
        --model_name <hf_model_or_path> \\
        --target_pos_path kva_result/pos_json/<Model>/10.json \\
        --save_dir models/loki_model_10 \\
        --torch_dtype bfloat16
"""

import argparse
import sys
from pathlib import Path

import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loki import create_loki_model
from loki.models import (
    get_loki_config_class,
    get_loki_model_class,
)
from loki.utils.logging_config import setup_logger

logger = setup_logger(__name__)

DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "auto": "auto",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create LoKI model from pretrained model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--model_type",
        type=str,
        required=False,
        help="Optional model architecture override; if omitted, inferred from model_name (auto registration supported).",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Pretrained model name or path (e.g., meta-llama/Llama-3.1-8B-Instruct)",
    )

    parser.add_argument(
        "--target_pos_path",
        type=Path,
        required=True,
        help="Path to JSON file containing trainable node positions",
    )

    parser.add_argument(
        "--save_dir",
        type=Path,
        required=True,
        help="Directory to save the LoKI model",
    )

    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="auto",
        choices=list(DTYPE_MAP.keys()),
        help="PyTorch dtype for model weights (default: auto)",
    )

    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Trust remote code when loading model",
    )

    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()

    # Validate target_pos file exists
    if not args.target_pos_path.exists():
        raise FileNotFoundError(f"Target position file not found: {args.target_pos_path}")

    # Get model and config classes
    model_identifier = args.model_type or args.model_name
    loki_model_class = get_loki_model_class(model_identifier)
    loki_config_class = get_loki_config_class(model_identifier)

    # Convert dtype string to torch dtype
    torch_dtype = DTYPE_MAP[args.torch_dtype]

    logger.info("Creating LoKI model:")
    logger.info(f"  Model type: {args.model_type or 'auto'}")
    logger.info(f"  Base model: {args.model_name}")
    logger.info(f"  Position file: {args.target_pos_path}")
    logger.info(f"  Save directory: {args.save_dir}")
    logger.info(f"  Dtype: {args.torch_dtype}")

    # Create LoKI model
    create_loki_model(
        loki_model_class=loki_model_class,
        loki_config_cls=loki_config_class,
        model_name=args.model_name,
        target_pos_path=str(args.target_pos_path),
        save_dir=str(args.save_dir),
        torch_dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
    )
    logger.info(f"✓ LoKI model successfully created and saved to {args.save_dir}")


if __name__ == "__main__":
    main()
