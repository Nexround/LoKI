#!/usr/bin/env python3
"""CLI tool for restoring LoKI models to original format.

This script converts a LoKI model (with LoKILinear layers) back to a standard
model format by merging the active and frozen weights.

Example:
    python scripts/restore_loki_model.py \\
        --model_path models/loki_model_10 \\
        --target_pos_path kva_result/pos_json/<Model>/10.json \\
        --output_path models/restored_model \\
        --model_name <hf_model_or_path>
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loki import restore_loki_model
from loki.utils.logging_config import setup_logger

logger = setup_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Restore LoKI model to original format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to LoKI model directory",
    )

    parser.add_argument(
        "--target_pos_path",
        type=str,
        required=True,
        help="Path to JSON file containing trainable node positions",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output directory for restored model",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Original pretrained model name or path",
    )

    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()

    # Validate paths
    if not Path(args.model_path).exists():
        raise FileNotFoundError(f"LoKI model not found: {args.model_path}")

    if not Path(args.target_pos_path).exists():
        raise FileNotFoundError(f"Target position file not found: {args.target_pos_path}")

    logger.info("Restoring LoKI model:")
    logger.info(f"  LoKI model: {args.model_path}")
    logger.info(f"  Position file: {args.target_pos_path}")
    logger.info(f"  Output path: {args.output_path}")
    logger.info(f"  Base model: {args.model_name}")

    # Restore model
    restore_loki_model(
        model_path=args.model_path,
        target_pos_path=args.target_pos_path,
        output_path=args.output_path,
        model_name=args.model_name,
    )
    logger.info(f"✓ Model successfully restored to {args.output_path}")


if __name__ == "__main__":
    main()
