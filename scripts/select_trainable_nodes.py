#!/usr/bin/env python3
"""CLI tool for selecting trainable nodes from KVA attribution scores.

This script provides a command-line interface to apply various node selection
strategies to attribution scores stored in HDF5 files from KVA analysis.

Examples:
    # Layer-balanced strategy (default)
    python scripts/select_trainable_nodes.py \\
        --hdf5_path kva_result/hdf5/Llama-3.1-8B-Instruct/kva_mmlu.h5 \\
        --quota 10 \\
        --output_dir kva_result/pos_json/Llama-3.1-8B-Instruct

    # Global lowest strategy
    python scripts/select_trainable_nodes.py \\
        --hdf5_path kva_result/hdf5/Qwen2.5-0.5B-Instruct/kva_mmlu.h5 \\
        --quota 30 \\
        --strategy global_lowest \\
        --output_name global_30_L.json
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loki.selection import (
    load_attributions_from_hdf5,
    save_positions_to_json,
    select_trainable_nodes_global_highest,
    select_trainable_nodes_global_lowest,
    select_trainable_nodes_layer_balanced,
)
from loki.utils.logging_config import setup_logger

logger = setup_logger(__name__)

STRATEGY_MAP = {
    "layer_balanced": select_trainable_nodes_layer_balanced,
    "global_lowest": select_trainable_nodes_global_lowest,
    "global_highest": select_trainable_nodes_global_highest,
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Select trainable nodes from KVA attribution scores",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--hdf5_path",
        type=Path,
        required=True,
        help="Path to HDF5 file containing attribution scores",
    )

    parser.add_argument(
        "--quota",
        type=float,
        required=True,
        help="Percentage of parameters to train (0-100)",
    )

    parser.add_argument(
        "--strategy",
        type=str,
        default="layer_balanced",
        choices=list(STRATEGY_MAP.keys()),
        help="Node selection strategy (default: layer_balanced)",
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory for position JSON file (default: auto-detect from HDF5 path)",
    )

    parser.add_argument(
        "--output_name",
        type=str,
        default=None,
        help="Output JSON filename (default: {quota}.json or {strategy}_{quota}.json)",
    )

    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()

    # Validate quota range
    if not 0 < args.quota <= 100:
        logger.error(f"Quota must be between 0 and 100, got {args.quota}")
        sys.exit(1)

    # Validate HDF5 file exists
    if not args.hdf5_path.exists():
        logger.error(f"HDF5 file not found: {args.hdf5_path}")
        sys.exit(1)

    # Auto-detect output directory if not specified
    if args.output_dir is None:
        model_name = args.hdf5_path.parent.name
        args.output_dir = Path("kva_result/pos_json") / model_name
        logger.info(f"Auto-detected output directory: {args.output_dir}")

    # Generate output filename if not specified
    if args.output_name is None:
        if args.strategy == "layer_balanced":
            args.output_name = f"{int(args.quota)}.json"
        else:
            args.output_name = f"{args.strategy}_{int(args.quota)}.json"
        logger.info(f"Using output filename: {args.output_name}")

    output_path = args.output_dir / args.output_name

    # Load attribution scores
    logger.info(f"Loading attribution scores from {args.hdf5_path}")
    attribution_scores = load_attributions_from_hdf5(args.hdf5_path)
    logger.info(f"Loaded attribution scores with shape: {attribution_scores.shape}")

    # Select nodes using specified strategy
    selection_fn = STRATEGY_MAP[args.strategy]
    logger.info(f"Applying {args.strategy} strategy with quota={args.quota}%")
    positions = selection_fn(attribution_scores, args.quota)

    # Calculate and log statistics
    num_layers = len(positions)
    total_selected = sum(len(layer_pos) for layer_pos in positions)
    avg_per_layer = total_selected / num_layers if num_layers > 0 else 0

    logger.info(f"Selected {total_selected} nodes across {num_layers} layers")
    logger.info(f"Average nodes per layer: {avg_per_layer:.1f}")

    # Save results
    save_positions_to_json(positions, output_path)
    logger.info(f"Selection complete! Results saved to {output_path}")


if __name__ == "__main__":
    main()
