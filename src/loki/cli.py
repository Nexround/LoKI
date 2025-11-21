"""Console interface for common LoKI workflows."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from loki import __version__
from loki.models import get_loki_config_class, get_loki_model_class
from loki.selection import (
    load_attributions_from_hdf5,
    save_positions_to_json,
    select_trainable_nodes_global_highest,
    select_trainable_nodes_global_lowest,
    select_trainable_nodes_layer_balanced,
)
from loki.utils.logging_config import configure_root_logger
from loki.utils.model_utils import create_loki_model, restore_loki_model

logger = logging.getLogger(__name__)

STRATEGY_MAP = {
    "layer_balanced": select_trainable_nodes_layer_balanced,
    "global_lowest": select_trainable_nodes_global_lowest,
    "global_highest": select_trainable_nodes_global_highest,
}


def add_select_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add node selection subcommand."""
    select = subparsers.add_parser(
        "select-nodes",
        help="Generate trainable node positions from attribution HDF5 output",
    )
    select.add_argument(
        "--hdf5-path",
        type=Path,
        required=True,
        help="Path to HDF5 file containing attribution scores",
    )
    select.add_argument(
        "--quota",
        type=float,
        required=True,
        help="Percentage of parameters to train (0-100)",
    )
    select.add_argument(
        "--strategy",
        type=str,
        default="layer_balanced",
        choices=list(STRATEGY_MAP.keys()),
        help="Node selection strategy",
    )
    select.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for position JSON (default: derived from HDF5 path)",
    )
    select.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Output JSON filename (default: {quota}.json or {strategy}_{quota}.json)",
    )
    select.set_defaults(func=run_select_nodes)


def add_create_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add LoKI model creation subcommand."""
    create = subparsers.add_parser(
        "create-model",
        help="Convert a pretrained model into a LoKI model",
    )
    create.add_argument(
        "--model-type",
        type=str,
        required=False,
        help="Optional architecture override; inferred from model_name when omitted",
    )
    create.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Pretrained model name or path",
    )
    create.add_argument(
        "--target-pos-path",
        type=Path,
        required=True,
        help="JSON file containing trainable node positions",
    )
    create.add_argument(
        "--save-dir",
        type=Path,
        required=True,
        help="Directory to save the LoKI model",
    )
    create.add_argument(
        "--torch-dtype",
        type=str,
        default="auto",
        choices=["float32", "float16", "bfloat16", "auto"],
        help="PyTorch dtype for model weights",
    )
    create.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow loading models with custom code",
    )
    create.set_defaults(func=run_create_model)


def add_restore_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add LoKI model restoration subcommand."""
    restore = subparsers.add_parser(
        "restore-model",
        help="Merge LoKI weights back into a standard model format",
    )
    restore.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to the saved LoKI model directory",
    )
    restore.add_argument(
        "--target-pos-path",
        type=Path,
        required=True,
        help="JSON file containing trainable node positions",
    )
    restore.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Output directory for the restored model",
    )
    restore.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Original pretrained model name or path",
    )
    restore.set_defaults(func=run_restore_model)


def run_select_nodes(args: argparse.Namespace) -> int:
    """Handle select-nodes command."""
    cmd_logger = logging.getLogger("loki.cli.select")

    if not 0 < args.quota <= 100:
        cmd_logger.error("Quota must be between 0 and 100, got %.2f", args.quota)
        return 1

    if not args.hdf5_path.exists():
        cmd_logger.error("HDF5 file not found: %s", args.hdf5_path)
        return 1

    output_dir = args.output_dir
    if output_dir is None:
        model_name = args.hdf5_path.parent.name
        output_dir = Path("kva_result/pos_json") / model_name
        cmd_logger.info("Auto-detected output directory: %s", output_dir)

    output_name = args.output_name
    if output_name is None:
        if args.strategy == "layer_balanced":
            output_name = f"{int(args.quota)}.json"
        else:
            output_name = f"{args.strategy}_{int(args.quota)}.json"
        cmd_logger.info("Using output filename: %s", output_name)

    output_path = Path(output_dir) / output_name

    cmd_logger.info("Loading attribution scores from %s", args.hdf5_path)
    attribution_scores = load_attributions_from_hdf5(args.hdf5_path)
    cmd_logger.info("Loaded attribution scores with shape %s", attribution_scores.shape)

    selection_fn = STRATEGY_MAP[args.strategy]
    cmd_logger.info("Applying %s strategy with quota=%.2f%%", args.strategy, args.quota)
    positions = selection_fn(attribution_scores, args.quota)

    num_layers = len(positions)
    total_selected = sum(len(layer_pos) for layer_pos in positions)
    avg_per_layer = total_selected / num_layers if num_layers else 0

    cmd_logger.info(
        "Selected %d nodes across %d layers (avg %.1f per layer)",
        total_selected,
        num_layers,
        avg_per_layer,
    )

    save_positions_to_json(positions, output_path)
    cmd_logger.info("Selection complete. Results saved to %s", output_path)
    return 0


def run_create_model(args: argparse.Namespace) -> int:
    """Handle create-model command."""
    cmd_logger = logging.getLogger("loki.cli.create")

    if not args.target_pos_path.exists():
        cmd_logger.error("Target position file not found: %s", args.target_pos_path)
        return 1

    try:
        import torch
    except ImportError:
        cmd_logger.error("PyTorch is required for create-model command")
        return 1

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "auto": "auto",
    }

    torch_dtype = dtype_map[args.torch_dtype]
    model_identifier = args.model_type or args.model_name

    try:
        loki_model_class = get_loki_model_class(model_identifier)
        loki_config_class = get_loki_config_class(model_identifier)
    except Exception as exc:
        cmd_logger.error("Failed to resolve model configuration: %s", exc)
        return 1

    cmd_logger.info("Creating LoKI model")
    cmd_logger.info("  Model type: %s", args.model_type or "auto")
    cmd_logger.info("  Base model: %s", args.model_name)
    cmd_logger.info("  Position file: %s", args.target_pos_path)
    cmd_logger.info("  Save directory: %s", args.save_dir)
    cmd_logger.info("  Dtype: %s", args.torch_dtype)

    try:
        create_loki_model(
            loki_model_class=loki_model_class,
            loki_config_cls=loki_config_class,
            model_name=args.model_name,
            target_pos_path=args.target_pos_path,
            save_dir=args.save_dir,
            torch_dtype=torch_dtype,
            trust_remote_code=args.trust_remote_code,
        )
    except Exception as exc:
        cmd_logger.exception("Failed to create LoKI model: %s", exc)
        return 1

    cmd_logger.info("LoKI model saved to %s", args.save_dir)
    return 0


def run_restore_model(args: argparse.Namespace) -> int:
    """Handle restore-model command."""
    cmd_logger = logging.getLogger("loki.cli.restore")

    if not args.model_path.exists():
        cmd_logger.error("LoKI model not found: %s", args.model_path)
        return 1

    if not args.target_pos_path.exists():
        cmd_logger.error("Target position file not found: %s", args.target_pos_path)
        return 1

    cmd_logger.info("Restoring LoKI model")
    cmd_logger.info("  LoKI model: %s", args.model_path)
    cmd_logger.info("  Position file: %s", args.target_pos_path)
    cmd_logger.info("  Output path: %s", args.output_path)
    cmd_logger.info("  Base model: %s", args.model_name)

    try:
        restore_loki_model(
            model_path=args.model_path,
            target_pos_path=args.target_pos_path,
            output_path=args.output_path,
            model_name=args.model_name,
        )
    except Exception as exc:
        cmd_logger.exception("Failed to restore model: %s", exc)
        return 1

    cmd_logger.info("Model successfully restored to %s", args.output_path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser."""
    parser = argparse.ArgumentParser(
        prog="loki",
        description="LoKI command-line interface",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    add_select_parser(subparsers)
    add_create_parser(subparsers)
    add_restore_parser(subparsers)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for console script."""
    parser = build_parser()
    args = parser.parse_args(argv)

    level = getattr(logging, args.log_level.upper(), logging.INFO)
    configure_root_logger(level=level)

    try:
        return args.func(args)
    except Exception as exc:  # pragma: no cover - defensive catch-all for CLI
        logger.exception("Unexpected error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
