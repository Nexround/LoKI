"""Logging configuration for LoKI project.

Provides standardized logging setup with consistent formatting across all modules.
"""

import logging
import sys


def setup_logger(
    name: str,
    level: int = logging.INFO,
    format_string: str | None = None,
) -> logging.Logger:
    """Configure logger with consistent formatting.

    Args:
        name: Logger name (typically __name__ from calling module)
        level: Logging level (default: INFO)
        format_string: Custom format string (optional)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers if logger already configured
    if logger.handlers:
        return logger

    handler = logging.StreamHandler(sys.stdout)

    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)

    return logger


def configure_root_logger(level: int = logging.INFO) -> None:
    """Configure root logger for the entire LoKI project.

    Args:
        level: Logging level for all LoKI modules
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
