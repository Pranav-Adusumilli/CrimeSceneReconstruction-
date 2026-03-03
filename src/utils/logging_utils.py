"""
Logging configuration for the Crime Scene Reconstruction system.
"""

import logging
import sys
from pathlib import Path


def setup_logging(log_dir: str = "outputs/reports", level: int = logging.INFO):
    """Configure logging to both console and file."""

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(log_dir) / "pipeline.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console_fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)-30s │ %(message)s",
        datefmt="%H:%M:%S",
    )
    console.setFormatter(console_fmt)
    root_logger.addHandler(console)

    # File handler
    file_handler = logging.FileHandler(str(log_file), mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )
    file_handler.setFormatter(file_fmt)
    root_logger.addHandler(file_handler)

    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("diffusers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)

    return root_logger
