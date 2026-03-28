"""Centralized logging setup."""

import logging
import sys
from pathlib import Path


def setup_logger(name="football_predictor", level=None, log_file=None):
    """Configure and return a logger with console and optional file output."""
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level or logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name=None):
    """Get a child logger under the main football_predictor logger."""
    base = logging.getLogger("football_predictor")
    if name:
        return base.getChild(name)
    return base
