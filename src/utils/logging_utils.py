"""Logging setup, git commit tracking, and library version reporting."""

import logging
import os
import subprocess
import sys
from typing import Dict, Optional

import numpy as np
import torch


def setup_logging(output_dir: str, experiment_name: str) -> logging.Logger:
    """Configure and return a logger that writes to both console and a log file.

    Creates the output directory if it does not exist. The log file is saved as
    ``<output_dir>/<experiment_name>.log``.

    Args:
        output_dir: Directory where the log file will be written.
        experiment_name: Name used for the logger and the log filename.

    Returns:
        A configured ``logging.Logger`` instance.
    """
    os.makedirs(output_dir, exist_ok=True)

    logger = logging.getLogger(experiment_name)
    logger.setLevel(logging.DEBUG)

    # Avoid adding duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # File handler — captures everything at DEBUG level
    log_path = os.path.join(output_dir, f"{experiment_name}.log")
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler — INFO and above
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def get_git_commit() -> Optional[str]:
    """Return the current git commit hash, or ``None`` if unavailable.

    Logs a warning when the git repository cannot be detected or the
    ``git`` command fails.

    Returns:
        The short git commit hash as a string, or ``None``.
    """
    logger = logging.getLogger(__name__)
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return commit.decode("utf-8").strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning(
            "Git repository not detected. Code-version tracking is unavailable."
        )
        return None


def log_library_versions() -> Dict[str, str]:
    """Return a dict of key library versions used by the pipeline.

    Returns:
        Dictionary with keys ``"torch"``, ``"numpy"``, and ``"python"``
        mapped to their version strings.
    """
    versions = {
        "torch": torch.__version__,
        "numpy": np.__version__,
        "python": sys.version.split()[0],
    }
    return versions
