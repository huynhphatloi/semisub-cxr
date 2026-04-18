"""Reproducibility seed setting for Python, NumPy, PyTorch, and CUDA."""

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_all_seeds(seed: int) -> None:
    """Set Python, NumPy, PyTorch, CUDA seeds. Warn if deterministic mode unavailable.

    Args:
        seed: Integer seed value for all random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Attempt to enable CUDA deterministic mode
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        logger.warning(
            "CUDA deterministic mode is unavailable. "
            "Exact GPU reproducibility cannot be guaranteed."
        )

    os.environ["PYTHONHASHSEED"] = str(seed)

    logger.info("All random seeds set to %d", seed)
