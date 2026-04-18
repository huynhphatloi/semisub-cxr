"""Checkpoint save/load utilities for model training persistence."""

import logging
import os
from typing import Any, Dict, Optional

import torch

from src.utils.logging_utils import get_git_commit, log_library_versions

logger = logging.getLogger(__name__)


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: Any,
    val_metrics: Dict[str, Any],
    extra_meta: Optional[Dict[str, Any]] = None,
) -> None:
    """Save a training checkpoint to disk.

    The checkpoint contains model and optimizer state dicts, the current epoch,
    experiment config, validation metrics, random seed, git commit hash, and
    library versions.

    Args:
        path: File path where the checkpoint will be saved.
        model: The model whose state dict will be saved.
        optimizer: The optimizer whose state dict will be saved.
        epoch: Current epoch number.
        config: Experiment configuration (stored as-is; use a dict or dataclass).
        val_metrics: Validation metrics dict (e.g. macro_auroc, per_class_auroc).
        extra_meta: Optional additional metadata to include in the checkpoint.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "config": config,
        "val_metrics": val_metrics,
        "seed": config.get("seed") if isinstance(config, dict) else getattr(config, "seed", None),
        "git_commit": get_git_commit(),
        "library_versions": log_library_versions(),
    }

    if extra_meta is not None:
        checkpoint.update(extra_meta)

    torch.save(checkpoint, path)
    logger.info("Checkpoint saved to %s (epoch %d)", path, epoch)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Load a training checkpoint from disk.

    Restores model (and optionally optimizer) state dicts and returns a
    metadata dict with epoch, config, val_metrics, seed, git_commit, and
    library_versions.

    Args:
        path: File path to the saved checkpoint.
        model: The model to load state into.
        optimizer: Optional optimizer to restore state into. If ``None``,
            optimizer state is skipped.
        device: Device to map tensors to (e.g. ``"cpu"`` or ``"cuda:0"``).

    Returns:
        A dict containing ``epoch``, ``config``, ``val_metrics``, ``seed``,
        ``git_commit``, and ``library_versions`` from the checkpoint.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info("Model state restored from %s", path)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info("Optimizer state restored from %s", path)

    metadata = {
        "epoch": checkpoint["epoch"],
        "config": checkpoint["config"],
        "val_metrics": checkpoint["val_metrics"],
        "seed": checkpoint.get("seed"),
        "git_commit": checkpoint.get("git_commit"),
        "library_versions": checkpoint.get("library_versions"),
    }
    return metadata
