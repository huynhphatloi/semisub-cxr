"""MC Dropout uncertainty estimation for pseudo-label filtering.

Performs multiple stochastic forward passes with dropout enabled at inference
time, then computes per-class uncertainty estimates (predictive entropy) from
the resulting distribution of predictions.

Validates: Requirements 6.1, 6.2, 6.6, 6.7, 6.8
"""

import logging
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

_SUPPORTED_METRICS = {"predictive_entropy"}


def _enable_dropout(model: nn.Module) -> None:
    """Enable dropout layers while keeping everything else in eval mode.

    Sets the model to eval mode first, then re-enables training mode on
    ``nn.Dropout`` layers only so that they apply stochastic masking during
    the forward pass.
    """
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def compute_predictive_entropy(mc_probs: np.ndarray) -> np.ndarray:
    """Compute binary predictive entropy from MC Dropout probability samples.

    Args:
        mc_probs: Array of shape ``(T, N, C)`` where *T* is the number of
            stochastic forward passes, *N* is the number of samples, and *C*
            is the number of classes.  Values should be in (0, 1).

    Returns:
        Predictive entropy array of shape ``(N, C)``.
        ``H = -[p * log(p) + (1 - p) * log(1 - p)]`` where
        ``p = mean(mc_probs, axis=0)``.
    """
    # Mean probability over T passes → (N, C)
    p = np.mean(mc_probs, axis=0)

    # Clamp to avoid log(0)
    eps = 1e-10
    p = np.clip(p, eps, 1.0 - eps)

    entropy = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))
    return entropy


def run_mc_dropout_inference(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_passes: int,
    uncertainty_metric: str = "predictive_entropy",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run MC Dropout inference and compute uncertainty estimates.

    Performs *num_passes* stochastic forward passes through the model with
    dropout enabled, collects sigmoid probabilities, and computes per-class
    uncertainty using the specified metric.

    The dataloader is expected to yield 3-tuples ``(images, labels, indices)``.

    Args:
        model: Trained model returning raw logits.
        dataloader: DataLoader over the unlabeled subset.
        device: Device to run inference on.
        num_passes: Number of stochastic forward passes (must be > 0).
        uncertainty_metric: Uncertainty metric to compute.  Currently only
            ``"predictive_entropy"`` is supported.

    Returns:
        sample_indices: 1-D int array of shape ``(N,)``.
        mean_probabilities: 2-D float array of shape ``(N, C)`` — mean
            predicted probability over all passes.
        uncertainties: 2-D float array of shape ``(N, C)`` — per-class
            uncertainty values.

    Raises:
        ValueError: If *num_passes* <= 0 or *uncertainty_metric* is not
            supported.
    """
    if num_passes <= 0:
        raise ValueError(
            f"num_passes must be > 0, got {num_passes}"
        )
    if uncertainty_metric not in _SUPPORTED_METRICS:
        raise ValueError(
            f"Unsupported uncertainty metric '{uncertainty_metric}'. "
            f"Supported: {sorted(_SUPPORTED_METRICS)}"
        )

    _enable_dropout(model)

    # First pass: collect indices and determine shapes
    all_indices = []
    # mc_probs_list[t] will be a list of batch arrays for pass t
    mc_probs_list: list = []

    for t in range(num_passes):
        pass_probs = []
        pass_indices = [] if t == 0 else None

        with torch.no_grad():
            for batch in dataloader:
                images, _labels, indices = batch[0], batch[1], batch[2]
                images = images.to(device)

                logits = model(images)
                probs = torch.sigmoid(logits)

                pass_probs.append(probs.cpu().numpy())
                if t == 0:
                    pass_indices.append(indices.cpu().numpy())

        mc_probs_list.append(np.concatenate(pass_probs, axis=0))
        if t == 0:
            all_indices = np.concatenate(pass_indices, axis=0)

    # Stack into (T, N, C)
    mc_probs = np.stack(mc_probs_list, axis=0)

    # Mean probabilities over T passes → (N, C)
    mean_probabilities = np.mean(mc_probs, axis=0)

    # Compute uncertainty
    uncertainties = compute_predictive_entropy(mc_probs)

    logger.info(
        "MC Dropout inference complete: %d passes, %d samples, %d classes, "
        "metric=%s",
        num_passes,
        mean_probabilities.shape[0],
        mean_probabilities.shape[1],
        uncertainty_metric,
    )

    return all_indices, mean_probabilities, uncertainties
