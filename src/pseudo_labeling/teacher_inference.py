"""Teacher model inference on unlabeled data for pseudo-label generation.

Runs a standard forward pass (no dropout) over a dataloader, collecting
sample indices and sigmoid probabilities.

Validates: Requirements 5.1, 5.2
"""

import logging
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def run_teacher_inference(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run teacher model inference and collect predictions.

    The dataloader is expected to yield 3-tuples ``(images, labels, indices)``.
    Labels are ignored; only indices and sigmoid(logits) are collected.

    Args:
        model: Trained teacher model returning raw logits.
        dataloader: DataLoader over the unlabeled subset.
        device: Device to run inference on.

    Returns:
        sample_indices: 1-D int array of shape ``(N,)`` with sample indices.
        probabilities: 2-D float array of shape ``(N, C)`` with predicted
            probabilities in [0, 1].
    """
    model.eval()
    all_indices = []
    all_probs = []

    with torch.no_grad():
        for batch in dataloader:
            images, _labels, indices = batch[0], batch[1], batch[2]
            images = images.to(device)

            logits = model(images)
            probs = torch.sigmoid(logits)

            all_indices.append(indices.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    sample_indices = np.concatenate(all_indices, axis=0)
    probabilities = np.concatenate(all_probs, axis=0)

    logger.info(
        "Teacher inference complete: %d samples, %d classes",
        probabilities.shape[0],
        probabilities.shape[1],
    )

    return sample_indices, probabilities
