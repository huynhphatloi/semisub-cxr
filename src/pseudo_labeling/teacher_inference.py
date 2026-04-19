"""Teacher model inference on unlabeled data for pseudo-label generation.

Runs a standard forward pass (no dropout) over a dataloader, collecting
sample indices and sigmoid probabilities.

Validates: Requirements 5.1, 5.2
"""

import logging
import time
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
    """Run teacher model inference and collect predictions."""
    model.eval()
    all_indices = []
    all_probs = []
    num_batches = len(dataloader)
    log_interval = max(1, num_batches // 20)  # Log ~20 times
    start = time.time()

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            images, _labels, indices = batch[0], batch[1], batch[2]
            images = images.to(device)

            logits = model(images)
            probs = torch.sigmoid(logits)

            all_indices.append(indices.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

            if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == num_batches:
                pct = 100.0 * (batch_idx + 1) / num_batches
                elapsed = time.time() - start
                eta = elapsed / (batch_idx + 1) * (num_batches - batch_idx - 1)
                logger.info(
                    "  Teacher inference: [%d/%d] %5.1f%% | "
                    "elapsed=%.0fs ETA=%.0fs",
                    batch_idx + 1, num_batches, pct, elapsed, eta,
                )

    sample_indices = np.concatenate(all_indices, axis=0)
    probabilities = np.concatenate(all_probs, axis=0)

    total_time = time.time() - start
    logger.info(
        "Teacher inference complete: %d samples, %d classes in %.1fs",
        probabilities.shape[0],
        probabilities.shape[1],
        total_time,
    )

    return sample_indices, probabilities
