"""Loss functions for multi-label classification training.

Provides masked binary cross-entropy with logits, supporting both standard
(all-ones mask) and pseudo-label (partial mask) modes.

Validates: Requirements 3.2, 5.6
"""

from typing import Optional

import torch
import torch.nn.functional as F


def masked_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Masked binary cross-entropy with logits.

    Computes element-wise BCE loss, multiplies by the loss mask, then takes
    the mean over all elements. When ``loss_mask`` is ``None``, all elements
    contribute equally (equivalent to an all-ones mask).

    Args:
        logits: ``(B, C)`` raw logits from the classifier.
        targets: ``(B, C)`` binary label tensor with values in {0, 1}.
        loss_mask: ``(B, C)`` mask tensor where 1.0 marks active elements
            and 0.0 marks masked (ignored) elements. If ``None``, an
            all-ones mask is used.

    Returns:
        Scalar loss value. If the mask sums to zero, returns ``0.0`` to
        avoid division by zero.
    """
    # Element-wise BCE (no reduction)
    unreduced = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )

    if loss_mask is not None:
        unreduced = unreduced * loss_mask
        mask_sum = loss_mask.sum()
        if mask_sum == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        return unreduced.sum() / mask_sum

    return unreduced.mean()
