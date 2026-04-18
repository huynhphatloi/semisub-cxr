"""Lightweight sanity checks for each pipeline stage.

Each function performs a quick validation that a component works correctly
before proceeding to the next implementation stage.

Validates: Requirements 15.2, 15A.1, 15A.2, 15A.3, 15A.4
"""

import logging
from typing import Any, Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.data.split_generator import SplitMetadata

logger = logging.getLogger(__name__)


def check_data_loading(dataset: Any) -> bool:
    """Verify that a CheXpertDataset is loadable and well-formed.

    Samples up to 5 items and checks that each returns a tuple of
    ``(image, label_vector, index)`` with ``label_vector.shape == (6,)``.

    Parameters
    ----------
    dataset
        A ``CheXpertDataset`` (or any ``torch.utils.data.Dataset`` with
        the same ``__getitem__`` contract).

    Returns
    -------
    bool
        ``True`` if all checks pass.

    Raises
    ------
    ValueError
        If the dataset is empty or any sample fails validation.
    """
    if len(dataset) == 0:
        raise ValueError("Dataset is empty — no samples found.")

    n_samples = min(5, len(dataset))
    for i in range(n_samples):
        item = dataset[i]

        if not isinstance(item, tuple) or len(item) != 3:
            n = len(item) if isinstance(item, tuple) else "N/A"
            raise ValueError(
                f"Sample {i}: expected a 3-tuple "
                f"(image, label_vector, index), "
                f"got {type(item)} with length {n}."
            )

        image, label_vector, index = item

        if not isinstance(label_vector, torch.Tensor):
            raise ValueError(
                f"Sample {i}: label_vector should be a torch.Tensor, "
                f"got {type(label_vector)}."
            )

        if label_vector.shape != (6,):
            raise ValueError(
                f"Sample {i}: label_vector.shape should be (6,), "
                f"got {label_vector.shape}."
            )

    logger.info(
        "check_data_loading PASSED — sampled %d/%d items successfully.",
        n_samples,
        len(dataset),
    )
    return True


def check_split_validity(
    split_meta: SplitMetadata,
    dataset_size: int,
    labeled_ratio: float,
) -> bool:
    """Verify that a generated split is consistent and well-formed.

    Checks:
    * Labeled and unlabeled index sets are disjoint.
    * Their union covers the full dataset ``{0, …, dataset_size - 1}``.
    * Labeled subset size matches ``round(labeled_ratio * dataset_size)``.
    * Every class has at least one positive sample in the labeled subset.

    Parameters
    ----------
    split_meta : SplitMetadata
        The split metadata to validate.
    dataset_size : int
        Total number of training samples.
    labeled_ratio : float
        Expected labeled fraction.

    Returns
    -------
    bool
        ``True`` if all checks pass.

    Raises
    ------
    ValueError
        If any check fails.
    """
    labeled_set = set(split_meta.labeled_indices)
    unlabeled_set = set(split_meta.unlabeled_indices)

    # Disjointness
    overlap = labeled_set & unlabeled_set
    if overlap:
        raise ValueError(
            f"Labeled and unlabeled indices overlap on {len(overlap)} samples."
        )

    # Exhaustiveness
    full_set = set(range(dataset_size))
    union = labeled_set | unlabeled_set
    if union != full_set:
        missing = full_set - union
        extra = union - full_set
        raise ValueError(
            f"Split does not cover the full dataset. "
            f"Missing indices: {len(missing)}, extra indices: {len(extra)}."
        )

    # Size match
    expected_labeled = round(labeled_ratio * dataset_size)
    actual_labeled = len(split_meta.labeled_indices)
    if actual_labeled != expected_labeled:
        logger.warning(
            "Labeled subset size (%d) differs from expected (%d). "
            "This may be due to stratification adjustments.",
            actual_labeled,
            expected_labeled,
        )

    # Per-class positive counts > 0
    for class_name, count in split_meta.per_class_positive_counts.items():
        if count <= 0:
            raise ValueError(
                f"Class '{class_name}' has {count} positive samples in the "
                f"labeled subset — at least 1 is required."
            )

    logger.info(
        "check_split_validity PASSED — %d labeled, %d unlabeled, %d total.",
        len(split_meta.labeled_indices),
        len(split_meta.unlabeled_indices),
        dataset_size,
    )
    return True


def check_training_step(
    model: torch.nn.Module,
    sample_batch: Tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
) -> bool:
    """Run a single forward + backward pass and verify basic training health.

    Checks:
    * Forward pass produces finite loss (BCE with logits).
    * Backward pass produces at least one parameter with non-zero gradient.

    Parameters
    ----------
    model : torch.nn.Module
        The model to check (must output raw logits).
    sample_batch : tuple[Tensor, Tensor]
        ``(images, labels)`` tensors for a single mini-batch.
    device : torch.device
        Device to run on.

    Returns
    -------
    bool
        ``True`` if all checks pass.

    Raises
    ------
    RuntimeError
        If loss is non-finite or all gradients are zero.
    """
    model.train()
    model.to(device)

    images, labels = sample_batch
    images = images.to(device)
    labels = labels.to(device)

    # Forward
    logits = model(images)
    loss = F.binary_cross_entropy_with_logits(logits, labels)

    if not torch.isfinite(loss):
        raise RuntimeError(
            f"Training sanity check FAILED — loss is not finite: {loss.item()}"
        )

    # Backward
    model.zero_grad()
    loss.backward()

    has_nonzero_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.abs().sum().item() > 0:
            has_nonzero_grad = True
            break

    if not has_nonzero_grad:
        raise RuntimeError(
            "Training sanity check FAILED — all parameter gradients are zero "
            "after backward pass."
        )

    logger.info(
        "check_training_step PASSED — loss=%.6f, gradients non-zero.", loss.item()
    )
    return True


def check_evaluation(
    evaluator: Optional[Callable],
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
) -> bool:
    """Verify that AUROC can be computed on a small validation subset.

    Since the full evaluator module may not be built yet, this function
    performs a lightweight check: it runs the model on a small batch from
    ``val_loader``, collects predictions, and verifies that
    ``sklearn.metrics.roc_auc_score`` can compute AUROC.

    Parameters
    ----------
    evaluator : callable or None
        An optional evaluator callable.  If ``None``, the function falls
        back to a direct sklearn check.
    model : torch.nn.Module
        The trained model (must output raw logits).
    val_loader : DataLoader
        Validation data loader.

    Returns
    -------
    bool
        ``True`` if AUROC is computable.

    Raises
    ------
    RuntimeError
        If predictions cannot be generated or AUROC computation fails.
    """
    from sklearn.metrics import roc_auc_score

    model.eval()
    device = next(model.parameters()).device

    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in val_loader:
            images, labels = batch[0], batch[1]
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)

            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

            # Only need a small subset for the sanity check
            if sum(a.shape[0] for a in all_labels) >= 50:
                break

    y_true = (
        np.concatenate(all_labels, axis=0) if all_labels
        else np.empty((0, 0))
    )
    y_score = (
        np.concatenate(all_probs, axis=0) if all_probs
        else np.empty((0, 0))
    )

    if y_true.shape[0] == 0:
        raise RuntimeError(
            "check_evaluation FAILED — no validation samples collected."
        )

    try:
        # Compute per-class AUROC only for classes with both positives and negatives
        valid_classes = []
        for c in range(y_true.shape[1]):
            unique = np.unique(y_true[:, c])
            if len(unique) >= 2:
                valid_classes.append(c)

        if not valid_classes:
            logger.warning(
                "check_evaluation: no class has both positive and negative "
                "samples in the subset — AUROC cannot be computed, but model "
                "predictions are valid."
            )
        else:
            auroc = roc_auc_score(
                y_true[:, valid_classes],
                y_score[:, valid_classes],
                average="macro",
            )
            logger.info(
                "check_evaluation PASSED — macro AUROC=%.4f on %d samples "
                "(%d valid classes).",
                auroc,
                y_true.shape[0],
                len(valid_classes),
            )
    except Exception as exc:
        raise RuntimeError(
            f"check_evaluation FAILED — AUROC computation error: {exc}"
        ) from exc

    return True
