"""Low-label split generation with iterative stratified sampling.

Generates reproducible labeled/unlabeled partitions of a training set,
persists them as JSON, and reloads existing splits when available.

Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8
"""

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

logger = logging.getLogger(__name__)

_MAX_RETRIES = 10


class SplitGenerationError(Exception):
    """Raised when split generation fails after all retries."""


@dataclass
class SplitMetadata:
    """Metadata for a labeled/unlabeled split."""

    seed: int
    labeled_ratio: float
    total_train_size: int
    labeled_size: int
    unlabeled_size: int
    labeled_indices: List[int]
    unlabeled_indices: List[int]
    per_class_positive_counts: Dict[str, int]
    generation_timestamp: str


def _split_path(splits_dir: str, labeled_ratio: float, seed: int) -> str:
    return os.path.join(splits_dir, f"split_r{labeled_ratio}_s{seed}.json")


def _extract_labels(dataset: Any) -> np.ndarray:
    """Extract a (N, C) numpy label matrix from *dataset*.

    Accepts either a numpy array directly or any object with a ``.labels``
    attribute (e.g. ``CheXpertDataset``).
    """
    if isinstance(dataset, np.ndarray):
        return dataset
    labels = getattr(dataset, "labels", None)
    if labels is None:
        raise TypeError(
            "dataset must be a numpy array or have a .labels attribute"
        )
    # Handle torch Tensors transparently
    if hasattr(labels, "numpy"):
        return labels.numpy()
    return np.asarray(labels)


def _validate_split(meta: SplitMetadata, total_size: int) -> None:
    """Validate loaded split metadata for consistency."""
    if meta.total_train_size != total_size:
        raise ValueError(
            f"Split total_train_size ({meta.total_train_size}) does not match "
            f"dataset size ({total_size})."
        )
    labeled_set = set(meta.labeled_indices)
    unlabeled_set = set(meta.unlabeled_indices)

    if labeled_set & unlabeled_set:
        raise ValueError("Labeled and unlabeled indices overlap.")

    if labeled_set | unlabeled_set != set(range(total_size)):
        raise ValueError(
            "Labeled and unlabeled indices do not cover the full dataset."
        )

    if len(meta.labeled_indices) != meta.labeled_size:
        raise ValueError("labeled_size does not match len(labeled_indices).")

    if len(meta.unlabeled_indices) != meta.unlabeled_size:
        raise ValueError("unlabeled_size does not match len(unlabeled_indices).")


def _save_split(meta: SplitMetadata, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(asdict(meta), f)


def _load_split(path: str) -> SplitMetadata:
    with open(path, "r") as f:
        data = json.load(f)
    return SplitMetadata(**data)



def generate_split(
    dataset: Any,
    labeled_ratio: float,
    seed: int,
    splits_dir: str,
    min_positive_per_class: int = 1,
    class_names: Optional[List[str]] = None,
) -> SplitMetadata:
    """Generate or load a labeled/unlabeled split.

    Parameters
    ----------
    dataset
        A numpy array of shape ``(N, C)`` or an object with a ``.labels``
        attribute (e.g. ``CheXpertDataset``).
    labeled_ratio : float
        Fraction of training samples to include in the labeled subset.
    seed : int
        Random seed for reproducibility.
    splits_dir : str
        Directory where split JSON files are persisted.
    min_positive_per_class : int
        Minimum number of positive samples required per class in the
        labeled subset.  Defaults to 1.
    class_names : list[str] or None
        Human-readable class names used in metadata.  If *None*, generic
        names ``class_0, class_1, …`` are used.

    Returns
    -------
    SplitMetadata

    Raises
    ------
    SplitGenerationError
        If a valid split cannot be produced after ``_MAX_RETRIES`` attempts.
    """
    labels = _extract_labels(dataset)
    n_samples, n_classes = labels.shape

    if class_names is None:
        class_names = [f"class_{i}" for i in range(n_classes)]

    # --- Try to load an existing split -----------------------------------
    path = _split_path(splits_dir, labeled_ratio, seed)
    if os.path.isfile(path):
        logger.info("Loading existing split from %s", path)
        meta = _load_split(path)
        _validate_split(meta, n_samples)
        return meta

    # --- Generate a new split with retry logic ---------------------------
    for attempt in range(_MAX_RETRIES):
        current_seed = seed + attempt

        splitter = MultilabelStratifiedShuffleSplit(
            n_splits=1,
            test_size=1.0 - labeled_ratio,
            random_state=current_seed,
        )

        # iterstrat expects (X, y); X can be a dummy index array
        X_dummy = np.arange(n_samples).reshape(-1, 1)
        labeled_idx, unlabeled_idx = next(
            splitter.split(X_dummy, labels)
        )

        # Check minimum positive count per class
        labeled_labels = labels[labeled_idx]
        per_class_counts = labeled_labels.sum(axis=0).astype(int)

        failed_classes = [
            class_names[c]
            for c in range(n_classes)
            if per_class_counts[c] < min_positive_per_class
        ]

        if not failed_classes:
            # Success — build metadata and persist
            meta = SplitMetadata(
                seed=current_seed,
                labeled_ratio=labeled_ratio,
                total_train_size=n_samples,
                labeled_size=len(labeled_idx),
                unlabeled_size=len(unlabeled_idx),
                labeled_indices=sorted(labeled_idx.tolist()),
                unlabeled_indices=sorted(unlabeled_idx.tolist()),
                per_class_positive_counts={
                    class_names[c]: int(per_class_counts[c])
                    for c in range(n_classes)
                },
                generation_timestamp=datetime.now(timezone.utc).isoformat(),
            )
            _save_split(meta, path)
            if attempt > 0:
                logger.info(
                    "Split succeeded on attempt %d (seed offset +%d).",
                    attempt + 1,
                    attempt,
                )
            return meta

        logger.warning(
            "Attempt %d (seed=%d): classes with insufficient positives: %s. "
            "Retrying with offset seed.",
            attempt + 1,
            current_seed,
            failed_classes,
        )

    raise SplitGenerationError(
        f"Failed to generate a valid split after {_MAX_RETRIES} retries. "
        f"Classes with insufficient positives at ratio={labeled_ratio}: "
        f"{failed_classes}. Consider increasing the labeled ratio or "
        f"checking the dataset for class imbalance."
    )
