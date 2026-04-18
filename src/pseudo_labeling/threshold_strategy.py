"""Pseudo-label acceptance strategies for class-wise thresholding.

Defines a pluggable ThresholdStrategy ABC and two concrete implementations:
- PositiveOnlyStrategy: accepts pseudo-label 1.0 where prob > threshold
- UncertaintyFilteredStrategy: accepts where prob > conf AND uncertainty < unc

Validates: Requirements 5.2, 5.3, 5.4, 5.9, 6.3, 6.4, 11.2, 16.1
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np


class ThresholdStrategy(ABC):
    """Abstract base class for pseudo-label acceptance strategies.

    Subclasses implement the ``accept`` method which decides, for each
    (sample, class) entry, whether to accept a pseudo-label or reject it.
    """

    @abstractmethod
    def accept(
        self,
        probabilities: np.ndarray,
        uncertainties: Optional[np.ndarray],
        class_names: List[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Decide which pseudo-labels to accept.

        Args:
            probabilities: (N, C) predicted probabilities in [0, 1].
            uncertainties: (N, C) uncertainty values, or None if not available.
            class_names: list of C class name strings (column order).

        Returns:
            pseudo_labels: (N, C) float array — 1.0 for accepted positives,
                           NaN for rejected entries.
            rejection_mask: (N, C) boolean array — True where rejected.
        """


class PositiveOnlyStrategy(ThresholdStrategy):
    """Accept pseudo-label 1.0 where probability exceeds a per-class threshold.

    Entries at or below the threshold are rejected (NaN). Thresholds are
    applied independently per class.

    Validates: Requirements 5.2, 5.3, 5.4, 5.9
    """

    def __init__(self, confidence_thresholds: Dict[str, float]) -> None:
        self.confidence_thresholds = confidence_thresholds

    def accept(
        self,
        probabilities: np.ndarray,
        uncertainties: Optional[np.ndarray],
        class_names: List[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        n, c = probabilities.shape
        if len(class_names) != c:
            raise ValueError(
                f"class_names length ({len(class_names)}) != "
                f"probabilities columns ({c})"
            )

        pseudo_labels = np.full((n, c), np.nan, dtype=np.float64)
        rejection_mask = np.ones((n, c), dtype=bool)  # True = rejected

        for j, name in enumerate(class_names):
            threshold = self.confidence_thresholds.get(name, 1.0)
            accepted = probabilities[:, j] > threshold
            pseudo_labels[accepted, j] = 1.0
            rejection_mask[accepted, j] = False

        return pseudo_labels, rejection_mask


class UncertaintyFilteredStrategy(ThresholdStrategy):
    """Accept pseudo-label 1.0 where probability exceeds a confidence threshold
    AND uncertainty is below an uncertainty threshold.

    Both conditions must be satisfied for acceptance. Entries failing either
    condition are rejected (NaN).

    Validates: Requirements 6.3, 6.4
    """

    def __init__(
        self,
        confidence_thresholds: Dict[str, float],
        uncertainty_thresholds: Dict[str, float],
    ) -> None:
        self.confidence_thresholds = confidence_thresholds
        self.uncertainty_thresholds = uncertainty_thresholds

    def accept(
        self,
        probabilities: np.ndarray,
        uncertainties: Optional[np.ndarray],
        class_names: List[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        n, c = probabilities.shape
        if len(class_names) != c:
            raise ValueError(
                f"class_names length ({len(class_names)}) != "
                f"probabilities columns ({c})"
            )
        if uncertainties is None:
            raise ValueError(
                "UncertaintyFilteredStrategy requires uncertainties, got None"
            )
        if uncertainties.shape != (n, c):
            raise ValueError(
                f"uncertainties shape {uncertainties.shape} != "
                f"probabilities shape {(n, c)}"
            )

        pseudo_labels = np.full((n, c), np.nan, dtype=np.float64)
        rejection_mask = np.ones((n, c), dtype=bool)

        for j, name in enumerate(class_names):
            conf_thresh = self.confidence_thresholds.get(name, 1.0)
            unc_thresh = self.uncertainty_thresholds.get(name, 0.0)

            conf_ok = probabilities[:, j] > conf_thresh
            unc_ok = uncertainties[:, j] < unc_thresh

            accepted = conf_ok & unc_ok
            pseudo_labels[accepted, j] = 1.0
            rejection_mask[accepted, j] = False

        return pseudo_labels, rejection_mask
