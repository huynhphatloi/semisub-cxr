"""Evaluation metrics for multi-label classification.

Provides AUROC computation (macro and per-class) and optional metrics
(macro F1, mean average precision) using scikit-learn as the reference
implementation.

Validates: Requirements 7.1, 7.4
"""

import logging
from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

logger = logging.getLogger(__name__)


def compute_auroc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    label_names: List[str],
) -> Dict[str, Any]:
    """Compute macro-AUROC and per-class AUROC.

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground-truth matrix of shape ``(N, C)``.
    y_score : np.ndarray
        Predicted score (probability) matrix of shape ``(N, C)``.
    label_names : list[str]
        Names for each of the ``C`` classes.

    Returns
    -------
    dict
        ``{'macro_auroc': float, 'per_class_auroc': {name: float}}``.
        Classes with only one label value get ``NaN`` for their AUROC
        and are excluded from the macro average.
    """
    per_class_auroc: Dict[str, float] = {}
    valid_aurocs: List[float] = []

    for i, name in enumerate(label_names):
        unique = np.unique(y_true[:, i])
        if len(unique) >= 2:
            auc = float(roc_auc_score(y_true[:, i], y_score[:, i]))
            per_class_auroc[name] = auc
            valid_aurocs.append(auc)
        else:
            per_class_auroc[name] = float("nan")
            logger.warning(
                "Class '%s' has only one label value — AUROC undefined.", name
            )

    macro_auroc = float(np.mean(valid_aurocs)) if valid_aurocs else 0.0

    return {"macro_auroc": macro_auroc, "per_class_auroc": per_class_auroc}


def compute_optional_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    label_names: List[str],
    metrics_list: List[str],
) -> Dict[str, float]:
    """Compute optional metrics if requested.

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground-truth matrix of shape ``(N, C)``.
    y_pred : np.ndarray
        Binary predicted label matrix of shape ``(N, C)``.
    y_score : np.ndarray
        Predicted score (probability) matrix of shape ``(N, C)``.
    label_names : list[str]
        Names for each of the ``C`` classes.
    metrics_list : list[str]
        Which optional metrics to compute. Supported: ``"macro_f1"``,
        ``"map"``.

    Returns
    -------
    dict
        Mapping from metric name to its computed value.
    """
    results: Dict[str, float] = {}

    if "macro_f1" in metrics_list:
        results["macro_f1"] = float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        )

    if "map" in metrics_list:
        per_class_ap: List[float] = []
        for i, name in enumerate(label_names):
            unique = np.unique(y_true[:, i])
            if len(unique) >= 2:
                ap = float(
                    average_precision_score(y_true[:, i], y_score[:, i])
                )
                per_class_ap.append(ap)
            else:
                logger.warning(
                    "Class '%s' has only one label value — AP undefined.", name
                )
        results["map"] = float(np.mean(per_class_ap)) if per_class_ap else 0.0

    return results
