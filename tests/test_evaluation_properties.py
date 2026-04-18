"""Property-based tests for the evaluation module.

Feature: semisup-multilabel-cxr, Property 14: AUROC computation matches reference implementation

Validates: Requirements 7.1
"""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.metrics import roc_auc_score

from src.evaluation.metrics import compute_auroc


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

def _label_score_matrices(draw):
    """Generate (y_true, y_score, label_names) where every class has ≥1
    positive and ≥1 negative sample."""
    n_classes = draw(st.integers(min_value=1, max_value=6))
    n_samples = draw(st.integers(min_value=4, max_value=60))

    label_names = [f"class_{i}" for i in range(n_classes)]

    # Build y_true ensuring each class has at least one 0 and one 1
    y_true = np.zeros((n_samples, n_classes), dtype=np.float64)
    for c in range(n_classes):
        # Force first sample positive, second negative
        y_true[0, c] = 1.0
        y_true[1, c] = 0.0
        # Fill remaining randomly
        for r in range(2, n_samples):
            y_true[r, c] = float(draw(st.integers(min_value=0, max_value=1)))

    # Scores in (0, 1)
    y_score = np.array(
        [
            [draw(st.floats(min_value=0.01, max_value=0.99)) for _ in range(n_classes)]
            for _ in range(n_samples)
        ]
    )

    return y_true, y_score, label_names


label_score_strategy = st.composite(_label_score_matrices)


# ---------------------------------------------------------------------------
# Property 14: AUROC computation matches reference implementation
# ---------------------------------------------------------------------------


@given(data=label_score_strategy())
@settings(max_examples=100, deadline=None)
def test_p14_auroc_matches_sklearn(data):
    """**Validates: Requirements 7.1**

    For any binary y_true and score matrices where each class has ≥1
    positive and ≥1 negative, macro_auroc must match sklearn within 1e-6.
    """
    y_true, y_score, label_names = data

    result = compute_auroc(y_true, y_score, label_names)

    # Reference: sklearn macro AUROC
    expected_macro = roc_auc_score(y_true, y_score, average="macro")

    assert abs(result["macro_auroc"] - expected_macro) < 1e-6, (
        f"macro_auroc mismatch: got {result['macro_auroc']}, "
        f"expected {expected_macro}"
    )

    # Also verify per-class values match sklearn per-class
    for i, name in enumerate(label_names):
        expected_auc = roc_auc_score(y_true[:, i], y_score[:, i])
        assert abs(result["per_class_auroc"][name] - expected_auc) < 1e-6, (
            f"per-class AUROC mismatch for {name}: "
            f"got {result['per_class_auroc'][name]}, expected {expected_auc}"
        )
