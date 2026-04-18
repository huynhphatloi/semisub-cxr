"""Property-based tests for uncertainty estimation and filtering.

Uses Hypothesis for property-based testing with 100 examples per property.
"""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from src.pseudo_labeling.mc_dropout import compute_predictive_entropy
from src.pseudo_labeling.threshold_strategy import UncertaintyFilteredStrategy


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


@st.composite
def _mc_probs_tensor(draw):
    """Draw a random (T, N, C) probability tensor with values in (0, 1).

    Returns mc_probs array of shape (T, N, C).
    """
    t = draw(st.integers(min_value=1, max_value=10))
    n = draw(st.integers(min_value=1, max_value=30))
    c = draw(st.integers(min_value=1, max_value=6))

    mc_probs = draw(
        arrays(
            dtype=np.float64,
            shape=(t, n, c),
            elements=st.floats(
                min_value=1e-6, max_value=1.0 - 1e-6, allow_nan=False
            ),
        )
    )
    return mc_probs


def _class_names_strategy(min_classes: int = 1, max_classes: int = 6):
    """Generate a list of unique class name strings."""
    return st.lists(
        st.text(
            alphabet="abcdefghijklmnopqrstuvwxyz_",
            min_size=1,
            max_size=12,
        ),
        min_size=min_classes,
        max_size=max_classes,
        unique=True,
    )


@st.composite
def _uncertainty_filtered_data(draw):
    """Draw random probs, uncertainties, conf thresholds, unc thresholds.

    Returns (probabilities, uncertainties, conf_thresholds, unc_thresholds,
             class_names).
    """
    class_names = draw(_class_names_strategy(min_classes=1, max_classes=6))
    c = len(class_names)
    n = draw(st.integers(min_value=1, max_value=50))

    probabilities = draw(
        arrays(
            dtype=np.float64,
            shape=(n, c),
            elements=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        )
    )

    uncertainties = draw(
        arrays(
            dtype=np.float64,
            shape=(n, c),
            elements=st.floats(min_value=0.0, max_value=2.0, allow_nan=False),
        )
    )

    conf_thresholds = {}
    unc_thresholds = {}
    for name in class_names:
        conf_thresholds[name] = draw(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
        )
        unc_thresholds[name] = draw(
            st.floats(min_value=0.0, max_value=2.0, allow_nan=False)
        )

    return probabilities, uncertainties, conf_thresholds, unc_thresholds, class_names


# ---------------------------------------------------------------------------
# Property 11: Predictive entropy computation follows the binary entropy
#              formula
# Feature: semisup-multilabel-cxr, Property 11: Predictive entropy
#          computation follows the binary entropy formula
# Validates: Requirements 6.2
# ---------------------------------------------------------------------------


@given(mc_probs=_mc_probs_tensor())
@settings(max_examples=100, deadline=None)
def test_property_p11_predictive_entropy_formula(mc_probs):
    """P11: Predictive entropy computation follows the binary entropy formula.

    **Validates: Requirements 6.2**

    For any MC Dropout probability tensor of shape (T, N, C) with values in
    (0, 1), the computed predictive entropy shall equal
    H = -[p * log(p) + (1 - p) * log(1 - p)] where p is the mean probability
    over the T passes, computed independently per sample and per class.
    """
    entropy = compute_predictive_entropy(mc_probs)

    # Compute expected entropy independently
    p = np.mean(mc_probs, axis=0)  # (N, C)
    eps = 1e-10
    p_clamped = np.clip(p, eps, 1.0 - eps)
    expected = -(
        p_clamped * np.log(p_clamped)
        + (1.0 - p_clamped) * np.log(1.0 - p_clamped)
    )

    # Shape check
    assert entropy.shape == expected.shape, (
        f"Shape mismatch: got {entropy.shape}, expected {expected.shape}"
    )

    # Value check within float tolerance
    np.testing.assert_allclose(
        entropy,
        expected,
        atol=1e-8,
        err_msg="Predictive entropy does not match binary entropy formula",
    )

    # Entropy should be non-negative
    assert np.all(entropy >= 0.0), "Entropy contains negative values"

    # Entropy should be at most log(2) (maximum binary entropy)
    assert np.all(entropy <= np.log(2) + 1e-8), (
        f"Entropy exceeds maximum binary entropy log(2)={np.log(2)}"
    )


# ---------------------------------------------------------------------------
# Property 12: Uncertainty-filtered thresholding requires both conditions
# Feature: semisup-multilabel-cxr, Property 12: Uncertainty-filtered
#          thresholding requires both conditions
# Validates: Requirements 6.3, 6.4
# ---------------------------------------------------------------------------


@given(data=_uncertainty_filtered_data())
@settings(max_examples=100, deadline=None)
def test_property_p12_uncertainty_filtered_both_conditions(data):
    """P12: Uncertainty-filtered thresholding requires both conditions.

    **Validates: Requirements 6.3, 6.4**

    For any probability matrix, uncertainty matrix, confidence thresholds,
    and uncertainty thresholds, the UncertaintyFilteredStrategy shall accept
    a pseudo-label for entry (i, j) if and only if
    probability(i, j) > confidence_threshold(j) AND
    uncertainty(i, j) < uncertainty_threshold(j).
    Entries failing either condition shall be rejected.
    """
    probabilities, uncertainties, conf_thresholds, unc_thresholds, class_names = data
    n, c = probabilities.shape

    strategy = UncertaintyFilteredStrategy(
        confidence_thresholds=conf_thresholds,
        uncertainty_thresholds=unc_thresholds,
    )

    pseudo_labels, rejection_mask = strategy.accept(
        probabilities=probabilities,
        uncertainties=uncertainties,
        class_names=class_names,
    )

    # Shape checks
    assert pseudo_labels.shape == (n, c)
    assert rejection_mask.shape == (n, c)

    # Verify per-element correctness
    for j, name in enumerate(class_names):
        conf_thresh = conf_thresholds[name]
        unc_thresh = unc_thresholds[name]

        for i in range(n):
            prob = probabilities[i, j]
            unc = uncertainties[i, j]
            label = pseudo_labels[i, j]
            rejected = rejection_mask[i, j]

            conf_ok = prob > conf_thresh
            unc_ok = unc < unc_thresh
            should_accept = conf_ok and unc_ok

            if should_accept:
                assert label == 1.0, (
                    f"({i},{j}) prob={prob} > conf={conf_thresh} AND "
                    f"unc={unc} < unc_thresh={unc_thresh} but label={label}, "
                    f"expected 1.0"
                )
                assert not rejected, (
                    f"({i},{j}) should be accepted but rejection_mask is True"
                )
            else:
                assert np.isnan(label), (
                    f"({i},{j}) should be rejected (conf_ok={conf_ok}, "
                    f"unc_ok={unc_ok}) but label={label}, expected NaN"
                )
                assert rejected, (
                    f"({i},{j}) should be rejected but rejection_mask is False"
                )
