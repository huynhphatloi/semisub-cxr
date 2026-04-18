"""Property-based tests for pseudo-labeling logic.

Uses Hypothesis for property-based testing with 100 examples per property.
"""

import tempfile

import numpy as np
import pandas as pd
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from torch.utils.data import TensorDataset

from src.data.combined_dataset import CombinedDataset
from src.pseudo_labeling.artifact_io import (
    load_pseudo_label_artifact,
    save_pseudo_label_artifact,
)
from src.pseudo_labeling.threshold_strategy import PositiveOnlyStrategy


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

def _class_names_strategy(min_classes: int = 1, max_classes: int = 8):
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
def _prob_matrix_and_thresholds(draw):
    """Draw an (N, C) probability matrix and a per-class threshold dict.

    Returns (probabilities, thresholds, class_names).
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

    # Per-class thresholds in (0, 1) — strict inequality in accept logic
    thresholds = {}
    for name in class_names:
        thresholds[name] = draw(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
        )

    return probabilities, thresholds, class_names


# ---------------------------------------------------------------------------
# Property 9: Positive-only confidence thresholding accepts only
#              high-confidence positives
# Feature: semisup-multilabel-cxr, Property 9: Positive-only confidence
#          thresholding accepts only high-confidence positives
# Validates: Requirements 5.2, 5.3, 5.4
# ---------------------------------------------------------------------------

@given(data=_prob_matrix_and_thresholds())
@settings(max_examples=100, deadline=None)
def test_property_positive_only_thresholding(data):
    """**Validates: Requirements 5.2, 5.3, 5.4**

    For any (N, C) probability matrix and per-class confidence threshold dict,
    PositiveOnlyStrategy shall:
      (a) assign pseudo-label 1.0 where prob > threshold,
      (b) assign NaN where prob <= threshold,
      (c) apply thresholds independently per class.
    """
    probabilities, thresholds, class_names = data
    n, c = probabilities.shape

    strategy = PositiveOnlyStrategy(confidence_thresholds=thresholds)
    pseudo_labels, rejection_mask = strategy.accept(
        probabilities=probabilities,
        uncertainties=None,
        class_names=class_names,
    )

    # Shape checks
    assert pseudo_labels.shape == (n, c)
    assert rejection_mask.shape == (n, c)

    # (a) & (b): verify per-element correctness
    for j, name in enumerate(class_names):
        thresh = thresholds[name]
        for i in range(n):
            prob = probabilities[i, j]
            label = pseudo_labels[i, j]
            rejected = rejection_mask[i, j]

            if prob > thresh:
                # (a) accepted — pseudo-label is 1.0
                assert label == 1.0, (
                    f"({i},{j}) prob={prob} > thresh={thresh} but label={label}"
                )
                assert not rejected, (
                    f"({i},{j}) accepted but rejection_mask is True"
                )
            else:
                # (b) rejected — pseudo-label is NaN
                assert np.isnan(label), (
                    f"({i},{j}) prob={prob} <= thresh={thresh} but label={label}, "
                    f"expected NaN"
                )
                assert rejected, (
                    f"({i},{j}) rejected but rejection_mask is False"
                )

    # (c) Independence: changing one class's threshold doesn't affect others.
    # Pick a class to perturb and verify the other columns are unchanged.
    if c >= 2:
        perturbed_thresholds = dict(thresholds)
        target_class = class_names[0]
        # Flip the threshold to the opposite extreme
        old_thresh = perturbed_thresholds[target_class]
        perturbed_thresholds[target_class] = 1.0 - old_thresh

        perturbed_strategy = PositiveOnlyStrategy(
            confidence_thresholds=perturbed_thresholds
        )
        perturbed_labels, perturbed_mask = perturbed_strategy.accept(
            probabilities=probabilities,
            uncertainties=None,
            class_names=class_names,
        )

        # All columns except the first should be identical
        for j in range(1, c):
            for i in range(n):
                orig = pseudo_labels[i, j]
                pert = perturbed_labels[i, j]
                # Both NaN or both equal
                if np.isnan(orig):
                    assert np.isnan(pert), (
                        f"Independence violated at ({i},{j}): "
                        f"orig=NaN but perturbed={pert}"
                    )
                else:
                    assert orig == pert, (
                        f"Independence violated at ({i},{j}): "
                        f"orig={orig} != perturbed={pert}"
                    )
                assert rejection_mask[i, j] == perturbed_mask[i, j], (
                    f"Independence violated for rejection_mask at ({i},{j})"
                )


# ---------------------------------------------------------------------------
# Property 10: Ground-truth labels are never overwritten by pseudo-labels
# Feature: semisup-multilabel-cxr, Property 10: Ground-truth labels are
#          never overwritten by pseudo-labels
# Validates: Requirements 5.7, 13.1
# ---------------------------------------------------------------------------

_FIXED_CLASSES = ["ClassA", "ClassB", "ClassC"]


@st.composite
def _labeled_and_pseudo_data(draw):
    """Generate random labeled data + pseudo-labels for CombinedDataset.

    Returns (labeled_dataset, pseudo_label_df, class_names, num_labeled).
    """
    num_classes = 3  # Keep small for speed
    class_names = _FIXED_CLASSES

    num_labeled = draw(st.integers(min_value=1, max_value=20))
    num_pseudo = draw(st.integers(min_value=1, max_value=20))

    # Labeled dataset: (image, label_vector, index)
    images = torch.randn(num_labeled, 3, 8, 8)
    labels = draw(
        arrays(
            dtype=np.float32,
            shape=(num_labeled, num_classes),
            elements=st.sampled_from([0.0, 1.0]),
        )
    )
    labels_tensor = torch.from_numpy(labels).float()
    indices = torch.arange(num_labeled)
    labeled_dataset = TensorDataset(images, labels_tensor, indices)

    # Pseudo-label DataFrame
    data = {"sample_index": np.arange(1000, 1000 + num_pseudo)}
    for cls in class_names:
        probs = draw(
            arrays(
                dtype=np.float64,
                shape=(num_pseudo,),
                elements=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
            )
        )
        data[f"prob_{cls}"] = probs
        pseudo = np.where(probs > 0.5, 1.0, np.nan)
        data[f"pseudo_{cls}"] = pseudo
        data[f"rejected_{cls}"] = probs <= 0.5

    pseudo_df = pd.DataFrame(data)

    return labeled_dataset, pseudo_df, class_names, num_labeled


@given(data=_labeled_and_pseudo_data())
@settings(max_examples=100, deadline=None)
def test_property_p10_ground_truth_never_overwritten(data):
    """P10: Ground-truth labels are never overwritten by pseudo-labels.

    **Validates: Requirements 5.7, 13.1**

    For any labeled dataset and any set of pseudo-labels, merging them into
    a CombinedDataset shall preserve the original ground-truth label vectors
    for all samples that belong to the labeled subset.
    """
    labeled_dataset, pseudo_df, class_names, num_labeled = data

    combined = CombinedDataset(
        labeled_dataset=labeled_dataset,
        pseudo_label_df=pseudo_df,
        label_set=class_names,
    )

    # Total length should be labeled + pseudo
    assert len(combined) == num_labeled + len(pseudo_df)

    # Every labeled sample must retain its original label vector
    for i in range(num_labeled):
        img, label, mask, is_pseudo = combined[i]
        orig_img, orig_label, _ = labeled_dataset[i]

        assert not is_pseudo, f"Sample {i} should not be pseudo-labeled"
        assert torch.all(mask == 1.0), (
            f"Labeled sample {i} should have all-ones mask"
        )
        assert torch.equal(label, orig_label), (
            f"Sample {i}: label {label} != original {orig_label}. "
            f"Ground-truth was overwritten!"
        )


# ---------------------------------------------------------------------------
# Property 13: Pseudo-label artifact round-trip persistence
# Feature: semisup-multilabel-cxr, Property 13: Pseudo-label artifact
#          round-trip persistence
# Validates: Requirements 6A.1, 6A.3
# ---------------------------------------------------------------------------

_ARTIFACT_CLASSES = ["Alpha", "Beta", "Gamma"]


@st.composite
def _pseudo_label_artifact_data(draw):
    """Generate random pseudo-label artifact data.

    Returns (sample_indices, probabilities, pseudo_labels, rejection_mask,
             uncertainties_or_none, class_names).
    """
    num_classes = 3
    class_names = _ARTIFACT_CLASSES
    n = draw(st.integers(min_value=1, max_value=30))

    sample_indices = np.arange(n)

    probabilities = draw(
        arrays(
            dtype=np.float64,
            shape=(n, num_classes),
            elements=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        )
    )

    # Pseudo-labels: 1.0 where prob > 0.5, NaN otherwise
    pseudo_labels = np.where(probabilities > 0.5, 1.0, np.nan)
    rejection_mask = probabilities <= 0.5

    include_unc = draw(st.booleans())
    if include_unc:
        uncertainties = draw(
            arrays(
                dtype=np.float64,
                shape=(n, num_classes),
                elements=st.floats(
                    min_value=0.0, max_value=1.0, allow_nan=False
                ),
            )
        )
    else:
        uncertainties = None

    return (
        sample_indices,
        probabilities,
        pseudo_labels,
        rejection_mask,
        uncertainties,
        class_names,
    )


@given(data=_pseudo_label_artifact_data())
@settings(max_examples=100, deadline=None)
def test_property_p13_artifact_roundtrip(data):
    """P13: Pseudo-label artifact round-trip persistence.

    **Validates: Requirements 6A.1, 6A.3**

    For any pseudo-label artifact, saving to disk and loading back shall
    produce a DataFrame with identical values within float tolerance.
    """
    (
        sample_indices,
        probabilities,
        pseudo_labels,
        rejection_mask,
        uncertainties,
        class_names,
    ) = data

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = save_pseudo_label_artifact(
            artifact_dir=tmpdir,
            sample_indices=sample_indices,
            probabilities=probabilities,
            pseudo_labels=pseudo_labels,
            rejection_mask=rejection_mask,
            uncertainties=uncertainties,
            config={"test": True},
            class_names=class_names,
        )

        loaded_df = load_pseudo_label_artifact(csv_path)

        # Verify row count
        n = len(sample_indices)
        assert len(loaded_df) == n, (
            f"Row count mismatch: saved {n}, loaded {len(loaded_df)}"
        )

        # Verify sample indices
        np.testing.assert_array_equal(
            loaded_df["sample_index"].values, sample_indices
        )

        # Verify probabilities within float tolerance
        for j, cls in enumerate(class_names):
            loaded_probs = loaded_df[f"prob_{cls}"].values
            np.testing.assert_allclose(
                loaded_probs,
                probabilities[:, j],
                atol=1e-6,
                err_msg=f"Probability mismatch for {cls}",
            )

        # Verify pseudo-labels (NaN equality)
        for j, cls in enumerate(class_names):
            loaded_pseudo = loaded_df[f"pseudo_{cls}"].values
            orig_pseudo = pseudo_labels[:, j]
            for i in range(n):
                if np.isnan(orig_pseudo[i]):
                    assert np.isnan(loaded_pseudo[i]), (
                        f"Expected NaN at ({i}, {cls}), got {loaded_pseudo[i]}"
                    )
                else:
                    np.testing.assert_allclose(
                        loaded_pseudo[i],
                        orig_pseudo[i],
                        atol=1e-6,
                        err_msg=f"Pseudo-label mismatch at ({i}, {cls})",
                    )

        # Verify uncertainties if present
        if uncertainties is not None:
            for j, cls in enumerate(class_names):
                col = f"uncertainty_{cls}"
                assert col in loaded_df.columns, (
                    f"Missing uncertainty column: {col}"
                )
                np.testing.assert_allclose(
                    loaded_df[col].values,
                    uncertainties[:, j],
                    atol=1e-6,
                    err_msg=f"Uncertainty mismatch for {cls}",
                )
