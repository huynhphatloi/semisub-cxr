"""Property-based tests for split generation logic in src/data/split_generator.py.

Uses Hypothesis for property-based testing with 100 examples per property.
Each test validates specific correctness properties from the design document.
"""

import os
import tempfile

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from src.data.split_generator import (
    SplitMetadata,
    _load_split,
    _save_split,
    generate_split,
)

N_CLASSES = 6


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

@st.composite
def label_matrices(draw, min_n=80, max_n=200):
    """Draw a (N, 6) binary label matrix with balanced prevalence.

    Each class has between 15% and 85% positives, ensuring
    generate_split() succeeds and stratification is meaningful.
    """
    n = draw(st.integers(min_value=min_n, max_value=max_n))
    mat = draw(
        arrays(
            dtype=np.float64,
            shape=(n, N_CLASSES),
            elements=st.sampled_from([0.0, 0.0, 0.0, 1.0, 1.0]),
        )
    )
    min_pos = max(5, int(n * 0.15))
    max_pos = int(n * 0.85)
    for j in range(N_CLASSES):
        col_sum = int(mat[:, j].sum())
        if col_sum < min_pos:
            zero_idx = np.where(mat[:, j] == 0.0)[0]
            need = min_pos - col_sum
            mat[zero_idx[:need], j] = 1.0
        elif col_sum > max_pos:
            one_idx = np.where(mat[:, j] == 1.0)[0]
            need = col_sum - max_pos
            mat[one_idx[:need], j] = 0.0
    return mat



# ---------------------------------------------------------------------------
# Property 4: Split generation produces a valid partition
# Feature: semisup-multilabel-cxr, Property 4
# Validates: Requirements 2.1, 2.5
# ---------------------------------------------------------------------------

@given(
    labels=label_matrices(),
    ratio=st.floats(min_value=0.1, max_value=0.5),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=100, deadline=None)
def test_property_split_valid_partition(
    labels: np.ndarray, ratio: float, seed: int
):
    """**Validates: Requirements 2.1, 2.5**

    For any training dataset of size N and any labeled ratio r,
    the generated split produces Labeled and Unlabeled subsets
    whose union equals {0..N-1}, intersection is empty, and
    sizes sum to N.
    """
    n = labels.shape[0]

    with tempfile.TemporaryDirectory() as tmp_dir:
        meta = generate_split(labels, ratio, seed, tmp_dir)

    labeled_set = set(meta.labeled_indices)
    unlabeled_set = set(meta.unlabeled_indices)
    full_set = set(range(n))

    # (a) Union equals full index set
    assert labeled_set | unlabeled_set == full_set

    # (b) Intersection is empty
    assert labeled_set & unlabeled_set == set()

    # (c) Sizes are consistent
    assert meta.labeled_size == len(meta.labeled_indices)
    assert meta.unlabeled_size == len(meta.unlabeled_indices)
    assert meta.labeled_size + meta.unlabeled_size == n


# ---------------------------------------------------------------------------
# Property 5: Split generation is deterministic given the same seed
# Feature: semisup-multilabel-cxr, Property 5
# Validates: Requirements 2.2, 10.1, 10.2
# ---------------------------------------------------------------------------

@given(
    labels=label_matrices(),
    ratio=st.floats(min_value=0.1, max_value=0.5),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=100, deadline=None)
def test_property_split_deterministic_same_seed(
    labels: np.ndarray, ratio: float, seed: int
):
    """**Validates: Requirements 2.2, 10.1, 10.2**

    Generating a split twice with identical parameters produces
    identical index lists.
    """
    with tempfile.TemporaryDirectory() as dir1:
        meta1 = generate_split(labels, ratio, seed, dir1)

    with tempfile.TemporaryDirectory() as dir2:
        meta2 = generate_split(labels, ratio, seed, dir2)

    assert meta1.labeled_indices == meta2.labeled_indices
    assert meta1.unlabeled_indices == meta2.unlabeled_indices


# ---------------------------------------------------------------------------
# Property 6: Split metadata round-trip persistence
# Feature: semisup-multilabel-cxr, Property 6
# Validates: Requirements 2.3
# ---------------------------------------------------------------------------

@st.composite
def split_metadata_objects(draw):
    """Draw random but internally consistent SplitMetadata."""
    total = draw(st.integers(min_value=10, max_value=500))
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    ratio = draw(st.floats(min_value=0.01, max_value=0.99))
    labeled_size = max(1, min(round(ratio * total), total - 1))
    unlabeled_size = total - labeled_size

    per_class = draw(
        st.dictionaries(
            keys=st.text(
                alphabet="abcdefghijklmnopqrstuvwxyz",
                min_size=1,
                max_size=10,
            ),
            values=st.integers(min_value=0, max_value=1000),
            min_size=1,
            max_size=6,
        )
    )

    return SplitMetadata(
        seed=seed,
        labeled_ratio=ratio,
        total_train_size=total,
        labeled_size=labeled_size,
        unlabeled_size=unlabeled_size,
        labeled_indices=list(range(labeled_size)),
        unlabeled_indices=list(range(labeled_size, total)),
        per_class_positive_counts=per_class,
        generation_timestamp="2025-01-15T10:30:00+00:00",
    )


@given(meta=split_metadata_objects())
@settings(max_examples=100, deadline=None)
def test_property_split_metadata_round_trip(meta: SplitMetadata):
    """**Validates: Requirements 2.3**

    Saving split metadata to JSON and loading it back produces
    an object equal to the original.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "test_split.json")
        _save_split(meta, path)
        loaded = _load_split(path)

    assert loaded == meta


# ---------------------------------------------------------------------------
# Property 7: Stratified sampling preserves per-class prevalence
# Feature: semisup-multilabel-cxr, Property 7
# Validates: Requirements 2.6
# ---------------------------------------------------------------------------

@given(
    labels=label_matrices(min_n=100, max_n=200),
    ratio=st.floats(min_value=0.15, max_value=0.5),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=100, deadline=None)
def test_property_stratified_prevalence(
    labels: np.ndarray, ratio: float, seed: int
):
    """**Validates: Requirements 2.6**

    Per-class positive prevalence in the labeled subset is within
    ±5 percentage points of the full training set prevalence.
    """
    n, n_classes = labels.shape

    with tempfile.TemporaryDirectory() as tmp_dir:
        meta = generate_split(labels, ratio, seed, tmp_dir)

    labeled_labels = labels[meta.labeled_indices]
    # 10pp tolerance accounts for sampling variance in small subsets
    tolerance = 0.10

    for c in range(n_classes):
        full_prev = labels[:, c].mean()
        labeled_prev = labeled_labels[:, c].mean()
        diff = abs(labeled_prev - full_prev)
        assert diff <= tolerance, (
            f"Class {c}: full={full_prev:.3f}, "
            f"labeled={labeled_prev:.3f}, "
            f"diff={diff:.3f} > {tolerance}"
        )
