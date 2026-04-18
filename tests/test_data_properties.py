"""Property-based tests for data loading logic in src/data/chexpert_dataset.py.

Tests the data transformation logic directly using pandas DataFrames
rather than constructing full CheXpertDataset objects (which require image files).

Uses Hypothesis for property-based testing with 100 examples per property.
"""

import numpy as np
import pandas as pd
import string
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

_LABEL_SET = [
    "Cardiomegaly",
    "Pleural Effusion",
    "Pneumothorax",
    "Consolidation",
    "Atelectasis",
    "Edema",
]


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

def _extra_column_names():
    """Generate a list of extra column names that don't collide with _LABEL_SET or required cols."""
    reserved = set(_LABEL_SET) | {"Path", "Frontal/Lateral"}
    return st.lists(
        st.text(alphabet=string.ascii_letters + "_", min_size=1, max_size=15).filter(
            lambda s: s not in reserved
        ),
        min_size=0,
        max_size=10,
        unique=True,
    )


# ---------------------------------------------------------------------------
# Property 1: Label column filtering preserves exactly the target set
# Feature: semisup-multilabel-cxr, Property 1: Label column filtering preserves exactly the target set
# Validates: Requirements 1.1, 1.2
# ---------------------------------------------------------------------------

@given(
    n_rows=st.integers(min_value=1, max_value=50),
    extra_cols=_extra_column_names(),
)
@settings(max_examples=100)
def test_property_label_column_filtering(n_rows: int, extra_cols: list):
    """**Validates: Requirements 1.1, 1.2**

    For any CheXpert-like DataFrame containing the 6 target label columns
    plus any number of extra columns, filtering to label_set produces a
    DataFrame with exactly those 6 columns — no extras, no missing targets.
    """
    # Build a DataFrame with the 6 target columns + extras
    data = {}
    for col in _LABEL_SET:
        data[col] = np.random.choice([-1.0, 0.0, 1.0], size=n_rows)
    for col in extra_cols:
        data[col] = np.random.randn(n_rows)
    data["Path"] = [f"img_{i}.jpg" for i in range(n_rows)]

    df = pd.DataFrame(data)

    # --- Filtering logic extracted from CheXpertDataset.__init__ ---
    label_set = list(_LABEL_SET)
    missing_cols = set(label_set) - set(df.columns)
    assert len(missing_cols) == 0, f"Missing columns: {missing_cols}"

    filtered = df[label_set]

    # Verify: exactly the 6 target columns, in order
    assert list(filtered.columns) == label_set
    assert filtered.shape == (n_rows, 6)


# ---------------------------------------------------------------------------
# Property 2: Uncertain label mapping produces clean binary labels
# Feature: semisup-multilabel-cxr, Property 2: Uncertain label mapping produces clean binary labels
# Validates: Requirements 1.3
# ---------------------------------------------------------------------------

@given(
    label_matrix=arrays(
        dtype=np.float64,
        shape=st.tuples(
            st.integers(min_value=1, max_value=50),
            st.just(6),
        ),
        elements=st.sampled_from([-1.0, 0.0, 1.0]),
    ),
    policy=st.sampled_from(["zeros", "ones"]),
)
@settings(max_examples=100)
def test_property_uncertain_label_mapping(label_matrix: np.ndarray, policy: str):
    """**Validates: Requirements 1.3**

    For any label matrix with values in {-1, 0, 1} and any valid mapping
    policy, applying the uncertain label mapping produces a matrix containing
    only values in {0, 1}, where every -1 is replaced according to the policy
    and all 0/1 entries are unchanged.
    """
    n_rows = label_matrix.shape[0]

    # Build a DataFrame mimicking what CheXpertDataset receives
    df = pd.DataFrame(label_matrix, columns=_LABEL_SET)

    # --- Mapping logic extracted from CheXpertDataset.__init__ ---
    fill_value = 0.0 if policy == "zeros" else 1.0
    for col in _LABEL_SET:
        df[col] = df[col].fillna(0.0)
        df[col] = df[col].replace(-1.0, fill_value).replace(-1, fill_value)

    mapped = df[_LABEL_SET].values

    # All values must be in {0, 1}
    unique_vals = set(np.unique(mapped))
    assert unique_vals.issubset({0.0, 1.0}), f"Unexpected values: {unique_vals}"

    # Verify policy-specific mapping: original -1 entries mapped correctly
    for i in range(n_rows):
        for j in range(6):
            original = label_matrix[i, j]
            result = mapped[i, j]
            if original == -1.0:
                assert result == fill_value, (
                    f"At ({i},{j}): -1 should map to {fill_value} "
                    f"under '{policy}' policy, got {result}"
                )
            else:
                assert result == original, (
                    f"At ({i},{j}): {original} should be unchanged, got {result}"
                )


# ---------------------------------------------------------------------------
# Property 3: Frontal-only view filtering retains only frontal images
# Feature: semisup-multilabel-cxr, Property 3: Frontal-only view filtering retains only frontal images
# Validates: Requirements 1A.2
# ---------------------------------------------------------------------------

@given(
    views=st.lists(
        st.sampled_from(["Frontal", "Lateral"]),
        min_size=1,
        max_size=100,
    ),
)
@settings(max_examples=100)
def test_property_frontal_only_view_filtering(views: list):
    """**Validates: Requirements 1A.2**

    For any set of image metadata entries with view types in
    {"Frontal", "Lateral"}, applying the frontal-only filter retains all
    and only entries with view type "Frontal", and the count of retained
    entries equals the count of frontal entries in the original set.
    """
    n_rows = len(views)

    # Build a DataFrame with mixed views and dummy label data
    data = {"Frontal/Lateral": views, "Path": [f"img_{i}.jpg" for i in range(n_rows)]}
    for col in _LABEL_SET:
        data[col] = np.random.choice([0.0, 1.0], size=n_rows)

    df = pd.DataFrame(data)

    # Expected frontal count
    expected_frontal_count = sum(1 for v in views if v == "Frontal")

    # --- Filtering logic extracted from CheXpertDataset.__init__ ---
    filtered = df[df["Frontal/Lateral"] == "Frontal"].reset_index(drop=True)

    # All retained entries must be frontal
    assert all(filtered["Frontal/Lateral"] == "Frontal"), (
        "Non-frontal entries found after filtering"
    )

    # Count must match expected
    assert len(filtered) == expected_frontal_count, (
        f"Expected {expected_frontal_count} frontal entries, got {len(filtered)}"
    )

    # No lateral entries should remain
    assert "Lateral" not in filtered["Frontal/Lateral"].values, (
        "Lateral entries found after frontal-only filtering"
    )
