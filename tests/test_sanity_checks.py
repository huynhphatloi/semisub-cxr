"""Unit tests for src/utils/sanity_checks.py.

Tests all four sanity-check functions using lightweight synthetic data
so that no real CheXpert images or GPU are required.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.split_generator import SplitMetadata
from src.utils.sanity_checks import (
    check_data_loading,
    check_evaluation,
    check_split_validity,
    check_training_step,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeDataset:
    """Minimal dataset returning (image, label_vector, index) tuples."""

    def __init__(self, n=10, label_dim=6):
        self._n = n
        self._label_dim = label_dim

    def __len__(self):
        return self._n

    def __getitem__(self, idx):
        image = torch.randn(3, 32, 32)
        label = torch.zeros(self._label_dim)
        return image, label, idx


class _BadShapeDataset(_FakeDataset):
    """Returns labels with wrong shape."""

    def __getitem__(self, idx):
        image = torch.randn(3, 32, 32)
        label = torch.zeros(4)  # wrong dim
        return image, label, idx


class _BadTupleDataset(_FakeDataset):
    """Returns a 2-tuple instead of 3-tuple."""

    def __getitem__(self, idx):
        return torch.randn(3, 32, 32), torch.zeros(6)


class _TinyModel(nn.Module):
    """Minimal model: linear layer producing 6 logits."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3 * 32 * 32, 6)

    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))


# ---------------------------------------------------------------------------
# check_data_loading
# ---------------------------------------------------------------------------

class TestCheckDataLoading:
    def test_passes_on_valid_dataset(self):
        ds = _FakeDataset(n=10)
        assert check_data_loading(ds) is True

    def test_fails_on_empty_dataset(self):
        ds = _FakeDataset(n=0)
        with pytest.raises(ValueError, match="empty"):
            check_data_loading(ds)

    def test_fails_on_wrong_label_shape(self):
        ds = _BadShapeDataset(n=3)
        with pytest.raises(ValueError, match="shape"):
            check_data_loading(ds)

    def test_fails_on_wrong_tuple_length(self):
        ds = _BadTupleDataset(n=3)
        with pytest.raises(ValueError, match="3-tuple"):
            check_data_loading(ds)


# ---------------------------------------------------------------------------
# check_split_validity
# ---------------------------------------------------------------------------

def _make_split_meta(
    n=100, ratio=0.1, seed=42, class_counts=None
):
    labeled_size = round(ratio * n)
    labeled = list(range(labeled_size))
    unlabeled = list(range(labeled_size, n))
    if class_counts is None:
        class_counts = {f"class_{i}": max(1, labeled_size // 6) for i in range(6)}
    return SplitMetadata(
        seed=seed,
        labeled_ratio=ratio,
        total_train_size=n,
        labeled_size=labeled_size,
        unlabeled_size=n - labeled_size,
        labeled_indices=labeled,
        unlabeled_indices=unlabeled,
        per_class_positive_counts=class_counts,
        generation_timestamp="2025-01-01T00:00:00Z",
    )


class TestCheckSplitValidity:
    def test_passes_on_valid_split(self):
        meta = _make_split_meta()
        assert check_split_validity(meta, 100, 0.1) is True

    def test_fails_on_overlap(self):
        meta = _make_split_meta()
        # Inject overlap
        meta.unlabeled_indices[0] = meta.labeled_indices[0]
        with pytest.raises(ValueError, match="overlap"):
            check_split_validity(meta, 100, 0.1)

    def test_fails_on_missing_indices(self):
        meta = _make_split_meta()
        # Remove one index from unlabeled
        meta.unlabeled_indices = meta.unlabeled_indices[:-1]
        with pytest.raises(ValueError, match="cover"):
            check_split_validity(meta, 100, 0.1)

    def test_fails_on_zero_positive_count(self):
        counts = {f"class_{i}": 1 for i in range(6)}
        counts["class_3"] = 0
        meta = _make_split_meta(class_counts=counts)
        with pytest.raises(ValueError, match="class_3"):
            check_split_validity(meta, 100, 0.1)


# ---------------------------------------------------------------------------
# check_training_step
# ---------------------------------------------------------------------------

class TestCheckTrainingStep:
    def test_passes_on_valid_model(self):
        model = _TinyModel()
        images = torch.randn(4, 3, 32, 32)
        labels = torch.zeros(4, 6)
        device = torch.device("cpu")
        assert check_training_step(model, (images, labels), device) is True

    def test_fails_on_nan_loss(self):
        """A model that produces NaN logits should fail the check."""

        class _NanModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.p = nn.Parameter(torch.tensor(1.0))

            def forward(self, x):
                return torch.full(
                    (x.size(0), 6), float("nan")
                ) * self.p

        model = _NanModel()
        images = torch.randn(2, 3, 32, 32)
        labels = torch.zeros(2, 6)
        with pytest.raises(RuntimeError, match="not finite"):
            check_training_step(model, (images, labels), torch.device("cpu"))


# ---------------------------------------------------------------------------
# check_evaluation
# ---------------------------------------------------------------------------

class TestCheckEvaluation:
    def test_passes_on_valid_model_and_loader(self):
        model = _TinyModel()
        # Create a small val set with both positive and negative labels
        images = torch.randn(20, 3, 32, 32)
        labels = torch.zeros(20, 6)
        labels[:10, :] = 1.0  # first half positive
        indices = torch.arange(20)
        val_ds = TensorDataset(images, labels, indices)
        val_loader = DataLoader(val_ds, batch_size=10)
        assert check_evaluation(None, model, val_loader) is True

    def test_fails_on_empty_loader(self):
        model = _TinyModel()
        images = torch.randn(0, 3, 32, 32)
        labels = torch.zeros(0, 6)
        indices = torch.arange(0)
        val_ds = TensorDataset(images, labels, indices)
        val_loader = DataLoader(val_ds, batch_size=1)
        with pytest.raises(RuntimeError, match="no validation"):
            check_evaluation(None, model, val_loader)
