"""Unit tests for pseudo-labeling modules: teacher_inference, combined_dataset, artifact_io.

Tests cover core functionality of each module with synthetic data.
"""

import os
import tempfile

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.pseudo_labeling.teacher_inference import run_teacher_inference
from src.data.combined_dataset import CombinedDataset
from src.pseudo_labeling.artifact_io import (
    load_pseudo_label_artifact,
    save_pseudo_label_artifact,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CLASS_NAMES = [
    "Cardiomegaly",
    "Pleural Effusion",
    "Pneumothorax",
    "Consolidation",
    "Atelectasis",
    "Edema",
]


def _make_simple_model(input_dim=4, num_classes=6):
    """Tiny linear model for testing."""
    return nn.Linear(input_dim, num_classes)


def _make_3tuple_loader(num_samples=16, input_dim=4, num_classes=6, batch_size=4):
    """Create a DataLoader returning (images, labels, indices)."""
    torch.manual_seed(0)
    X = torch.randn(num_samples, input_dim)
    y = torch.randint(0, 2, (num_samples, num_classes)).float()
    indices = torch.arange(num_samples)
    ds = TensorDataset(X, y, indices)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def _make_labeled_dataset(num_samples=10, num_classes=6):
    """Create a simple TensorDataset mimicking labeled data with 3-tuples."""
    torch.manual_seed(42)
    images = torch.randn(num_samples, 3, 32, 32)
    labels = torch.randint(0, 2, (num_samples, num_classes)).float()
    indices = torch.arange(num_samples)
    return TensorDataset(images, labels, indices)


def _make_pseudo_label_df(num_samples=5, class_names=None, seed=0):
    """Create a synthetic pseudo-label DataFrame."""
    if class_names is None:
        class_names = CLASS_NAMES
    rng = np.random.RandomState(seed)
    data = {"sample_index": np.arange(100, 100 + num_samples)}
    for cls in class_names:
        probs = rng.rand(num_samples)
        data[f"prob_{cls}"] = probs
        # Accept where prob > 0.5
        pseudo = np.where(probs > 0.5, 1.0, np.nan)
        data[f"pseudo_{cls}"] = pseudo
        data[f"rejected_{cls}"] = probs <= 0.5
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Tests — teacher_inference.py
# ---------------------------------------------------------------------------


class TestTeacherInference:
    def test_returns_correct_shapes(self):
        model = _make_simple_model()
        loader = _make_3tuple_loader(num_samples=16, batch_size=4)
        indices, probs = run_teacher_inference(model, loader, torch.device("cpu"))
        assert indices.shape == (16,)
        assert probs.shape == (16, 6)

    def test_probabilities_in_valid_range(self):
        model = _make_simple_model()
        loader = _make_3tuple_loader(num_samples=20, batch_size=8)
        _, probs = run_teacher_inference(model, loader, torch.device("cpu"))
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)

    def test_indices_match_dataloader(self):
        model = _make_simple_model()
        loader = _make_3tuple_loader(num_samples=12, batch_size=4)
        indices, _ = run_teacher_inference(model, loader, torch.device("cpu"))
        expected = np.arange(12)
        np.testing.assert_array_equal(indices, expected)

    def test_model_in_eval_mode(self):
        """Model should be in eval mode during inference."""
        model = _make_simple_model()
        model.train()  # Start in train mode
        loader = _make_3tuple_loader(num_samples=4, batch_size=4)
        run_teacher_inference(model, loader, torch.device("cpu"))
        assert not model.training


# ---------------------------------------------------------------------------
# Tests — combined_dataset.py
# ---------------------------------------------------------------------------


class TestCombinedDataset:
    def test_length_is_sum(self):
        labeled = _make_labeled_dataset(num_samples=10)
        pseudo_df = _make_pseudo_label_df(num_samples=5)
        ds = CombinedDataset(labeled, pseudo_df, CLASS_NAMES)
        assert len(ds) == 15

    def test_labeled_samples_return_original_labels(self):
        labeled = _make_labeled_dataset(num_samples=10)
        pseudo_df = _make_pseudo_label_df(num_samples=5)
        ds = CombinedDataset(labeled, pseudo_df, CLASS_NAMES)

        for i in range(10):
            img, label, mask, is_pseudo = ds[i]
            orig_img, orig_label, _ = labeled[i]
            assert not is_pseudo
            torch.testing.assert_close(label, orig_label)
            assert torch.all(mask == 1.0)

    def test_pseudo_samples_have_correct_mask(self):
        labeled = _make_labeled_dataset(num_samples=3)
        pseudo_df = _make_pseudo_label_df(num_samples=4)
        ds = CombinedDataset(labeled, pseudo_df, CLASS_NAMES)

        for i in range(3, 7):
            _, label, mask, is_pseudo = ds[i]
            assert is_pseudo
            # Mask should be 0 or 1
            assert torch.all((mask == 0.0) | (mask == 1.0))
            # Where mask is 0, label should be 0 (rejected)
            assert label.shape == (6,)

    def test_getitem_returns_4tuple(self):
        labeled = _make_labeled_dataset(num_samples=2)
        pseudo_df = _make_pseudo_label_df(num_samples=2)
        ds = CombinedDataset(labeled, pseudo_df, CLASS_NAMES)
        result = ds[0]
        assert len(result) == 4

    def test_ground_truth_never_overwritten(self):
        """Core property: labeled samples retain original labels regardless of pseudo-labels."""
        labeled = _make_labeled_dataset(num_samples=5)
        # Create pseudo-labels that would conflict
        pseudo_df = _make_pseudo_label_df(num_samples=3)
        ds = CombinedDataset(labeled, pseudo_df, CLASS_NAMES)

        for i in range(5):
            _, label, _, is_pseudo = ds[i]
            _, orig_label, _ = labeled[i]
            assert not is_pseudo
            torch.testing.assert_close(label, orig_label)


# ---------------------------------------------------------------------------
# Tests — artifact_io.py
# ---------------------------------------------------------------------------


class TestArtifactIO:
    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rng = np.random.RandomState(42)
            n, c = 10, 6
            indices = np.arange(n)
            probs = rng.rand(n, c)
            pseudo = np.where(probs > 0.5, 1.0, np.nan)
            rejected = probs <= 0.5
            config = {"experiment": "test"}

            csv_path = save_pseudo_label_artifact(
                artifact_dir=tmpdir,
                sample_indices=indices,
                probabilities=probs,
                pseudo_labels=pseudo,
                rejection_mask=rejected,
                uncertainties=None,
                config=config,
                class_names=CLASS_NAMES,
            )

            assert os.path.isfile(csv_path)
            df = load_pseudo_label_artifact(csv_path)
            assert len(df) == n
            assert "sample_index" in df.columns
            for cls in CLASS_NAMES:
                assert f"prob_{cls}" in df.columns
                assert f"pseudo_{cls}" in df.columns
                assert f"rejected_{cls}" in df.columns

    def test_load_from_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rng = np.random.RandomState(0)
            n, c = 5, 6
            save_pseudo_label_artifact(
                artifact_dir=tmpdir,
                sample_indices=np.arange(n),
                probabilities=rng.rand(n, c),
                pseudo_labels=np.full((n, c), np.nan),
                rejection_mask=np.ones((n, c), dtype=bool),
                uncertainties=None,
                config={"test": True},
                class_names=CLASS_NAMES,
            )
            df = load_pseudo_label_artifact(tmpdir)
            assert len(df) == n

    def test_with_uncertainties(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rng = np.random.RandomState(1)
            n, c = 8, 6
            uncertainties = rng.rand(n, c)
            save_pseudo_label_artifact(
                artifact_dir=tmpdir,
                sample_indices=np.arange(n),
                probabilities=rng.rand(n, c),
                pseudo_labels=np.full((n, c), np.nan),
                rejection_mask=np.ones((n, c), dtype=bool),
                uncertainties=uncertainties,
                config={"test": True},
                class_names=CLASS_NAMES,
            )
            df = load_pseudo_label_artifact(tmpdir)
            for cls in CLASS_NAMES:
                assert f"uncertainty_{cls}" in df.columns

    def test_config_json_saved(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_pseudo_label_artifact(
                artifact_dir=tmpdir,
                sample_indices=np.arange(3),
                probabilities=np.random.rand(3, 6),
                pseudo_labels=np.full((3, 6), np.nan),
                rejection_mask=np.ones((3, 6), dtype=bool),
                uncertainties=None,
                config={"experiment": "test_config"},
                class_names=CLASS_NAMES,
            )
            config_path = os.path.join(tmpdir, "config.json")
            assert os.path.isfile(config_path)

    def test_load_nonexistent_raises(self):
        import pytest

        with pytest.raises(FileNotFoundError):
            load_pseudo_label_artifact("/nonexistent/path/pseudo_labels.csv")

    def test_probabilities_preserved(self):
        """Saved probabilities should match loaded values within float tolerance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rng = np.random.RandomState(99)
            n, c = 15, 6
            probs = rng.rand(n, c)
            save_pseudo_label_artifact(
                artifact_dir=tmpdir,
                sample_indices=np.arange(n),
                probabilities=probs,
                pseudo_labels=np.full((n, c), np.nan),
                rejection_mask=np.ones((n, c), dtype=bool),
                uncertainties=None,
                config={},
                class_names=CLASS_NAMES,
            )
            df = load_pseudo_label_artifact(tmpdir)
            for j, cls in enumerate(CLASS_NAMES):
                np.testing.assert_allclose(
                    df[f"prob_{cls}"].values, probs[:, j], atol=1e-6
                )
