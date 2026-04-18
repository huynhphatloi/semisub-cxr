"""Unit tests for MC Dropout uncertainty estimation module.

Tests cover core functionality of mc_dropout.py with synthetic data.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.pseudo_labeling.mc_dropout import (
    _enable_dropout,
    compute_predictive_entropy,
    run_mc_dropout_inference,
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


class _SimpleModelWithDropout(nn.Module):
    """Tiny model with a dropout layer for testing."""

    def __init__(self, input_dim=4, num_classes=6, dropout_rate=0.5):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        return self.fc(self.dropout(x))


def _make_3tuple_loader(num_samples=16, input_dim=4, num_classes=6, batch_size=4):
    """Create a DataLoader returning (images, labels, indices)."""
    torch.manual_seed(0)
    X = torch.randn(num_samples, input_dim)
    y = torch.randint(0, 2, (num_samples, num_classes)).float()
    indices = torch.arange(num_samples)
    ds = TensorDataset(X, y, indices)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


# ---------------------------------------------------------------------------
# Tests — _enable_dropout
# ---------------------------------------------------------------------------


class TestEnableDropout:
    def test_model_in_eval_mode(self):
        model = _SimpleModelWithDropout()
        model.train()
        _enable_dropout(model)
        # Overall model should be in eval mode
        assert not model.fc.training

    def test_dropout_layers_in_train_mode(self):
        model = _SimpleModelWithDropout()
        _enable_dropout(model)
        assert model.dropout.training

    def test_non_dropout_layers_in_eval_mode(self):
        model = _SimpleModelWithDropout()
        _enable_dropout(model)
        assert not model.fc.training


# ---------------------------------------------------------------------------
# Tests — compute_predictive_entropy
# ---------------------------------------------------------------------------


class TestComputePredictiveEntropy:
    def test_output_shape(self):
        mc_probs = np.random.rand(5, 10, 6)
        entropy = compute_predictive_entropy(mc_probs)
        assert entropy.shape == (10, 6)

    def test_entropy_non_negative(self):
        mc_probs = np.random.rand(10, 20, 6)
        entropy = compute_predictive_entropy(mc_probs)
        assert np.all(entropy >= 0.0)

    def test_maximum_entropy_at_half(self):
        """Entropy is maximized when p = 0.5."""
        T, N, C = 5, 1, 1
        mc_probs = np.full((T, N, C), 0.5)
        entropy = compute_predictive_entropy(mc_probs)
        expected = np.log(2)  # Maximum binary entropy
        np.testing.assert_allclose(entropy[0, 0], expected, atol=1e-6)

    def test_low_entropy_near_zero(self):
        """Entropy should be near zero when p is close to 0."""
        T, N, C = 5, 1, 1
        mc_probs = np.full((T, N, C), 0.01)
        entropy = compute_predictive_entropy(mc_probs)
        assert entropy[0, 0] < 0.1

    def test_low_entropy_near_one(self):
        """Entropy should be near zero when p is close to 1."""
        T, N, C = 5, 1, 1
        mc_probs = np.full((T, N, C), 0.99)
        entropy = compute_predictive_entropy(mc_probs)
        assert entropy[0, 0] < 0.1

    def test_known_value(self):
        """Verify against a hand-computed value."""
        # p values: 0.2, 0.4, 0.6 → mean = 0.4
        mc_probs = np.array([[[0.2]], [[0.4]], [[0.6]]])
        entropy = compute_predictive_entropy(mc_probs)
        p = 0.4
        expected = -(p * np.log(p) + (1 - p) * np.log(1 - p))
        np.testing.assert_allclose(entropy[0, 0], expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Tests — run_mc_dropout_inference
# ---------------------------------------------------------------------------


class TestRunMCDropoutInference:
    def test_returns_correct_shapes(self):
        model = _SimpleModelWithDropout()
        loader = _make_3tuple_loader(num_samples=16, batch_size=4)
        indices, mean_probs, uncertainties = run_mc_dropout_inference(
            model, loader, torch.device("cpu"), num_passes=3
        )
        assert indices.shape == (16,)
        assert mean_probs.shape == (16, 6)
        assert uncertainties.shape == (16, 6)

    def test_probabilities_in_valid_range(self):
        model = _SimpleModelWithDropout()
        loader = _make_3tuple_loader(num_samples=20, batch_size=8)
        _, mean_probs, _ = run_mc_dropout_inference(
            model, loader, torch.device("cpu"), num_passes=5
        )
        assert np.all(mean_probs >= 0.0)
        assert np.all(mean_probs <= 1.0)

    def test_uncertainties_non_negative(self):
        model = _SimpleModelWithDropout()
        loader = _make_3tuple_loader(num_samples=12, batch_size=4)
        _, _, uncertainties = run_mc_dropout_inference(
            model, loader, torch.device("cpu"), num_passes=3
        )
        assert np.all(uncertainties >= 0.0)

    def test_indices_match_dataloader(self):
        model = _SimpleModelWithDropout()
        loader = _make_3tuple_loader(num_samples=12, batch_size=4)
        indices, _, _ = run_mc_dropout_inference(
            model, loader, torch.device("cpu"), num_passes=2
        )
        expected = np.arange(12)
        np.testing.assert_array_equal(indices, expected)

    def test_invalid_num_passes_raises(self):
        model = _SimpleModelWithDropout()
        loader = _make_3tuple_loader(num_samples=4, batch_size=4)
        with pytest.raises(ValueError, match="num_passes must be > 0"):
            run_mc_dropout_inference(
                model, loader, torch.device("cpu"), num_passes=0
            )

    def test_negative_num_passes_raises(self):
        model = _SimpleModelWithDropout()
        loader = _make_3tuple_loader(num_samples=4, batch_size=4)
        with pytest.raises(ValueError, match="num_passes must be > 0"):
            run_mc_dropout_inference(
                model, loader, torch.device("cpu"), num_passes=-1
            )

    def test_unsupported_metric_raises(self):
        model = _SimpleModelWithDropout()
        loader = _make_3tuple_loader(num_samples=4, batch_size=4)
        with pytest.raises(ValueError, match="Unsupported uncertainty metric"):
            run_mc_dropout_inference(
                model,
                loader,
                torch.device("cpu"),
                num_passes=2,
                uncertainty_metric="unknown_metric",
            )

    def test_dropout_enabled_during_inference(self):
        """Multiple passes should produce different outputs due to dropout."""
        torch.manual_seed(42)
        model = _SimpleModelWithDropout(dropout_rate=0.5)
        loader = _make_3tuple_loader(num_samples=8, batch_size=8)

        # Run with many passes — with dropout active, outputs should vary
        _, mean_probs, uncertainties = run_mc_dropout_inference(
            model, loader, torch.device("cpu"), num_passes=20
        )
        # Uncertainties should not all be zero (dropout causes variation)
        assert np.any(uncertainties > 0.0)
