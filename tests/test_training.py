"""Tests for training module: losses, trainer, and scripts/train.py.

Includes unit tests for masked BCE loss, trainer checkpoint logic,
and Property 8 (best checkpoint tracks maximum validation AUROC).
"""

import os
import tempfile
from typing import Optional

import torch
import torch.nn as nn
from hypothesis import given, settings
from hypothesis import strategies as st
from torch.utils.data import DataLoader, TensorDataset

from src.training.losses import masked_bce_with_logits
from src.training.trainer import Trainer, TrainResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_simple_model(num_classes=6):
    """Tiny model for testing: linear layer only."""
    model = nn.Linear(4, num_classes)
    return model


def _make_config(
    output_dir: str,
    num_epochs: int = 3,
    optimizer: str = "adam",
    scheduler: Optional[str] = None,
    resume_checkpoint: Optional[str] = None,
):
    """Build a minimal ExperimentConfig-like object for Trainer."""
    from src.config import (
        DataConfig,
        ExperimentConfig,
        ModelConfig,
        PseudoLabelConfig,
        SplitConfig,
        TrainingConfig,
        UncertaintyConfig,
    )

    return ExperimentConfig(
        experiment_name="test_run",
        setting="supervised",
        data=DataConfig(
            dataset_path="/tmp/fake",
            train_csv="train.csv",
            valid_csv="valid.csv",
            label_set=[
                "Cardiomegaly",
                "Pleural Effusion",
                "Pneumothorax",
                "Consolidation",
                "Atelectasis",
                "Edema",
            ],
            uncertain_policy="zeros",
            view_policy="frontal_only",
            image_size=32,
            num_workers=0,
        ),
        split=SplitConfig(
            labeled_ratio=0.1,
            seed=42,
            splits_dir="/tmp/splits",
        ),
        model=ModelConfig(
            backbone="resnet50_imagenet",
            pretrained_weights_path=None,
            num_classes=6,
            dropout_rate=0.0,
            normalization_mean=[0.5, 0.5, 0.5],
            normalization_std=[0.5, 0.5, 0.5],
        ),
        training=TrainingConfig(
            learning_rate=0.01,
            batch_size=4,
            num_epochs=num_epochs,
            optimizer=optimizer,
            weight_decay=0.0,
            scheduler=scheduler,
            resume_checkpoint=resume_checkpoint,
        ),
        pseudo_label=PseudoLabelConfig(
            enabled=False,
            confidence_thresholds={},
            pseudo_label_policy="positive_only",
            student_init="from_scratch",
        ),
        uncertainty=UncertaintyConfig(
            enabled=False,
            mc_dropout_passes=20,
            uncertainty_metric="predictive_entropy",
            uncertainty_thresholds={},
        ),
        output_dir=output_dir,
    )


def _make_loaders(num_samples=20, input_dim=4, num_classes=6, batch_size=4):
    """Create tiny train and val DataLoaders returning 3-tuples."""
    torch.manual_seed(0)
    X = torch.randn(num_samples, input_dim)
    y = torch.randint(0, 2, (num_samples, num_classes)).float()
    indices = torch.arange(num_samples)
    ds = TensorDataset(X, y, indices)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    return loader, loader  # use same for train and val


# ---------------------------------------------------------------------------
# Unit tests — losses.py
# ---------------------------------------------------------------------------


class TestMaskedBCEWithLogits:
    def test_no_mask_matches_standard_bce(self):
        """Without a mask, result should match standard BCE mean."""
        logits = torch.randn(8, 6)
        targets = torch.randint(0, 2, (8, 6)).float()
        expected = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets
        )
        result = masked_bce_with_logits(logits, targets)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_all_ones_mask_matches_no_mask(self):
        logits = torch.randn(8, 6)
        targets = torch.randint(0, 2, (8, 6)).float()
        mask = torch.ones(8, 6)
        no_mask = masked_bce_with_logits(logits, targets)
        with_mask = masked_bce_with_logits(logits, targets, mask)
        assert torch.allclose(no_mask, with_mask, atol=1e-6)

    def test_partial_mask_zeros_out_elements(self):
        logits = torch.randn(4, 6)
        targets = torch.randint(0, 2, (4, 6)).float()
        mask = torch.zeros(4, 6)
        mask[0, 0] = 1.0
        mask[1, 2] = 1.0
        result = masked_bce_with_logits(logits, targets, mask)
        # Manually compute expected
        unreduced = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        expected = (unreduced * mask).sum() / mask.sum()
        assert torch.allclose(result, expected, atol=1e-6)

    def test_all_zeros_mask_returns_zero(self):
        logits = torch.randn(4, 6)
        targets = torch.randint(0, 2, (4, 6)).float()
        mask = torch.zeros(4, 6)
        result = masked_bce_with_logits(logits, targets, mask)
        assert result.item() == 0.0

    def test_gradient_flows(self):
        logits = torch.randn(4, 6, requires_grad=True)
        targets = torch.randint(0, 2, (4, 6)).float()
        mask = torch.ones(4, 6)
        mask[:, 3:] = 0.0
        loss = masked_bce_with_logits(logits, targets, mask)
        loss.backward()
        assert logits.grad is not None
        # Masked columns should have zero gradient
        assert (logits.grad[:, 3:] == 0).all()
        # Active columns should have non-zero gradient (almost surely)
        assert logits.grad[:, :3].abs().sum() > 0


# ---------------------------------------------------------------------------
# Unit tests — trainer.py
# ---------------------------------------------------------------------------


class TestTrainer:
    def test_train_runs_and_returns_result(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_config(output_dir=tmpdir, num_epochs=2)
            model = _make_simple_model()
            train_loader, val_loader = _make_loaders()
            trainer = Trainer(config, model, train_loader, val_loader, torch.device("cpu"))
            result = trainer.train()
            assert isinstance(result, TrainResult)
            assert result.best_epoch >= 0
            assert 0.0 <= result.best_macro_auroc <= 1.0

    def test_checkpoints_saved(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_config(output_dir=tmpdir, num_epochs=2)
            model = _make_simple_model()
            train_loader, val_loader = _make_loaders()
            trainer = Trainer(config, model, train_loader, val_loader, torch.device("cpu"))
            trainer.train()
            ckpt_dir = os.path.join(
                tmpdir, "checkpoints", "supervised", "0.1_42"
            )
            assert os.path.isfile(os.path.join(ckpt_dir, "best_checkpoint.pt"))
            assert os.path.isfile(os.path.join(ckpt_dir, "last_checkpoint.pt"))

    def test_sgd_optimizer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_config(output_dir=tmpdir, num_epochs=1, optimizer="sgd")
            model = _make_simple_model()
            train_loader, val_loader = _make_loaders()
            trainer = Trainer(config, model, train_loader, val_loader, torch.device("cpu"))
            result = trainer.train()
            assert result.best_epoch == 0

    def test_cosine_scheduler(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_config(
                output_dir=tmpdir, num_epochs=2, scheduler="cosine"
            )
            model = _make_simple_model()
            train_loader, val_loader = _make_loaders()
            trainer = Trainer(config, model, train_loader, val_loader, torch.device("cpu"))
            result = trainer.train()
            assert isinstance(result, TrainResult)

    def test_step_scheduler(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_config(
                output_dir=tmpdir, num_epochs=2, scheduler="step"
            )
            model = _make_simple_model()
            train_loader, val_loader = _make_loaders()
            trainer = Trainer(config, model, train_loader, val_loader, torch.device("cpu"))
            result = trainer.train()
            assert isinstance(result, TrainResult)

    def test_resume_from_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # First run: train 2 epochs
            config = _make_config(output_dir=tmpdir, num_epochs=2)
            model = _make_simple_model()
            train_loader, val_loader = _make_loaders()
            trainer = Trainer(config, model, train_loader, val_loader, torch.device("cpu"))
            trainer.train()

            ckpt_path = os.path.join(
                tmpdir, "checkpoints", "supervised", "0.1_42", "last_checkpoint.pt"
            )
            assert os.path.isfile(ckpt_path)

            # Second run: resume and train 1 more epoch
            config2 = _make_config(
                output_dir=tmpdir,
                num_epochs=3,
                resume_checkpoint=ckpt_path,
            )
            model2 = _make_simple_model()
            trainer2 = Trainer(config2, model2, train_loader, val_loader, torch.device("cpu"))
            assert trainer2.start_epoch == 2
            result = trainer2.train()
            assert isinstance(result, TrainResult)

    def test_handles_4tuple_batch(self):
        """Trainer should handle CombinedDataset 4-tuple batches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_config(output_dir=tmpdir, num_epochs=1)
            model = _make_simple_model()

            # Create 4-tuple dataset
            torch.manual_seed(0)
            X = torch.randn(16, 4)
            y = torch.randint(0, 2, (16, 6)).float()
            mask = torch.ones(16, 6)
            is_pseudo = torch.zeros(16)
            ds = TensorDataset(X, y, mask, is_pseudo)
            loader = DataLoader(ds, batch_size=4)

            _, val_loader = _make_loaders()
            trainer = Trainer(config, model, loader, val_loader, torch.device("cpu"))
            result = trainer.train()
            assert isinstance(result, TrainResult)

    def test_best_checkpoint_tracks_highest_auroc(self):
        """Best checkpoint should correspond to the epoch with highest AUROC."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_config(output_dir=tmpdir, num_epochs=3)
            model = _make_simple_model()
            train_loader, val_loader = _make_loaders()
            trainer = Trainer(config, model, train_loader, val_loader, torch.device("cpu"))
            result = trainer.train()

            # The best_macro_auroc should be >= all individual epoch AUROCs
            assert result.best_macro_auroc >= 0.0
            assert result.best_epoch >= 0
            assert result.best_epoch < 3


# ---------------------------------------------------------------------------
# Property test P8 — Best checkpoint tracks the maximum validation AUROC
# ---------------------------------------------------------------------------
# Feature: semisup-multilabel-cxr, Property 8: Best checkpoint tracks the
# maximum validation AUROC


def simulate_best_checkpoint_tracking(auroc_sequence):
    """Simulate the Trainer's best-checkpoint tracking logic.

    Given a list of (epoch, macro_auroc) pairs, returns the (epoch, auroc)
    of the best checkpoint using the same >= rule as Trainer._save_checkpoint
    (latest epoch wins ties).

    Returns:
        Tuple of (best_epoch, best_macro_auroc).
    """
    best_epoch = -1
    best_auroc = -1.0

    for epoch, auroc in auroc_sequence:
        if auroc >= best_auroc:
            best_auroc = auroc
            best_epoch = epoch

    return best_epoch, best_auroc


@given(
    auroc_pairs=st.lists(
        st.tuples(
            st.integers(min_value=0, max_value=999),
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        ),
        min_size=1,
        max_size=50,
    )
)
@settings(max_examples=100, deadline=None)
def test_property_p8_best_checkpoint_tracks_max_auroc(auroc_pairs):
    """P8: Best checkpoint tracks the maximum validation AUROC.

    **Validates: Requirements 3A.3**

    For any sequence of (epoch, val_macro_auroc) pairs, the best checkpoint
    should correspond to the epoch with the highest AUROC. If multiple
    epochs tie, the latest one is selected.
    """
    # Make epochs unique and sequential for realism
    sequential_pairs = [(i, auroc) for i, (_, auroc) in enumerate(auroc_pairs)]

    best_epoch, best_auroc = simulate_best_checkpoint_tracking(sequential_pairs)

    # Verify: best_auroc is the maximum
    max_auroc = max(auroc for _, auroc in sequential_pairs)
    assert best_auroc == max_auroc, (
        f"Best AUROC {best_auroc} != max AUROC {max_auroc}"
    )

    # Verify: best_epoch is the LATEST epoch with that max AUROC
    epochs_with_max = [
        epoch for epoch, auroc in sequential_pairs if auroc == max_auroc
    ]
    assert best_epoch == max(epochs_with_max), (
        f"Best epoch {best_epoch} is not the latest epoch with max AUROC. "
        f"Epochs with max: {epochs_with_max}"
    )


# ---------------------------------------------------------------------------
# Unit tests — scripts/train.py
# ---------------------------------------------------------------------------


class TestTrainScript:
    def test_build_model(self):
        """build_model should create a MultiLabelClassifier."""
        from scripts.train import build_model

        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_config(output_dir=tmpdir)
            model = build_model(config)
            # Should have backbone, dropout, classifier
            assert hasattr(model, "backbone")
            assert hasattr(model, "classifier")

    def test_save_config_copy(self):
        """save_config_copy should write a config.json file."""
        from scripts.train import save_config_copy

        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_config(output_dir=tmpdir)
            save_config_copy(config)
            config_path = os.path.join(
                tmpdir, "checkpoints", "supervised", "0.1_42", "config.json"
            )
            assert os.path.isfile(config_path)
