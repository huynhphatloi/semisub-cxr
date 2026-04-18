"""Integration tests for the semi-supervised multi-label CXR pipeline.

Uses tiny synthetic data (TensorDataset with random features) and simple
nn.Linear models instead of full ResNet50 to keep tests fast. No real
CheXpert images are needed.

Validates: Requirements 3, 3A, 5, 6, 7, 11.4, 15A
"""

import os
import tempfile
from typing import Optional

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    PseudoLabelConfig,
    SplitConfig,
    TrainingConfig,
    UncertaintyConfig,
)
from src.data.combined_dataset import CombinedDataset
from src.evaluation.metrics import compute_auroc
from src.pseudo_labeling.mc_dropout import run_mc_dropout_inference
from src.pseudo_labeling.teacher_inference import run_teacher_inference
from src.pseudo_labeling.threshold_strategy import (
    PositiveOnlyStrategy,
    UncertaintyFilteredStrategy,
)
from src.training.trainer import Trainer, TrainResult
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.sanity_checks import (
    check_data_loading,
    check_evaluation,
    check_split_validity,
    check_training_step,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_SAMPLES = 20
INPUT_DIM = 4
NUM_CLASSES = 6
BATCH_SIZE = 4
LABEL_NAMES = [
    "Cardiomegaly",
    "Pleural Effusion",
    "Pneumothorax",
    "Consolidation",
    "Atelectasis",
    "Edema",
]
DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(
    output_dir: str,
    num_epochs: int = 2,
    setting: str = "supervised",
    pseudo_enabled: bool = False,
    uncertainty_enabled: bool = False,
    resume_checkpoint: Optional[str] = None,
) -> ExperimentConfig:
    """Build a minimal ExperimentConfig for integration tests."""
    conf_thresholds = {n: 0.5 for n in LABEL_NAMES} if pseudo_enabled else {}
    unc_thresholds = {n: 0.5 for n in LABEL_NAMES} if uncertainty_enabled else {}

    return ExperimentConfig(
        experiment_name="integration_test",
        setting=setting,
        data=DataConfig(
            dataset_path="/tmp/fake",
            train_csv="train.csv",
            valid_csv="valid.csv",
            label_set=LABEL_NAMES,
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
            num_classes=NUM_CLASSES,
            dropout_rate=0.5,
            normalization_mean=[0.5, 0.5, 0.5],
            normalization_std=[0.5, 0.5, 0.5],
        ),
        training=TrainingConfig(
            learning_rate=0.01,
            batch_size=BATCH_SIZE,
            num_epochs=num_epochs,
            optimizer="adam",
            weight_decay=0.0,
            scheduler=None,
            resume_checkpoint=resume_checkpoint,
        ),
        pseudo_label=PseudoLabelConfig(
            enabled=pseudo_enabled,
            confidence_thresholds=conf_thresholds,
            pseudo_label_policy="positive_only",
            student_init="from_scratch",
        ),
        uncertainty=UncertaintyConfig(
            enabled=uncertainty_enabled,
            mc_dropout_passes=3,
            uncertainty_metric="predictive_entropy",
            uncertainty_thresholds=unc_thresholds,
        ),
        output_dir=output_dir,
    )


def _make_simple_model(input_dim: int = INPUT_DIM, num_classes: int = NUM_CLASSES) -> nn.Module:
    """Simple linear model standing in for the full backbone+classifier."""
    return nn.Linear(input_dim, num_classes)


def _make_dropout_model(input_dim: int = INPUT_DIM, num_classes: int = NUM_CLASSES) -> nn.Module:
    """Linear model with dropout for MC Dropout tests."""
    return nn.Sequential(
        nn.Linear(input_dim, 16),
        nn.Dropout(0.5),
        nn.Linear(16, num_classes),
    )


def _make_synthetic_data(
    num_samples: int = NUM_SAMPLES,
    input_dim: int = INPUT_DIM,
    num_classes: int = NUM_CLASSES,
    seed: int = 0,
):
    """Create synthetic features and binary labels with both 0s and 1s per class."""
    torch.manual_seed(seed)
    X = torch.randn(num_samples, input_dim)
    # Ensure each class has at least one positive and one negative
    y = torch.zeros(num_samples, num_classes)
    for c in range(num_classes):
        # Set roughly half to positive
        pos_count = max(2, num_samples // 3)
        y[:pos_count, c] = 1.0
    # Shuffle rows
    perm = torch.randperm(num_samples)
    y = y[perm]
    return X, y


def _make_3tuple_loader(X, y, batch_size=BATCH_SIZE):
    """DataLoader yielding (features, labels, indices)."""
    indices = torch.arange(len(X))
    ds = TensorDataset(X, y, indices)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def _make_4tuple_loader(X, y, masks, batch_size=BATCH_SIZE):
    """DataLoader yielding (features, labels, loss_mask, is_pseudo)."""
    is_pseudo = torch.zeros(len(X))
    ds = TensorDataset(X, y, masks, is_pseudo)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestSupervisedEndToEnd:
    """End-to-end supervised: train 2 epochs, verify checkpoint + metrics."""

    def test_train_and_checkpoint(self):
        """Train 2 epochs on tiny data, verify checkpoints and metrics saved."""
        X, y = _make_synthetic_data()
        train_loader = _make_3tuple_loader(X, y)
        val_loader = _make_3tuple_loader(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_config(output_dir=tmpdir, num_epochs=2)
            model = _make_simple_model()
            trainer = Trainer(config, model, train_loader, val_loader, DEVICE)
            result = trainer.train()

            # Verify TrainResult
            assert isinstance(result, TrainResult)
            assert result.best_epoch >= 0
            assert result.best_epoch < 2
            assert 0.0 <= result.best_macro_auroc <= 1.0
            assert len(result.per_class_auroc) > 0

            # Verify checkpoint files exist
            ckpt_dir = os.path.join(
                tmpdir, "checkpoints", "supervised", "0.1_42"
            )
            best_path = os.path.join(ckpt_dir, "best_checkpoint.pt")
            last_path = os.path.join(ckpt_dir, "last_checkpoint.pt")
            assert os.path.isfile(best_path)
            assert os.path.isfile(last_path)

            # Verify checkpoint can be loaded
            model2 = _make_simple_model()
            meta = load_checkpoint(last_path, model2, device="cpu")
            assert "epoch" in meta
            assert "val_metrics" in meta
            assert meta["epoch"] == 1  # 0-indexed, last epoch of 2

    def test_resume_training(self):
        """Train 2 epochs, resume from checkpoint, train 1 more."""
        X, y = _make_synthetic_data()
        train_loader = _make_3tuple_loader(X, y)
        val_loader = _make_3tuple_loader(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_config(output_dir=tmpdir, num_epochs=2)
            model = _make_simple_model()
            trainer = Trainer(config, model, train_loader, val_loader, DEVICE)
            trainer.train()

            ckpt_path = os.path.join(
                tmpdir, "checkpoints", "supervised", "0.1_42",
                "last_checkpoint.pt",
            )

            # Resume
            config2 = _make_config(
                output_dir=tmpdir, num_epochs=3,
                resume_checkpoint=ckpt_path,
            )
            model2 = _make_simple_model()
            trainer2 = Trainer(config2, model2, train_loader, val_loader, DEVICE)
            assert trainer2.start_epoch == 2
            result = trainer2.train()
            assert isinstance(result, TrainResult)


@pytest.mark.integration
class TestPseudoLabelEndToEnd:
    """End-to-end pseudo-label: teacher train → inference → threshold → student train."""

    def test_pseudo_label_pipeline(self):
        X, y = _make_synthetic_data()
        train_loader = _make_3tuple_loader(X, y)
        val_loader = _make_3tuple_loader(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Step 1: Train teacher
            config = _make_config(
                output_dir=tmpdir, num_epochs=2, setting="pseudo_label",
                pseudo_enabled=True,
            )
            teacher = _make_simple_model()
            trainer = Trainer(config, teacher, train_loader, val_loader, DEVICE)
            teacher_result = trainer.train()
            assert isinstance(teacher_result, TrainResult)

            # Step 2: Teacher inference on "unlabeled" data
            unlabeled_X, unlabeled_y = _make_synthetic_data(seed=99)
            unlabeled_loader = _make_3tuple_loader(unlabeled_X, unlabeled_y)

            teacher.eval()
            sample_indices, probabilities = run_teacher_inference(
                teacher, unlabeled_loader, DEVICE,
            )
            assert sample_indices.shape == (NUM_SAMPLES,)
            assert probabilities.shape == (NUM_SAMPLES, NUM_CLASSES)
            assert np.all(probabilities >= 0.0) and np.all(probabilities <= 1.0)

            # Step 3: Apply threshold strategy
            strategy = PositiveOnlyStrategy(config.pseudo_label.confidence_thresholds)
            pseudo_labels, rejection_mask = strategy.accept(
                probabilities, None, LABEL_NAMES,
            )
            assert pseudo_labels.shape == (NUM_SAMPLES, NUM_CLASSES)
            assert rejection_mask.shape == (NUM_SAMPLES, NUM_CLASSES)

            # Step 4: Build CombinedDataset and train student
            # Create a pseudo-label DataFrame
            pseudo_df = pd.DataFrame({"sample_index": sample_indices.astype(int)})
            for j, name in enumerate(LABEL_NAMES):
                pseudo_df[f"pseudo_{name}"] = pseudo_labels[:, j]
                pseudo_df[f"prob_{name}"] = probabilities[:, j]
                pseudo_df[f"rejected_{name}"] = rejection_mask[:, j]

            # Use a simple TensorDataset as the labeled dataset
            labeled_ds = TensorDataset(X, y, torch.arange(len(X)))
            combined = CombinedDataset(
                labeled_dataset=labeled_ds,
                pseudo_label_df=pseudo_df,
                label_set=LABEL_NAMES,
            )
            assert len(combined) == NUM_SAMPLES + NUM_SAMPLES

            # Verify labeled samples preserve ground-truth
            for i in range(min(5, NUM_SAMPLES)):
                img, lbl, mask, is_pseudo = combined[i]
                assert not is_pseudo
                assert torch.allclose(lbl, y[i])
                assert torch.all(mask == 1.0)

            # Build a tensor-based 4-tuple loader for student training
            # (CombinedDataset returns image tensors for pseudo samples,
            # but our simple linear model expects flat features, so we
            # construct a synthetic combined loader directly.)
            pseudo_label_tensor = torch.from_numpy(
                np.nan_to_num(pseudo_labels, nan=0.0)
            ).float()
            pseudo_mask_tensor = torch.from_numpy(
                (~rejection_mask).astype(np.float32)
            )
            combined_X = torch.cat([X, unlabeled_X], dim=0)
            combined_y = torch.cat([y, pseudo_label_tensor], dim=0)
            combined_mask = torch.cat(
                [torch.ones(NUM_SAMPLES, NUM_CLASSES), pseudo_mask_tensor], dim=0,
            )
            combined_is_pseudo = torch.cat(
                [torch.zeros(NUM_SAMPLES), torch.ones(NUM_SAMPLES)], dim=0,
            )
            combined_loader = _make_4tuple_loader(
                combined_X, combined_y, combined_mask,
            )

            student_config = _make_config(
                output_dir=tmpdir, num_epochs=2, setting="pseudo_label",
                pseudo_enabled=True,
            )
            student = _make_simple_model()
            student_trainer = Trainer(
                student_config, student, combined_loader, val_loader, DEVICE,
            )
            student_result = student_trainer.train()
            assert isinstance(student_result, TrainResult)
            assert student_result.best_macro_auroc >= 0.0


@pytest.mark.integration
class TestUncertaintyEndToEnd:
    """End-to-end uncertainty: teacher train → MC Dropout → filter → student train."""

    def test_uncertainty_pipeline(self):
        X, y = _make_synthetic_data()
        train_loader = _make_3tuple_loader(X, y)
        val_loader = _make_3tuple_loader(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Step 1: Train teacher (with dropout)
            config = _make_config(
                output_dir=tmpdir, num_epochs=2,
                setting="uncertainty_filter",
                pseudo_enabled=True, uncertainty_enabled=True,
            )
            teacher = _make_dropout_model()
            trainer = Trainer(config, teacher, train_loader, val_loader, DEVICE)
            teacher_result = trainer.train()
            assert isinstance(teacher_result, TrainResult)

            # Step 2: MC Dropout inference
            unlabeled_X, unlabeled_y = _make_synthetic_data(seed=77)
            unlabeled_loader = _make_3tuple_loader(unlabeled_X, unlabeled_y)

            sample_indices, mean_probs, uncertainties = run_mc_dropout_inference(
                teacher, unlabeled_loader, DEVICE,
                num_passes=3, uncertainty_metric="predictive_entropy",
            )
            assert sample_indices.shape == (NUM_SAMPLES,)
            assert mean_probs.shape == (NUM_SAMPLES, NUM_CLASSES)
            assert uncertainties.shape == (NUM_SAMPLES, NUM_CLASSES)
            assert np.all(uncertainties >= 0.0)

            # Step 3: Apply uncertainty-filtered threshold
            strategy = UncertaintyFilteredStrategy(
                confidence_thresholds=config.pseudo_label.confidence_thresholds,
                uncertainty_thresholds=config.uncertainty.uncertainty_thresholds,
            )
            pseudo_labels, rejection_mask = strategy.accept(
                mean_probs, uncertainties, LABEL_NAMES,
            )
            assert pseudo_labels.shape == (NUM_SAMPLES, NUM_CLASSES)

            # Step 4: Build CombinedDataset and train student
            # Verify CombinedDataset construction works
            pseudo_df = pd.DataFrame({"sample_index": sample_indices.astype(int)})
            for j, name in enumerate(LABEL_NAMES):
                pseudo_df[f"pseudo_{name}"] = pseudo_labels[:, j]
                pseudo_df[f"prob_{name}"] = mean_probs[:, j]
                pseudo_df[f"rejected_{name}"] = rejection_mask[:, j]

            labeled_ds = TensorDataset(X, y, torch.arange(len(X)))
            combined = CombinedDataset(
                labeled_dataset=labeled_ds,
                pseudo_label_df=pseudo_df,
                label_set=LABEL_NAMES,
            )
            assert len(combined) == NUM_SAMPLES + NUM_SAMPLES

            # Build a tensor-based 4-tuple loader for student training
            pseudo_label_tensor = torch.from_numpy(
                np.nan_to_num(pseudo_labels, nan=0.0)
            ).float()
            pseudo_mask_tensor = torch.from_numpy(
                (~rejection_mask).astype(np.float32)
            )
            combined_X = torch.cat([X, unlabeled_X], dim=0)
            combined_y = torch.cat([y, pseudo_label_tensor], dim=0)
            combined_mask = torch.cat(
                [torch.ones(NUM_SAMPLES, NUM_CLASSES), pseudo_mask_tensor], dim=0,
            )
            combined_loader = _make_4tuple_loader(
                combined_X, combined_y, combined_mask,
            )

            student = _make_dropout_model()
            student_trainer = Trainer(
                config, student, combined_loader, val_loader, DEVICE,
            )
            student_result = student_trainer.train()
            assert isinstance(student_result, TrainResult)


@pytest.mark.integration
class TestEvaluationStandalone:
    """Evaluation standalone: load checkpoint, run evaluation, verify results."""

    def test_evaluate_from_checkpoint(self):
        """Save a checkpoint, load it, compute metrics, verify CSV output."""
        X, y = _make_synthetic_data()
        train_loader = _make_3tuple_loader(X, y)
        val_loader = _make_3tuple_loader(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Train and save checkpoint
            config = _make_config(output_dir=tmpdir, num_epochs=2)
            model = _make_simple_model()
            trainer = Trainer(config, model, train_loader, val_loader, DEVICE)
            trainer.train()

            # Load checkpoint and evaluate manually
            ckpt_path = os.path.join(
                tmpdir, "checkpoints", "supervised", "0.1_42",
                "best_checkpoint.pt",
            )
            model2 = _make_simple_model()
            meta = load_checkpoint(ckpt_path, model2, device="cpu")
            model2.eval()

            # Run inference
            all_labels = []
            all_probs = []
            with torch.no_grad():
                for batch in val_loader:
                    images, labels = batch[0], batch[1]
                    logits = model2(images)
                    probs = torch.sigmoid(logits)
                    all_labels.append(labels.numpy())
                    all_probs.append(probs.numpy())

            y_true = np.concatenate(all_labels, axis=0)
            y_score = np.concatenate(all_probs, axis=0)

            # Compute metrics
            metrics = compute_auroc(y_true, y_score, LABEL_NAMES)
            assert "macro_auroc" in metrics
            assert "per_class_auroc" in metrics
            assert 0.0 <= metrics["macro_auroc"] <= 1.0

            # Save results CSV
            results_dir = os.path.join(tmpdir, "results")
            os.makedirs(results_dir, exist_ok=True)
            row = {
                "setting": config.setting,
                "labeled_ratio": config.split.labeled_ratio,
                "seed": config.split.seed,
                "macro_auroc": metrics["macro_auroc"],
            }
            for name, auc in metrics["per_class_auroc"].items():
                row[f"auroc_{name}"] = auc

            csv_path = os.path.join(results_dir, "evaluation_results.csv")
            pd.DataFrame([row]).to_csv(csv_path, index=False)
            assert os.path.isfile(csv_path)

            # Verify CSV content
            df = pd.read_csv(csv_path)
            assert len(df) == 1
            assert "macro_auroc" in df.columns
            assert df["setting"].iloc[0] == "supervised"


@pytest.mark.integration
class TestSanityChecks:
    """Run all 4 sanity check functions on valid synthetic data."""

    def test_check_data_loading(self):
        """check_data_loading passes on a well-formed dataset."""
        X, y = _make_synthetic_data()
        indices = torch.arange(len(X))
        ds = TensorDataset(X, y, indices)
        assert check_data_loading(ds) is True

    def test_check_split_validity(self):
        """check_split_validity passes on a correct split."""
        from src.data.split_generator import SplitMetadata

        n = 100
        labeled_ratio = 0.1
        labeled_size = round(labeled_ratio * n)
        all_indices = list(range(n))
        labeled_indices = all_indices[:labeled_size]
        unlabeled_indices = all_indices[labeled_size:]

        meta = SplitMetadata(
            seed=42,
            labeled_ratio=labeled_ratio,
            total_train_size=n,
            labeled_size=labeled_size,
            unlabeled_size=n - labeled_size,
            labeled_indices=labeled_indices,
            unlabeled_indices=unlabeled_indices,
            per_class_positive_counts={name: 2 for name in LABEL_NAMES},
            generation_timestamp="2025-01-01T00:00:00Z",
        )
        assert check_split_validity(meta, n, labeled_ratio) is True

    def test_check_training_step(self):
        """check_training_step passes with a valid model and batch."""
        model = _make_simple_model()
        X, y = _make_synthetic_data(num_samples=8)
        assert check_training_step(model, (X, y), DEVICE) is True

    def test_check_evaluation(self):
        """check_evaluation passes with a trained model and val loader."""
        X, y = _make_synthetic_data()
        model = _make_simple_model()
        # Do a quick training pass so model produces varied outputs
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        model.train()
        for _ in range(5):
            logits = model(X)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_loader = _make_3tuple_loader(X, y)
        assert check_evaluation(None, model, val_loader) is True
