"""CLI entry point for generating pseudo-labels from a trained teacher model.

Usage::

    python -m scripts.generate_pseudo_labels \\
        --config configs/pseudo_label.yaml \\
        --teacher-checkpoint outputs/checkpoints/lvmmed/0.05_42/best_checkpoint.pt

Loads the teacher model, runs inference on the unlabeled set, applies the
configured threshold strategy, saves artifacts, and logs per-class counts.

Validates: Requirements 5.1, 5.2, 5.8, 6A.1
"""

import argparse
import logging
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.config import ExperimentConfig, load_config
from src.data.chexpert_dataset import CheXpertDataset
from src.data.split_generator import generate_split
from src.data.transforms import get_eval_transform
from src.models.backbone_factory import create_backbone
from src.models.classifier import MultiLabelClassifier
from src.pseudo_labeling.artifact_io import save_pseudo_label_artifact
from src.pseudo_labeling.teacher_inference import run_teacher_inference
from src.pseudo_labeling.threshold_strategy import (
    PositiveOnlyStrategy,
    UncertaintyFilteredStrategy,
)
from src.utils.checkpoint import load_checkpoint
from src.utils.seed import set_all_seeds

logger = logging.getLogger(__name__)


def build_teacher_model(
    config: ExperimentConfig, checkpoint_path: str, device: torch.device
) -> torch.nn.Module:
    """Build and load the teacher model from a checkpoint."""
    backbone = create_backbone(
        config.model.backbone,
        weights_path=config.model.pretrained_weights_path,
    )
    model = MultiLabelClassifier(
        backbone=backbone,
        feature_dim=2048,
        num_classes=config.model.num_classes,
        dropout_rate=config.model.dropout_rate,
    )
    load_checkpoint(checkpoint_path, model, device=str(device))
    model = model.to(device)
    logger.info("Loaded teacher model from %s", checkpoint_path)
    return model


def build_unlabeled_loader(config: ExperimentConfig) -> DataLoader:
    """Build a DataLoader over the unlabeled subset."""
    eval_transform = get_eval_transform(
        image_size=config.data.image_size,
        normalization_mean=config.model.normalization_mean,
        normalization_std=config.model.normalization_std,
    )

    train_csv = os.path.join(config.data.dataset_path, config.data.train_csv)
    full_train_dataset = CheXpertDataset(
        csv_path=train_csv,
        image_root=config.data.dataset_path,
        label_set=config.data.label_set,
        uncertain_policy=config.data.uncertain_policy,
        view_policy=config.data.view_policy,
        transform=eval_transform,
    )

    split_meta = generate_split(
        dataset=full_train_dataset,
        labeled_ratio=config.split.labeled_ratio,
        seed=config.split.seed,
        splits_dir=config.split.splits_dir,
        min_positive_per_class=config.split.min_positive_per_class,
        class_names=config.data.label_set,
    )

    unlabeled_subset = Subset(full_train_dataset, split_meta.unlabeled_indices)

    loader = DataLoader(
        unlabeled_subset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True,
    )
    return loader


def create_threshold_strategy(config: ExperimentConfig):
    """Create the appropriate threshold strategy from config."""
    if config.uncertainty.enabled:
        return UncertaintyFilteredStrategy(
            confidence_thresholds=config.pseudo_label.confidence_thresholds,
            uncertainty_thresholds=config.uncertainty.uncertainty_thresholds,
        )
    return PositiveOnlyStrategy(
        confidence_thresholds=config.pseudo_label.confidence_thresholds,
    )


def log_per_class_counts(
    pseudo_labels: np.ndarray,
    rejection_mask: np.ndarray,
    class_names: list,
) -> None:
    """Log per-class accepted/rejected counts."""
    n, c = rejection_mask.shape
    for j, cls in enumerate(class_names):
        accepted = int((~rejection_mask[:, j]).sum())
        rejected = int(rejection_mask[:, j].sum())
        logger.info(
            "  %s: %d accepted, %d rejected (of %d total)",
            cls,
            accepted,
            rejected,
            n,
        )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate pseudo-labels from a trained teacher model."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML experiment config file.",
    )
    parser.add_argument(
        "--teacher-checkpoint",
        type=str,
        required=True,
        help="Path to the teacher model checkpoint.",
    )
    args = parser.parse_args(argv)

    # Setup
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = load_config(args.config)
    logger.info("Loaded config: %s", config.experiment_name)

    set_all_seeds(config.split.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Build teacher model
    model = build_teacher_model(config, args.teacher_checkpoint, device)

    # Build unlabeled data loader
    unlabeled_loader = build_unlabeled_loader(config)

    # Run teacher inference
    logger.info("Running teacher inference on unlabeled set...")
    sample_indices, probabilities = run_teacher_inference(
        model, unlabeled_loader, device
    )
    logger.info("Inference complete: %d samples", len(sample_indices))

    # Run MC Dropout if uncertainty filtering is enabled
    uncertainties = None
    if config.uncertainty.enabled:
        from src.pseudo_labeling.mc_dropout import run_mc_dropout_inference
        logger.info(
            "Running MC Dropout uncertainty estimation (%d passes)...",
            config.uncertainty.mc_dropout_passes,
        )
        sample_indices, probabilities, uncertainties = run_mc_dropout_inference(
            model=model,
            dataloader=unlabeled_loader,
            device=device,
            num_passes=config.uncertainty.mc_dropout_passes,
            uncertainty_metric=config.uncertainty.uncertainty_metric,
        )
        logger.info("MC Dropout complete.")

    # Apply threshold strategy
    class_names = config.data.label_set
    strategy = create_threshold_strategy(config)
    pseudo_labels, rejection_mask = strategy.accept(
        probabilities=probabilities,
        uncertainties=uncertainties,
        class_names=class_names,
    )

    # Log per-class counts
    logger.info("Per-class pseudo-label counts:")
    log_per_class_counts(pseudo_labels, rejection_mask, class_names)

    # Save artifacts
    artifact_dir = os.path.join(
        config.output_dir,
        "pseudo_labels",
        config.setting,
        f"{config.split.labeled_ratio}_{config.split.seed}",
    )
    csv_path = save_pseudo_label_artifact(
        artifact_dir=artifact_dir,
        sample_indices=sample_indices,
        probabilities=probabilities,
        pseudo_labels=pseudo_labels,
        rejection_mask=rejection_mask,
        uncertainties=None,
        config=config,
        class_names=class_names,
    )

    logger.info("Pseudo-label artifacts saved to %s", artifact_dir)

    # Summary
    total_accepted = int((~rejection_mask).sum())
    total_possible = rejection_mask.size
    print("\n=== Pseudo-Label Generation Summary ===")
    print(f"  Experiment : {config.experiment_name}")
    print(f"  Samples    : {len(sample_indices)}")
    print(f"  Accepted   : {total_accepted} / {total_possible} entries")
    print(f"  Artifact   : {csv_path}")
    print("========================================\n")


if __name__ == "__main__":
    main()
