"""CLI entry point for training a multi-label CXR classifier.

Usage::

    python -m scripts.train --config configs/supervised_baseline.yaml

Loads config, sets seeds, builds model, creates data loaders, runs a
training-step sanity check, then launches the full training loop.

Validates: Requirements 9.1, 10.1, 14.1, 14.2, 14.3, 14.4, 15.1
"""

import argparse
import json
import logging
import os

import torch
from torch.utils.data import DataLoader, Subset

from src.config import ExperimentConfig, load_config
from src.data.chexpert_dataset import CheXpertDataset
from src.data.combined_dataset import CombinedDataset
from src.data.split_generator import generate_split
from src.data.transforms import get_eval_transform, get_train_transform
from src.models.backbone_factory import create_backbone
from src.models.classifier import MultiLabelClassifier
from src.pseudo_labeling.artifact_io import load_pseudo_label_artifact
from src.training.trainer import Trainer
from src.utils.sanity_checks import check_training_step
from src.utils.seed import set_all_seeds

logger = logging.getLogger(__name__)


def build_model(config: ExperimentConfig) -> MultiLabelClassifier:
    """Create a MultiLabelClassifier from the experiment config."""
    backbone = create_backbone(
        config.model.backbone,
        weights_path=config.model.pretrained_weights_path,
    )
    return MultiLabelClassifier(
        backbone=backbone,
        feature_dim=2048,
        num_classes=config.model.num_classes,
        dropout_rate=config.model.dropout_rate,
    )


def build_data_loaders(config: ExperimentConfig):
    """Build train and validation DataLoaders from the config.

    For supervised setting: trains on labeled subset only.
    For pseudo_label / uncertainty_filter: trains on labeled + pseudo-labeled
    data using CombinedDataset.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    train_transform = get_train_transform(
        image_size=config.data.image_size,
        normalization_mean=config.model.normalization_mean,
        normalization_std=config.model.normalization_std,
    )
    eval_transform = get_eval_transform(
        image_size=config.data.image_size,
        normalization_mean=config.model.normalization_mean,
        normalization_std=config.model.normalization_std,
    )

    train_csv = os.path.join(config.data.dataset_path, config.data.train_csv)
    val_csv = os.path.join(config.data.dataset_path, config.data.valid_csv)

    full_train_dataset = CheXpertDataset(
        csv_path=train_csv,
        image_root=config.data.dataset_path,
        label_set=config.data.label_set,
        uncertain_policy=config.data.uncertain_policy,
        view_policy=config.data.view_policy,
        transform=train_transform,
    )

    val_dataset = CheXpertDataset(
        csv_path=val_csv,
        image_root=config.data.dataset_path,
        label_set=config.data.label_set,
        uncertain_policy=config.data.uncertain_policy,
        view_policy=config.data.view_policy,
        transform=eval_transform,
    )

    # Generate or load split
    split_meta = generate_split(
        dataset=full_train_dataset,
        labeled_ratio=config.split.labeled_ratio,
        seed=config.split.seed,
        splits_dir=config.split.splits_dir,
        min_positive_per_class=config.split.min_positive_per_class,
        class_names=config.data.label_set,
    )

    train_subset = Subset(full_train_dataset, split_meta.labeled_indices)

    # For pseudo-label / uncertainty settings, load pseudo-label artifacts
    # and create a CombinedDataset
    if config.pseudo_label.enabled and config.setting in (
        "pseudo_label", "uncertainty_filter"
    ):
        artifact_dir = os.path.join(
            config.output_dir,
            "pseudo_labels",
            config.setting,
            f"{config.split.labeled_ratio}_{config.split.seed}",
        )
        pseudo_csv = os.path.join(artifact_dir, "pseudo_labels.csv")

        if os.path.isfile(pseudo_csv):
            logger.info("Loading pseudo-label artifacts from %s", pseudo_csv)
            pseudo_df = load_pseudo_label_artifact(pseudo_csv)

            # Add image_path column from the unlabeled subset
            unlabeled_paths = [
                full_train_dataset.image_paths[i]
                for i in split_meta.unlabeled_indices
            ]
            # Map sample_index to image_path
            idx_to_path = {
                idx: path
                for idx, path in zip(
                    split_meta.unlabeled_indices, unlabeled_paths
                )
            }
            pseudo_df["image_path"] = pseudo_df["sample_index"].map(idx_to_path)

            train_dataset = CombinedDataset(
                labeled_dataset=train_subset,
                pseudo_label_df=pseudo_df,
                label_set=config.data.label_set,
                transform=train_transform,
                image_root=config.data.dataset_path,
            )
            logger.info(
                "Using CombinedDataset: %d labeled + %d pseudo-labeled",
                len(train_subset),
                len(pseudo_df),
            )
        else:
            logger.warning(
                "Pseudo-label artifacts not found at %s. "
                "Training on labeled data only.",
                pseudo_csv,
            )
            train_dataset = train_subset
    else:
        train_dataset = train_subset

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True,
        persistent_workers=config.data.num_workers > 0,
        prefetch_factor=2 if config.data.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True,
        persistent_workers=config.data.num_workers > 0,
        prefetch_factor=2 if config.data.num_workers > 0 else None,
    )

    return train_loader, val_loader


def save_config_copy(config: ExperimentConfig) -> None:
    """Save a JSON copy of the config alongside checkpoints."""
    ckpt_dir = os.path.join(
        config.output_dir,
        "checkpoints",
        config.setting,
        f"{config.split.labeled_ratio}_{config.split.seed}",
    )
    os.makedirs(ckpt_dir, exist_ok=True)
    import dataclasses

    config_path = os.path.join(ckpt_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(dataclasses.asdict(config), f, indent=2)
    logger.info("Config saved to %s", config_path)


def _find_last_checkpoint(config):
    """Auto-detect last_checkpoint.pt for resume after disconnect."""
    ckpt_dir = os.path.join(
        config.output_dir,
        "checkpoints",
        config.setting,
        f"{config.split.labeled_ratio}_{config.split.seed}",
    )
    last_path = os.path.join(ckpt_dir, "last_checkpoint.pt")
    if os.path.isfile(last_path):
        return last_path
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Train a multi-label CXR classifier."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML experiment config file.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Auto-resume from last_checkpoint.pt if it exists.",
    )
    args = parser.parse_args(argv)

    # Load config
    config = load_config(args.config)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger.info("Loaded config: %s", config.experiment_name)

    # Auto-resume: find last checkpoint
    if args.resume and config.training.resume_checkpoint is None:
        last_ckpt = _find_last_checkpoint(config)
        if last_ckpt:
            config.training.resume_checkpoint = last_ckpt
            logger.info("Auto-resume enabled: found %s", last_ckpt)
        else:
            logger.info("Auto-resume: no checkpoint found, training from scratch.")

    # Set seeds
    set_all_seeds(config.split.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Enable cuDNN auto-tuner for faster convolutions
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # Build model
    model = build_model(config)

    # Build data loaders
    train_loader, val_loader = build_data_loaders(config)

    # Sanity check: single training step (skip if resuming)
    if config.training.resume_checkpoint is None:
        sample_batch = next(iter(train_loader))
        images, labels = sample_batch[0], sample_batch[1]
        check_training_step(model, (images, labels), device)

    # Save config copy
    save_config_copy(config)

    # Train
    trainer = Trainer(config, model, train_loader, val_loader, device)
    result = trainer.train()

    # Summary
    print("\n=== Training Summary ===")
    print(f"  Experiment  : {config.experiment_name}")
    print(f"  Best epoch  : {result.best_epoch}")
    print(f"  Best AUROC  : {result.best_macro_auroc:.4f}")
    print("  Per-class AUROC:")
    for cls, auc in result.per_class_auroc.items():
        print(f"    {cls}: {auc:.4f}")
    print("========================\n")


if __name__ == "__main__":
    main()
