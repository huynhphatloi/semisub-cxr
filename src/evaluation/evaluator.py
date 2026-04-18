"""Standalone evaluation runner for saved checkpoints.

Loads a model from a checkpoint, runs inference on the validation set,
computes all metrics, and stores results in a structured format.

Validates: Requirements 7.1, 7.2, 7.3, 11.4
"""

import json
import logging
import os
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import ExperimentConfig
from src.data.chexpert_dataset import CheXpertDataset
from src.data.transforms import get_eval_transform
from src.evaluation.metrics import compute_auroc, compute_optional_metrics
from src.models.backbone_factory import create_backbone
from src.models.classifier import MultiLabelClassifier
from src.utils.checkpoint import load_checkpoint

logger = logging.getLogger(__name__)

_RESNET50_FEATURE_DIM = 2048


def evaluate_checkpoint(
    checkpoint_path: str,
    config: ExperimentConfig,
    device: torch.device,
) -> Dict[str, Any]:
    """Load model from checkpoint, run on val set, return all metrics.

    Parameters
    ----------
    checkpoint_path : str
        Path to the saved checkpoint file.
    config : ExperimentConfig
        Experiment configuration (used for model construction and data loading).
    device : torch.device
        Device to run inference on.

    Returns
    -------
    dict
        Contains ``macro_auroc``, ``per_class_auroc``, and any optional
        metrics requested in ``config.optional_metrics``.
    """
    # Build model
    backbone = create_backbone(
        config.model.backbone, config.model.pretrained_weights_path
    )
    model = MultiLabelClassifier(
        backbone=backbone,
        feature_dim=_RESNET50_FEATURE_DIM,
        num_classes=config.model.num_classes,
        dropout_rate=config.model.dropout_rate,
    )

    # Load checkpoint weights
    load_checkpoint(checkpoint_path, model, device=str(device))
    model = model.to(device)
    model.eval()

    # Build validation loader
    transform = get_eval_transform(
        image_size=config.data.image_size,
        normalization_mean=config.model.normalization_mean,
        normalization_std=config.model.normalization_std,
    )
    val_dataset = CheXpertDataset(
        csv_path=os.path.join(config.data.dataset_path, config.data.valid_csv),
        image_root=config.data.dataset_path,
        label_set=config.data.label_set,
        uncertain_policy=config.data.uncertain_policy,
        view_policy=config.data.view_policy,
        transform=transform,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
    )

    # Run inference
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in val_loader:
            images, labels = batch[0], batch[1]
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    y_true = np.concatenate(all_labels, axis=0)
    y_score = np.concatenate(all_probs, axis=0)

    # Compute AUROC
    label_names = config.data.label_set
    metrics = compute_auroc(y_true, y_score, label_names)

    # Compute optional metrics
    if config.optional_metrics:
        y_pred = (y_score >= 0.5).astype(np.float64)
        optional = compute_optional_metrics(
            y_true, y_pred, y_score, label_names, config.optional_metrics
        )
        metrics.update(optional)

    logger.info("Evaluation complete — macro_auroc=%.4f", metrics["macro_auroc"])
    return metrics


def save_evaluation_results(
    metrics: Dict[str, Any],
    config: ExperimentConfig,
    output_dir: str,
) -> str:
    """Save evaluation results as CSV and JSON.

    Parameters
    ----------
    metrics : dict
        Metrics dict returned by :func:`evaluate_checkpoint`.
    config : ExperimentConfig
        Experiment configuration.
    output_dir : str
        Directory where results files are written.

    Returns
    -------
    str
        Path to the saved CSV file.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Build a flat row for the CSV
    row: Dict[str, Any] = {
        "setting": config.setting,
        "labeled_ratio": config.split.labeled_ratio,
        "seed": config.split.seed,
        "macro_auroc": metrics["macro_auroc"],
    }
    for name, auc in metrics.get("per_class_auroc", {}).items():
        row[f"auroc_{name}"] = auc

    # Optional metrics
    for key in ("macro_f1", "map"):
        if key in metrics:
            row[key] = metrics[key]

    # Save CSV (append if exists)
    csv_path = os.path.join(output_dir, "evaluation_results.csv")
    df = pd.DataFrame([row])
    if os.path.isfile(csv_path):
        existing = pd.read_csv(csv_path)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(csv_path, index=False)

    # Save JSON for this run
    json_path = os.path.join(
        output_dir,
        f"eval_{config.setting}_{config.split.labeled_ratio}_{config.split.seed}.json",
    )
    with open(json_path, "w") as f:
        payload = {
            "config_name": config.experiment_name,
            **row,
            "per_class_auroc": metrics.get("per_class_auroc", {}),
        }
        json.dump(payload, f, indent=2, default=str)

    logger.info("Results saved to %s", csv_path)
    return csv_path
