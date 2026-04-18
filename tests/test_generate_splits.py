"""Unit tests for scripts/generate_splits.py CLI entry point.

Uses a tiny synthetic CheXpert-like CSV + images to exercise the
full generate_splits flow without real data.
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from scripts.generate_splits import main


def _create_tiny_chexpert(root, n=30):
    """Create a minimal CheXpert-like directory with CSV and images."""
    img_dir = os.path.join(root, "images")
    os.makedirs(img_dir, exist_ok=True)

    label_cols = [
        "Cardiomegaly",
        "Pleural Effusion",
        "Pneumothorax",
        "Consolidation",
        "Atelectasis",
        "Edema",
    ]

    rows = []
    rng = np.random.RandomState(0)
    for i in range(n):
        fname = f"img_{i:04d}.jpg"
        fpath = os.path.join(img_dir, fname)
        Image.fromarray(
            rng.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        ).save(fpath)
        labels = rng.choice([0.0, 1.0], size=len(label_cols)).tolist()
        rows.append(
            {"Path": os.path.join("images", fname),
             "Frontal/Lateral": "Frontal",
             **{c: labels[j] for j, c in enumerate(label_cols)}}
        )

    # Ensure at least one positive per class
    for j, c in enumerate(label_cols):
        rows[j][c] = 1.0

    df = pd.DataFrame(rows)
    csv_path = os.path.join(root, "train.csv")
    df.to_csv(csv_path, index=False)
    return csv_path


def _write_config(root, csv_path, splits_dir):
    """Write a minimal YAML config pointing at the synthetic data."""
    cfg = f"""\
experiment_name: "test_split"
setting: "supervised"

data:
  dataset_path: "{root}"
  train_csv: "train.csv"
  valid_csv: "train.csv"
  label_set: ["Cardiomegaly", "Pleural Effusion", "Pneumothorax", "Consolidation", "Atelectasis", "Edema"]
  uncertain_policy: "zeros"
  view_policy: "frontal_only"
  image_size: 32
  num_workers: 0

split:
  labeled_ratio: 0.2
  seed: 42
  splits_dir: "{splits_dir}"
  min_positive_per_class: 1

model:
  backbone: "resnet50_imagenet"
  pretrained_weights_path: null
  num_classes: 6
  dropout_rate: 0.5
  normalization_mean: [0.485, 0.456, 0.406]
  normalization_std: [0.229, 0.224, 0.225]

training:
  learning_rate: 0.0001
  batch_size: 4
  num_epochs: 1
  optimizer: "adam"
  weight_decay: 0.0001

pseudo_label:
  enabled: false
  confidence_thresholds: {{}}
  pseudo_label_policy: "positive_only"
  student_init: "from_scratch"

uncertainty:
  enabled: false
  mc_dropout_passes: 5
  uncertainty_metric: "predictive_entropy"
  uncertainty_thresholds: {{}}

output_dir: "{root}/outputs"
"""
    cfg_path = os.path.join(root, "config.yaml")
    with open(cfg_path, "w") as f:
        f.write(cfg)
    return cfg_path


class TestGenerateSplitsCLI:
    def test_end_to_end(self, tmp_path):
        root = str(tmp_path)
        splits_dir = os.path.join(root, "splits")
        _create_tiny_chexpert(root, n=30)
        cfg_path = _write_config(root, "train.csv", splits_dir)

        # Should run without error
        main(["--config", cfg_path])

        # Split file should exist
        split_files = os.listdir(splits_dir)
        assert len(split_files) == 1
        assert split_files[0].endswith(".json")
