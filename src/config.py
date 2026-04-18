"""Configuration dataclasses and YAML config loader.

Defines all experiment configuration as nested dataclasses and provides
a loader that parses YAML files into a typed ExperimentConfig using dacite.

Validates: Requirements 9.1, 9.2, 9.3, 9.4, 11.5
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import dacite
import yaml


@dataclass
class DataConfig:
    """Dataset paths, label set, and preprocessing policies."""

    dataset_path: str
    train_csv: str
    valid_csv: str
    label_set: List[str]
    uncertain_policy: str  # "zeros" | "ones"
    view_policy: str  # "frontal_only" | "all"
    image_size: int
    num_workers: int


@dataclass
class SplitConfig:
    """Low-label split generation parameters."""

    labeled_ratio: float
    seed: int
    splits_dir: str
    min_positive_per_class: int = 1


@dataclass
class ModelConfig:
    """Backbone selection, classifier head, and normalization parameters."""

    backbone: str  # "resnet50_imagenet" | "resnet50_lvmmed"
    pretrained_weights_path: Optional[str]
    num_classes: int
    dropout_rate: float
    normalization_mean: List[float]
    normalization_std: List[float]


@dataclass
class TrainingConfig:
    """Training hyperparameters and resume settings."""

    learning_rate: float
    batch_size: int
    num_epochs: int
    optimizer: str  # "adam" | "sgd"
    weight_decay: float
    scheduler: Optional[str] = None  # "cosine" | "step" | None
    resume_checkpoint: Optional[str] = None


@dataclass
class PseudoLabelConfig:
    """Pseudo-labeling strategy and thresholds."""

    enabled: bool
    confidence_thresholds: Dict[str, float]
    pseudo_label_policy: str  # "positive_only" | "positive_negative"
    student_init: str  # "from_scratch" | "warm_start"
    student_pretrained_weights_path: Optional[str] = None


@dataclass
class UncertaintyConfig:
    """MC Dropout uncertainty estimation settings."""

    enabled: bool
    mc_dropout_passes: int
    uncertainty_metric: str  # "predictive_entropy" | "variance"
    uncertainty_thresholds: Dict[str, float]


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration aggregating all sub-configs."""

    experiment_name: str
    setting: str  # "supervised" | "lvmmed" | "pseudo_label" | "uncertainty_filter"
    data: DataConfig
    split: SplitConfig
    model: ModelConfig
    training: TrainingConfig
    pseudo_label: PseudoLabelConfig
    uncertainty: UncertaintyConfig
    output_dir: str
    optional_metrics: List[str] = field(default_factory=list)
    git_commit: Optional[str] = None  # Auto-populated at runtime


def load_config(yaml_path: str) -> ExperimentConfig:
    """Load a YAML config file and parse it into an ExperimentConfig.

    Args:
        yaml_path: Path to the YAML configuration file.

    Returns:
        A fully typed ExperimentConfig instance.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        dacite.exceptions.DaciteError: If required fields are missing or types mismatch.
    """
    with open(yaml_path, "r") as f:
        yaml_dict = yaml.safe_load(f)

    return dacite.from_dict(
        data_class=ExperimentConfig,
        data=yaml_dict,
        config=dacite.Config(strict=True),
    )
