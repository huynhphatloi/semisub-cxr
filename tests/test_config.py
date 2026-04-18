"""Unit tests for src/config.py — config loading and validation.

Validates: Requirements 9.1, 9.2, 9.3, 9.4
"""

import dacite
import pytest
import yaml

from src.config import ExperimentConfig, load_config

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LABEL_SET = [
    "Cardiomegaly",
    "Pleural Effusion",
    "Pneumothorax",
    "Consolidation",
    "Atelectasis",
    "Edema",
]


def _assert_common_fields(cfg: ExperimentConfig):
    """Verify fields shared across all 4 config templates."""
    # Data section
    assert cfg.data.label_set == _LABEL_SET
    assert cfg.data.uncertain_policy == "zeros"
    assert cfg.data.view_policy == "frontal_only"
    assert cfg.data.image_size == 224
    assert cfg.data.num_workers == 2

    # Split section
    assert cfg.split.labeled_ratio == 0.05
    assert cfg.split.seed == 42
    assert cfg.split.min_positive_per_class == 1

    # Model section
    assert cfg.model.num_classes == 6
    assert cfg.model.dropout_rate == 0.5
    assert cfg.model.normalization_mean == [0.485, 0.456, 0.406]
    assert cfg.model.normalization_std == [0.229, 0.224, 0.225]

    # Training section
    assert cfg.training.learning_rate == 0.0001
    assert cfg.training.batch_size == 32
    assert cfg.training.num_epochs == 30
    assert cfg.training.optimizer == "adam"
    assert cfg.training.weight_decay == 0.0001
    assert cfg.training.scheduler == "cosine"

    # Top-level
    assert cfg.optional_metrics == []
    assert cfg.git_commit is None


# ---------------------------------------------------------------------------
# Loading each of the 4 YAML templates
# ---------------------------------------------------------------------------


def test_load_supervised_baseline():
    """supervised_baseline.yaml loads with correct setting and disabled extras."""
    cfg = load_config("configs/supervised_baseline.yaml")

    assert cfg.experiment_name == "supervised_baseline_5pct_seed42"
    assert cfg.setting == "supervised"
    assert cfg.model.backbone == "resnet50_imagenet"
    assert cfg.model.pretrained_weights_path is None

    assert cfg.pseudo_label.enabled is False
    assert cfg.pseudo_label.confidence_thresholds == {}
    assert cfg.uncertainty.enabled is False
    assert cfg.uncertainty.uncertainty_thresholds == {}

    _assert_common_fields(cfg)


def test_load_lvmmed_finetune():
    """lvmmed_finetune.yaml loads with LVM-Med backbone and disabled extras."""
    cfg = load_config("configs/lvmmed_finetune.yaml")

    assert cfg.experiment_name == "lvmmed_finetune_5pct_seed42"
    assert cfg.setting == "lvmmed"
    assert cfg.model.backbone == "resnet50_lvmmed"
    assert cfg.model.pretrained_weights_path == "/path/to/lvmmed_weights.pth"

    assert cfg.pseudo_label.enabled is False
    assert cfg.uncertainty.enabled is False

    _assert_common_fields(cfg)


def test_load_pseudo_label():
    """pseudo_label.yaml loads with pseudo-labeling enabled and per-class thresholds."""
    cfg = load_config("configs/pseudo_label.yaml")

    assert cfg.experiment_name == "pseudo_label_5pct_seed42"
    assert cfg.setting == "pseudo_label"
    assert cfg.model.backbone == "resnet50_lvmmed"

    assert cfg.pseudo_label.enabled is True
    assert cfg.pseudo_label.pseudo_label_policy == "positive_only"
    assert cfg.pseudo_label.student_init == "from_scratch"
    # Per-class thresholds present for all 6 labels
    thresholds = cfg.pseudo_label.confidence_thresholds
    assert set(thresholds.keys()) == set(_LABEL_SET)
    for val in thresholds.values():
        assert val == 0.7

    assert cfg.uncertainty.enabled is False

    _assert_common_fields(cfg)


def test_load_uncertainty_filter():
    """uncertainty_filter.yaml loads with both pseudo-labeling and
    uncertainty enabled."""
    cfg = load_config("configs/uncertainty_filter.yaml")

    assert cfg.experiment_name == "uncertainty_filter_5pct_seed42"
    assert cfg.setting == "uncertainty_filter"

    assert cfg.pseudo_label.enabled is True
    assert set(cfg.pseudo_label.confidence_thresholds.keys()) == set(_LABEL_SET)

    assert cfg.uncertainty.enabled is True
    assert cfg.uncertainty.mc_dropout_passes == 20
    assert cfg.uncertainty.uncertainty_metric == "predictive_entropy"
    assert set(cfg.uncertainty.uncertainty_thresholds.keys()) == set(_LABEL_SET)
    for val in cfg.uncertainty.uncertainty_thresholds.values():
        assert val == 0.5

    _assert_common_fields(cfg)


# ---------------------------------------------------------------------------
# Missing required fields raise dacite errors
# ---------------------------------------------------------------------------


def _minimal_config_dict():
    """Return a valid config dict that can be mutated for negative tests."""
    with open("configs/supervised_baseline.yaml", "r") as f:
        return yaml.safe_load(f)


def test_missing_top_level_field_raises():
    """Removing a required top-level field (e.g. 'setting') raises DaciteError."""
    d = _minimal_config_dict()
    del d["setting"]

    with pytest.raises(dacite.exceptions.DaciteError):
        dacite.from_dict(
            data_class=ExperimentConfig,
            data=d,
            config=dacite.Config(strict=True),
        )


def test_missing_nested_field_raises():
    """Removing a required nested field (e.g. data.dataset_path) raises DaciteError."""
    d = _minimal_config_dict()
    del d["data"]["dataset_path"]

    with pytest.raises(dacite.exceptions.DaciteError):
        dacite.from_dict(
            data_class=ExperimentConfig,
            data=d,
            config=dacite.Config(strict=True),
        )


def test_missing_entire_section_raises():
    """Removing an entire required section (e.g. 'model') raises DaciteError."""
    d = _minimal_config_dict()
    del d["model"]

    with pytest.raises(dacite.exceptions.DaciteError):
        dacite.from_dict(
            data_class=ExperimentConfig,
            data=d,
            config=dacite.Config(strict=True),
        )


def test_extra_unknown_field_raises_strict():
    """An extra unknown field raises DaciteError in strict mode."""
    d = _minimal_config_dict()
    d["unknown_field"] = "should_fail"

    with pytest.raises(dacite.exceptions.DaciteError):
        dacite.from_dict(
            data_class=ExperimentConfig,
            data=d,
            config=dacite.Config(strict=True),
        )


# ---------------------------------------------------------------------------
# Pseudo-label config with enabled=true and empty thresholds
# ---------------------------------------------------------------------------


def test_pseudo_label_enabled_with_empty_thresholds_loads():
    """Dacite does not validate that confidence_thresholds is non-empty when
    pseudo_label.enabled is True.  This test documents the current behaviour —
    the config loads successfully.  Semantic validation (e.g. requiring
    thresholds when enabled) could be added as a post-load check later.
    """
    d = _minimal_config_dict()
    d["pseudo_label"]["enabled"] = True
    d["pseudo_label"]["confidence_thresholds"] = {}

    cfg = dacite.from_dict(
        data_class=ExperimentConfig,
        data=d,
        config=dacite.Config(strict=True),
    )

    assert cfg.pseudo_label.enabled is True
    assert cfg.pseudo_label.confidence_thresholds == {}


# ---------------------------------------------------------------------------
# load_config file-not-found
# ---------------------------------------------------------------------------


def test_load_config_nonexistent_file_raises():
    """load_config raises FileNotFoundError for a missing YAML path."""
    with pytest.raises(FileNotFoundError):
        load_config("configs/does_not_exist.yaml")
