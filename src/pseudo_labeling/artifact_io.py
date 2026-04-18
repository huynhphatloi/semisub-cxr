"""Pseudo-label artifact persistence: save and load CSV + config JSON.

Artifacts are stored in a directory structure:
``outputs/pseudo_labels/{setting}/{ratio}_{seed}/``

Validates: Requirements 6A.1, 6A.2, 6A.3
"""

import dataclasses
import json
import logging
import os
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def save_pseudo_label_artifact(
    artifact_dir: str,
    sample_indices: np.ndarray,
    probabilities: np.ndarray,
    pseudo_labels: np.ndarray,
    rejection_mask: np.ndarray,
    uncertainties: Optional[np.ndarray],
    config,
    class_names: Optional[List[str]] = None,
) -> str:
    """Save pseudo-label artifacts as CSV + config JSON.

    Args:
        artifact_dir: Directory to save artifacts in.
        sample_indices: 1-D int array of shape ``(N,)``.
        probabilities: 2-D float array of shape ``(N, C)``.
        pseudo_labels: 2-D float array of shape ``(N, C)`` with NaN for rejected.
        rejection_mask: 2-D bool array of shape ``(N, C)`` — True = rejected.
        uncertainties: Optional 2-D float array of shape ``(N, C)``.
        config: ExperimentConfig (or any object convertible via ``dataclasses.asdict``).
        class_names: Optional list of class names. If None, uses config.data.label_set.

    Returns:
        Path to the saved CSV file.
    """
    os.makedirs(artifact_dir, exist_ok=True)

    if class_names is None:
        class_names = config.data.label_set

    n, c = probabilities.shape

    # Build DataFrame
    data = {"sample_index": sample_indices.astype(int)}

    for j, cls in enumerate(class_names):
        data[f"prob_{cls}"] = probabilities[:, j]
        data[f"pseudo_{cls}"] = pseudo_labels[:, j]
        data[f"rejected_{cls}"] = rejection_mask[:, j]

    if uncertainties is not None:
        for j, cls in enumerate(class_names):
            data[f"uncertainty_{cls}"] = uncertainties[:, j]

    df = pd.DataFrame(data)

    csv_path = os.path.join(artifact_dir, "pseudo_labels.csv")
    df.to_csv(csv_path, index=False)
    logger.info("Saved pseudo-label CSV to %s (%d samples)", csv_path, n)

    # Save config JSON
    config_path = os.path.join(artifact_dir, "config.json")
    try:
        config_dict = dataclasses.asdict(config)
    except TypeError:
        # Fallback for non-dataclass configs (e.g. dicts in tests)
        config_dict = config if isinstance(config, dict) else {"config": str(config)}

    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2, default=str)
    logger.info("Saved config JSON to %s", config_path)

    return csv_path


def load_pseudo_label_artifact(artifact_path: str) -> pd.DataFrame:
    """Load a previously saved pseudo-label artifact.

    Args:
        artifact_path: Path to the ``pseudo_labels.csv`` file, or the
            directory containing it.

    Returns:
        DataFrame with columns: sample_index, prob_{class}, pseudo_{class},
        rejected_{class}, and optionally uncertainty_{class}.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
    """
    if os.path.isdir(artifact_path):
        artifact_path = os.path.join(artifact_path, "pseudo_labels.csv")

    if not os.path.isfile(artifact_path):
        raise FileNotFoundError(
            f"Pseudo-label artifact not found: {artifact_path}"
        )

    df = pd.read_csv(artifact_path)
    logger.info(
        "Loaded pseudo-label artifact from %s (%d samples)",
        artifact_path,
        len(df),
    )
    return df
