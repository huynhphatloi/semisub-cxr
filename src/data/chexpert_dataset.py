"""CheXpert dataset loader for multi-label chest X-ray classification.

Loads the CheXpert CSV, filters to the target label set, applies uncertain
label mapping and view filtering, and serves (image, label_vector, index)
tuples suitable for PyTorch DataLoader.

Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1A.1, 1A.2, 1A.5
"""

import logging
import os
from typing import Callable, List, Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

_VALID_UNCERTAIN_POLICIES = ("zeros", "ones")
_VALID_VIEW_POLICIES = ("frontal_only", "all")


class CheXpertDataset(Dataset):
    """PyTorch Dataset for the CheXpert chest X-ray dataset.

    Parameters
    ----------
    csv_path : str
        Path to the CheXpert CSV file (train.csv or valid.csv).
    image_root : str
        Root directory for resolving relative image paths in the CSV.
    label_set : list[str]
        Target pathology columns to retain (e.g. the 6-label set).
    uncertain_policy : str
        How to map uncertain labels (-1): ``"zeros"`` maps to 0,
        ``"ones"`` maps to 1.
    view_policy : str
        ``"frontal_only"`` keeps only frontal views, ``"all"`` keeps
        everything.
    transform : callable or None
        Torchvision-style transform applied to each PIL image.
    """

    def __init__(
        self,
        csv_path: str,
        image_root: str,
        label_set: List[str],
        uncertain_policy: str,
        view_policy: str,
        transform: Optional[Callable] = None,
    ) -> None:
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        if uncertain_policy not in _VALID_UNCERTAIN_POLICIES:
            raise ValueError(
                f"Unknown uncertain_policy '{uncertain_policy}'. "
                f"Must be one of {_VALID_UNCERTAIN_POLICIES}."
            )

        if view_policy not in _VALID_VIEW_POLICIES:
            raise ValueError(
                f"Unknown view_policy '{view_policy}'. "
                f"Must be one of {_VALID_VIEW_POLICIES}."
            )

        self.image_root = image_root
        self.label_set = list(label_set)
        self.transform = transform

        # Load CSV
        df = pd.read_csv(csv_path)

        # Validate that all requested label columns exist
        missing_cols = set(self.label_set) - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"Label columns missing from CSV: {sorted(missing_cols)}"
            )

        # Filter views
        if view_policy == "frontal_only":
            if "Frontal/Lateral" not in df.columns:
                raise ValueError(
                    "CSV missing 'Frontal/Lateral' column required for "
                    "frontal_only view_policy."
                )
            df = df[df["Frontal/Lateral"] == "Frontal"].reset_index(drop=True)

        # Apply uncertain-label mapping on label columns
        fill_value = 0.0 if uncertain_policy == "zeros" else 1.0
        for col in self.label_set:
            df[col] = df[col].fillna(0.0)
            df[col] = df[col].replace(-1.0, fill_value).replace(-1, fill_value)

        # Filter out samples whose image files are missing
        valid_mask = []
        for path_val in df["Path"]:
            full_path = os.path.join(self.image_root, path_val)
            if os.path.isfile(full_path):
                valid_mask.append(True)
            else:
                logger.warning("Image file not found, skipping: %s", full_path)
                valid_mask.append(False)

        df = df[valid_mask].reset_index(drop=True)

        # Store only what we need
        self.image_paths: List[str] = df["Path"].tolist()
        self.labels = torch.FloatTensor(df[self.label_set].values)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Return (image_tensor, label_vector, sample_index).

        Parameters
        ----------
        index : int
            Sample index.

        Returns
        -------
        tuple[Tensor, Tensor, int]
            ``image_tensor`` after transform, ``label_vector`` as
            FloatTensor(6,), and the integer ``index``.
        """
        img_path = os.path.join(self.image_root, self.image_paths[index])
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label_vector = self.labels[index]
        return image, label_vector, index
