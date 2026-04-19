"""Combined dataset merging labeled and pseudo-labeled samples.

Ground-truth labels from the labeled subset are never overwritten by
pseudo-labels. For pseudo-labeled samples, only classes with accepted
pseudo-labels contribute to the loss via a binary mask tensor.

Validates: Requirements 5.6, 5.7, 13.1
"""

import logging
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class CombinedDataset(Dataset):
    """Merges a labeled dataset with pseudo-labeled samples.

    For labeled samples, ``__getitem__`` returns the original ground-truth
    labels with an all-ones loss mask. For pseudo-labeled samples, it returns
    the pseudo-labels with a mask that is 1.0 only for accepted classes.

    Args:
        labeled_dataset: A dataset (e.g. ``CheXpertDataset`` or ``Subset``)
            whose ``__getitem__`` returns ``(image_tensor, label_vector, index)``
            or ``(image_tensor, label_vector)``.
        pseudo_label_df: DataFrame from ``artifact_io`` with columns like
            ``pseudo_{class}``, ``prob_{class}``, and ``rejected_{class}``.
            Each row corresponds to one unlabeled sample. Must contain a
            ``sample_index`` column and an ``image_path`` column.
        label_set: Ordered list of class names (e.g. the 6 CheXpert labels).
        transform: Optional torchvision-style transform applied to images.
        image_root: Root directory for resolving image paths in pseudo_label_df.
    """

    def __init__(
        self,
        labeled_dataset: Dataset,
        pseudo_label_df: pd.DataFrame,
        label_set: List[str],
        transform: Optional[Callable] = None,
        image_root: str = "",
    ) -> None:
        self.labeled_dataset = labeled_dataset
        self.label_set = list(label_set)
        self.transform = transform
        self.image_root = image_root
        self.num_labeled = len(labeled_dataset)

        # Parse pseudo-label DataFrame into arrays
        num_pseudo = len(pseudo_label_df)
        num_classes = len(self.label_set)

        self.pseudo_labels = np.zeros((num_pseudo, num_classes), dtype=np.float32)
        self.pseudo_masks = np.zeros((num_pseudo, num_classes), dtype=np.float32)

        for j, cls in enumerate(self.label_set):
            pseudo_col = f"pseudo_{cls}"
            rejected_col = f"rejected_{cls}"

            if pseudo_col in pseudo_label_df.columns:
                vals = pseudo_label_df[pseudo_col].values
                for i in range(num_pseudo):
                    if not np.isnan(vals[i]):
                        self.pseudo_labels[i, j] = float(vals[i])
                        self.pseudo_masks[i, j] = 1.0
                    # else: label stays 0.0, mask stays 0.0 (rejected)
            elif rejected_col in pseudo_label_df.columns:
                rejected = pseudo_label_df[rejected_col].values
                for i in range(num_pseudo):
                    if not rejected[i]:
                        self.pseudo_masks[i, j] = 1.0

        # Store image paths for pseudo-labeled samples if available
        self.pseudo_image_paths: Optional[List[str]] = None
        if "image_path" in pseudo_label_df.columns:
            self.pseudo_image_paths = pseudo_label_df["image_path"].tolist()

        self.pseudo_label_df = pseudo_label_df

        logger.info(
            "CombinedDataset: %d labeled + %d pseudo-labeled = %d total",
            self.num_labeled,
            num_pseudo,
            len(self),
        )

    def __len__(self) -> int:
        return self.num_labeled + len(self.pseudo_label_df)

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        """Return ``(image_tensor, label_vector, loss_mask, is_pseudo)``.

        For labeled samples (index < num_labeled):
            - label_vector: original ground-truth labels (never overwritten)
            - loss_mask: all ones
            - is_pseudo: False

        For pseudo-labeled samples (index >= num_labeled):
            - label_vector: pseudo-labels (1.0 for accepted, 0.0 for rejected)
            - loss_mask: 1.0 for accepted classes, 0.0 for rejected
            - is_pseudo: True
        """
        num_classes = len(self.label_set)

        if index < self.num_labeled:
            # Labeled sample — ground-truth labels preserved
            item = self.labeled_dataset[index]
            image = item[0]
            label_vector = item[1]
            loss_mask = torch.ones(num_classes, dtype=torch.float32)
            return image, label_vector, loss_mask, False

        # Pseudo-labeled sample
        pseudo_idx = index - self.num_labeled
        label_vector = torch.from_numpy(
            self.pseudo_labels[pseudo_idx]
        ).float()
        loss_mask = torch.from_numpy(
            self.pseudo_masks[pseudo_idx]
        ).float()

        # Load image for pseudo-labeled sample
        if self.pseudo_image_paths is not None:
            import os

            img_path = os.path.join(
                self.image_root, self.pseudo_image_paths[pseudo_idx]
            )
            image = Image.open(img_path).convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        else:
            # Fallback: return a zero tensor if no image path available
            image = torch.zeros(3, 224, 224)

        return image, label_vector, loss_mask, True
