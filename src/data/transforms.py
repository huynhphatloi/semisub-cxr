"""Image transforms for training and evaluation pipelines.

Provides factory functions that build torchvision transform pipelines
with backbone-specific normalization parameters.

Validates: Requirements 13.3, 13.4
"""

from typing import List

from torchvision import transforms


def get_train_transform(
    image_size: int,
    normalization_mean: List[float],
    normalization_std: List[float],
) -> transforms.Compose:
    """Build the training augmentation pipeline.

    Resize -> RandomHorizontalFlip -> RandomRotation(±10°) -> ToTensor -> Normalize.

    Parameters
    ----------
    image_size : int
        Target spatial size (height and width) for ``Resize``.
    normalization_mean : list[float]
        Per-channel mean for ``Normalize`` (backbone-specific).
    normalization_std : list[float]
        Per-channel std for ``Normalize`` (backbone-specific).

    Returns
    -------
    torchvision.transforms.Compose
        The composed training transform.
    """
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=normalization_mean, std=normalization_std),
        ]
    )


def get_eval_transform(
    image_size: int,
    normalization_mean: List[float],
    normalization_std: List[float],
) -> transforms.Compose:
    """Build the evaluation (inference) transform pipeline.

    Resize -> ToTensor -> Normalize.

    Parameters
    ----------
    image_size : int
        Target spatial size (height and width) for ``Resize``.
    normalization_mean : list[float]
        Per-channel mean for ``Normalize`` (backbone-specific).
    normalization_std : list[float]
        Per-channel std for ``Normalize`` (backbone-specific).

    Returns
    -------
    torchvision.transforms.Compose
        The composed evaluation transform.
    """
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=normalization_mean, std=normalization_std),
        ]
    )
