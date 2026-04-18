"""Backbone factory with registry pattern for extensible backbone loading.

Provides a registry-based factory for creating backbone feature extractors.
New backbones can be registered without modifying existing training code.

Validates: Requirements 3.1, 4.1, 11.3, 16.2
"""

import logging
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights

logger = logging.getLogger(__name__)

_REGISTRY: Dict[str, Callable] = {}


def register_backbone(name: str, loader: Callable) -> None:
    """Register a backbone loader function under the given name.

    Args:
        name: Unique identifier for the backbone (e.g. "resnet50_imagenet").
        loader: Callable that accepts an optional weights_path and returns an nn.Module.
    """
    _REGISTRY[name] = loader


def create_backbone(name: str, weights_path: Optional[str] = None) -> nn.Module:
    """Create a backbone by name from the registry.

    Args:
        name: Registered backbone identifier.
        weights_path: Optional path to pretrained weights file.

    Returns:
        An nn.Module backbone that outputs feature vectors.

    Raises:
        KeyError: If the backbone name is not registered.
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"Backbone '{name}' not found in registry. "
            f"Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[name](weights_path)


def _load_resnet50_imagenet(weights_path: Optional[str] = None) -> nn.Module:
    """Load ResNet50 with ImageNet pretrained weights, removing the final FC layer."""
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Identity()
    logger.info("Loaded ResNet50 backbone with ImageNet pretrained weights.")
    return model


def _load_resnet50_lvmmed(weights_path: Optional[str] = None) -> nn.Module:
    """Load ResNet50 architecture with LVM-Med weights from disk.

    Args:
        weights_path: Path to the LVM-Med pretrained weights file.

    Returns:
        ResNet50 backbone with LVM-Med weights loaded and final FC removed.

    Raises:
        ValueError: If weights_path is None.
    """
    if weights_path is None:
        raise ValueError(
            "weights_path is required for 'resnet50_lvmmed' backbone. "
            "Provide the path to LVM-Med pretrained weights."
        )

    model = models.resnet50(weights=None)
    state_dict = torch.load(weights_path, map_location="cpu")

    # Handle different key formats from LVM-Med checkpoints
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    elif "model" in state_dict:
        state_dict = state_dict["model"]

    # Strip common prefixes (e.g. "backbone.", "encoder.", "module.")
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in ("backbone.", "encoder.", "module."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        cleaned[new_key] = value

    model.load_state_dict(cleaned, strict=False)
    model.fc = nn.Identity()
    logger.info("Loaded ResNet50 backbone with LVM-Med weights from %s.", weights_path)
    return model


# Register built-in backbones
register_backbone("resnet50_imagenet", _load_resnet50_imagenet)
register_backbone("resnet50_lvmmed", _load_resnet50_lvmmed)
