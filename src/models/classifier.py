"""Multi-label classifier with pluggable backbone.

Combines a backbone feature extractor with a dropout + linear classification head.
Returns raw logits; sigmoid is applied externally in loss computation or evaluation.

Validates: Requirements 3.2, 16.1
"""

import torch
import torch.nn as nn


class MultiLabelClassifier(nn.Module):
    """Multi-label classifier with backbone, dropout, and linear head.

    Args:
        backbone: Feature extractor module outputting (B, feature_dim) tensors.
        feature_dim: Dimensionality of backbone output features.
        num_classes: Number of output classes (labels).
        dropout_rate: Dropout probability applied before the classification head.
    """

    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        num_classes: int,
        dropout_rate: float,
    ):
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logits.

        Args:
            x: Input image tensor of shape (B, C, H, W).

        Returns:
            Logits tensor of shape (B, num_classes). Sigmoid is NOT applied.
        """
        features = self.backbone(x)  # (B, feature_dim)
        features = self.dropout(features)
        logits = self.classifier(features)  # (B, num_classes)
        return logits
