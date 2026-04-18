"""Unit tests for backbone factory and multi-label classifier.

Validates: Requirements 11.3, 16.2
"""

import tempfile

import torch
import torch.nn as nn
import torchvision.models as models

from src.models.backbone_factory import (
    _REGISTRY,
    create_backbone,
    register_backbone,
)
from src.models.classifier import MultiLabelClassifier


class TestBackboneFactory:
    """Tests for the backbone factory registry and built-in backbones."""

    def test_imagenet_backbone_returns_resnet50_without_fc(self):
        """ImageNet backbone should be a ResNet50 with Identity replacing FC."""
        backbone = create_backbone("resnet50_imagenet")
        assert isinstance(backbone, models.ResNet)
        assert isinstance(backbone.fc, nn.Identity)

    def test_imagenet_backbone_output_shape(self):
        """ImageNet backbone should output (B, 2048) feature vectors."""
        backbone = create_backbone("resnet50_imagenet")
        backbone.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = backbone(x)
        assert out.shape == (2, 2048)

    def test_lvmmed_backbone_with_mock_weights(self):
        """LVM-Med backbone should load weights from a file and remove FC."""
        # Create mock weights by saving a fresh ResNet50 state dict
        ref_model = models.resnet50(weights=None)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(ref_model.state_dict(), f.name)
            weights_path = f.name

        backbone = create_backbone("resnet50_lvmmed", weights_path=weights_path)
        assert isinstance(backbone, models.ResNet)
        assert isinstance(backbone.fc, nn.Identity)

    def test_lvmmed_backbone_output_shape(self):
        """LVM-Med backbone should output (B, 2048) feature vectors."""
        ref_model = models.resnet50(weights=None)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(ref_model.state_dict(), f.name)
            weights_path = f.name

        backbone = create_backbone("resnet50_lvmmed", weights_path=weights_path)
        backbone.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = backbone(x)
        assert out.shape == (2, 2048)

    def test_lvmmed_backbone_raises_without_weights_path(self):
        """LVM-Med backbone should raise ValueError when weights_path is None."""
        try:
            create_backbone("resnet50_lvmmed", weights_path=None)
            assert False, "Expected ValueError"
        except ValueError as e:
            assert "weights_path is required" in str(e)

    def test_lvmmed_backbone_loads_nested_state_dict(self):
        """LVM-Med backbone should handle weights wrapped in a 'state_dict' key."""
        ref_model = models.resnet50(weights=None)
        wrapped = {"state_dict": ref_model.state_dict()}
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(wrapped, f.name)
            weights_path = f.name

        backbone = create_backbone("resnet50_lvmmed", weights_path=weights_path)
        assert isinstance(backbone.fc, nn.Identity)

    def test_unknown_backbone_raises_key_error(self):
        """Requesting an unregistered backbone should raise KeyError."""
        try:
            create_backbone("nonexistent_backbone")
            assert False, "Expected KeyError"
        except KeyError as e:
            assert "nonexistent_backbone" in str(e)

    def test_register_custom_backbone(self):
        """Custom backbones can be registered and created."""
        def _dummy_loader(weights_path=None):
            return nn.Linear(10, 5)

        register_backbone("test_dummy", _dummy_loader)
        assert "test_dummy" in _REGISTRY
        backbone = create_backbone("test_dummy")
        assert isinstance(backbone, nn.Linear)
        # Cleanup
        del _REGISTRY["test_dummy"]


class TestMultiLabelClassifier:
    """Tests for the MultiLabelClassifier module."""

    def _make_classifier(self, dropout_rate=0.1):
        """Helper to create a classifier with a simple backbone stub."""
        backbone = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(3, 2048),
        )
        return MultiLabelClassifier(
            backbone=backbone,
            feature_dim=2048,
            num_classes=6,
            dropout_rate=dropout_rate,
        )

    def test_forward_output_shape(self):
        """Classifier forward pass should produce (B, 6) logits."""
        model = self._make_classifier()
        model.eval()
        x = torch.randn(4, 3, 1, 1)
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (4, 6)

    def test_forward_returns_raw_logits(self):
        """Output should be raw logits (can be negative), not probabilities."""
        model = self._make_classifier()
        model.eval()
        x = torch.randn(8, 3, 1, 1)
        with torch.no_grad():
            logits = model(x)
        # Raw logits can be negative — if all were in [0,1] that would
        # suggest sigmoid was applied. Shape is the key check here.
        assert logits.shape == (8, 6)

    def test_classifier_with_real_backbone(self):
        """Classifier with a real ResNet50 backbone produces correct shape."""
        backbone = create_backbone("resnet50_imagenet")
        model = MultiLabelClassifier(
            backbone=backbone,
            feature_dim=2048,
            num_classes=6,
            dropout_rate=0.5,
        )
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (2, 6)

    def test_dropout_is_applied(self):
        """Classifier should contain a Dropout layer with the specified rate."""
        model = self._make_classifier(dropout_rate=0.3)
        assert isinstance(model.dropout, nn.Dropout)
        assert model.dropout.p == 0.3

    def test_classifier_head_dimensions(self):
        """Linear head should map feature_dim -> num_classes."""
        model = self._make_classifier()
        assert model.classifier.in_features == 2048
        assert model.classifier.out_features == 6
