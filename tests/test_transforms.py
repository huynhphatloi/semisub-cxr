"""Tests for src/data/transforms.py."""

import torch
from PIL import Image
from torchvision import transforms

from src.data.transforms import get_eval_transform, get_train_transform

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224


def _make_dummy_image(width: int = 300, height: int = 400) -> Image.Image:
    """Create a random RGB PIL image."""
    return Image.fromarray(
        torch.randint(0, 256, (height, width, 3), dtype=torch.uint8).numpy()
    )


class TestGetTrainTransform:
    """Tests for get_train_transform."""

    def test_output_shape(self):
        tfm = get_train_transform(IMAGE_SIZE, MEAN, STD)
        img = _make_dummy_image()
        out = tfm(img)
        assert out.shape == (3, IMAGE_SIZE, IMAGE_SIZE)

    def test_output_is_float_tensor(self):
        tfm = get_train_transform(IMAGE_SIZE, MEAN, STD)
        out = tfm(_make_dummy_image())
        assert isinstance(out, torch.Tensor)
        assert out.dtype == torch.float32

    def test_pipeline_contains_expected_transforms(self):
        tfm = get_train_transform(IMAGE_SIZE, MEAN, STD)
        types = [type(t) for t in tfm.transforms]
        assert types == [
            transforms.Resize,
            transforms.RandomHorizontalFlip,
            transforms.RandomRotation,
            transforms.ToTensor,
            transforms.Normalize,
        ]

    def test_rotation_degrees(self):
        tfm = get_train_transform(IMAGE_SIZE, MEAN, STD)
        rot = [
            t for t in tfm.transforms
            if isinstance(t, transforms.RandomRotation)
        ][0]
        assert rot.degrees == [-10.0, 10.0]

    def test_normalize_params(self):
        tfm = get_train_transform(IMAGE_SIZE, MEAN, STD)
        norm = [t for t in tfm.transforms if isinstance(t, transforms.Normalize)][0]
        assert list(norm.mean) == MEAN
        assert list(norm.std) == STD

    def test_different_image_size(self):
        tfm = get_train_transform(320, MEAN, STD)
        out = tfm(_make_dummy_image())
        assert out.shape == (3, 320, 320)


class TestGetEvalTransform:
    """Tests for get_eval_transform."""

    def test_output_shape(self):
        tfm = get_eval_transform(IMAGE_SIZE, MEAN, STD)
        out = tfm(_make_dummy_image())
        assert out.shape == (3, IMAGE_SIZE, IMAGE_SIZE)

    def test_output_is_float_tensor(self):
        tfm = get_eval_transform(IMAGE_SIZE, MEAN, STD)
        out = tfm(_make_dummy_image())
        assert isinstance(out, torch.Tensor)
        assert out.dtype == torch.float32

    def test_pipeline_contains_expected_transforms(self):
        tfm = get_eval_transform(IMAGE_SIZE, MEAN, STD)
        types = [type(t) for t in tfm.transforms]
        assert types == [
            transforms.Resize,
            transforms.ToTensor,
            transforms.Normalize,
        ]

    def test_no_augmentation_transforms(self):
        """Eval pipeline must not include random augmentations."""
        tfm = get_eval_transform(IMAGE_SIZE, MEAN, STD)
        for t in tfm.transforms:
            assert not isinstance(t, (transforms.RandomHorizontalFlip, transforms.RandomRotation))

    def test_normalize_params(self):
        tfm = get_eval_transform(IMAGE_SIZE, MEAN, STD)
        norm = [t for t in tfm.transforms if isinstance(t, transforms.Normalize)][0]
        assert list(norm.mean) == MEAN
        assert list(norm.std) == STD
