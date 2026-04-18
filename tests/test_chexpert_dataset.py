"""Unit tests for src/data/chexpert_dataset.py.

Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1A.1, 1A.2, 1A.5
"""

import os

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.data.chexpert_dataset import CheXpertDataset

_LABEL_SET = [
    "Cardiomegaly",
    "Pleural Effusion",
    "Pneumothorax",
    "Consolidation",
    "Atelectasis",
    "Edema",
]


@pytest.fixture()
def tmp_dataset(tmp_path):
    """Create a minimal CheXpert-like CSV and dummy images on disk."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    rows = []
    for i in range(10):
        rel_path = f"img_{i}.jpg"
        img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
        img.save(img_dir / rel_path)
        view = "Frontal" if i < 7 else "Lateral"
        label_vals = {col: float(np.random.choice([-1, 0, 1])) for col in _LABEL_SET}
        rows.append({"Path": rel_path, "Frontal/Lateral": view, "Extra_Col": 0.0, **label_vals})

    df = pd.DataFrame(rows)
    csv_path = tmp_path / "train.csv"
    df.to_csv(csv_path, index=False)

    return {
        "csv_path": str(csv_path),
        "image_root": str(img_dir),
        "df": df,
    }


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_file_not_found_raises():
    """FileNotFoundError when CSV path does not exist."""
    with pytest.raises(FileNotFoundError):
        CheXpertDataset(
            csv_path="/nonexistent/train.csv",
            image_root="/nonexistent",
            label_set=_LABEL_SET,
            uncertain_policy="zeros",
            view_policy="all",
        )


def test_unknown_uncertain_policy_raises(tmp_dataset):
    """ValueError for an unsupported uncertain_policy."""
    with pytest.raises(ValueError, match="uncertain_policy"):
        CheXpertDataset(
            csv_path=tmp_dataset["csv_path"],
            image_root=tmp_dataset["image_root"],
            label_set=_LABEL_SET,
            uncertain_policy="bad_policy",
            view_policy="all",
        )


def test_unknown_view_policy_raises(tmp_dataset):
    """ValueError for an unsupported view_policy."""
    with pytest.raises(ValueError, match="view_policy"):
        CheXpertDataset(
            csv_path=tmp_dataset["csv_path"],
            image_root=tmp_dataset["image_root"],
            label_set=_LABEL_SET,
            uncertain_policy="zeros",
            view_policy="bad_view",
        )


def test_missing_label_columns_raises(tmp_dataset):
    """ValueError when requested label columns are absent from CSV."""
    with pytest.raises(ValueError, match="missing"):
        CheXpertDataset(
            csv_path=tmp_dataset["csv_path"],
            image_root=tmp_dataset["image_root"],
            label_set=_LABEL_SET + ["NonExistentLabel"],
            uncertain_policy="zeros",
            view_policy="all",
        )


# ---------------------------------------------------------------------------
# Label filtering and uncertain mapping
# ---------------------------------------------------------------------------


def test_label_vector_shape_and_dtype(tmp_dataset):
    """label_vector is FloatTensor of shape (6,)."""
    ds = CheXpertDataset(
        csv_path=tmp_dataset["csv_path"],
        image_root=tmp_dataset["image_root"],
        label_set=_LABEL_SET,
        uncertain_policy="zeros",
        view_policy="all",
    )
    _, label_vec, _ = ds[0]
    assert label_vec.shape == (6,)
    assert label_vec.dtype == torch.float32


def test_uncertain_zeros_maps_minus_one_to_zero(tmp_dataset):
    """With uncertain_policy='zeros', all labels are in {0, 1}."""
    ds = CheXpertDataset(
        csv_path=tmp_dataset["csv_path"],
        image_root=tmp_dataset["image_root"],
        label_set=_LABEL_SET,
        uncertain_policy="zeros",
        view_policy="all",
    )
    for i in range(len(ds)):
        _, label_vec, _ = ds[i]
        assert set(label_vec.tolist()).issubset({0.0, 1.0})


def test_uncertain_ones_maps_minus_one_to_one(tmp_dataset):
    """With uncertain_policy='ones', all labels are in {0, 1}."""
    ds = CheXpertDataset(
        csv_path=tmp_dataset["csv_path"],
        image_root=tmp_dataset["image_root"],
        label_set=_LABEL_SET,
        uncertain_policy="ones",
        view_policy="all",
    )
    for i in range(len(ds)):
        _, label_vec, _ = ds[i]
        assert set(label_vec.tolist()).issubset({0.0, 1.0})


# ---------------------------------------------------------------------------
# View filtering
# ---------------------------------------------------------------------------


def test_frontal_only_filters_lateral(tmp_dataset):
    """frontal_only retains only frontal views (7 out of 10 in fixture)."""
    ds = CheXpertDataset(
        csv_path=tmp_dataset["csv_path"],
        image_root=tmp_dataset["image_root"],
        label_set=_LABEL_SET,
        uncertain_policy="zeros",
        view_policy="frontal_only",
    )
    assert len(ds) == 7


def test_all_view_policy_keeps_everything(tmp_dataset):
    """view_policy='all' retains all 10 samples."""
    ds = CheXpertDataset(
        csv_path=tmp_dataset["csv_path"],
        image_root=tmp_dataset["image_root"],
        label_set=_LABEL_SET,
        uncertain_policy="zeros",
        view_policy="all",
    )
    assert len(ds) == 10


# ---------------------------------------------------------------------------
# __getitem__ returns correct tuple
# ---------------------------------------------------------------------------


def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """Minimal PIL → Tensor transform (no torchvision dependency)."""
    arr = np.array(img.resize((32, 32))).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)  # HWC → CHW


def test_getitem_returns_correct_tuple(tmp_dataset):
    """__getitem__ returns (image_tensor, label_vector, index)."""
    ds = CheXpertDataset(
        csv_path=tmp_dataset["csv_path"],
        image_root=tmp_dataset["image_root"],
        label_set=_LABEL_SET,
        uncertain_policy="zeros",
        view_policy="all",
        transform=_pil_to_tensor,
    )
    img, label_vec, idx = ds[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, 32, 32)
    assert label_vec.shape == (6,)
    assert idx == 0


# ---------------------------------------------------------------------------
# Missing image files are skipped with warning
# ---------------------------------------------------------------------------


def test_missing_images_skipped(tmp_dataset, caplog):
    """Samples with missing image files are filtered out with a warning."""
    # Remove one image file
    first_path = tmp_dataset["df"]["Path"].iloc[0]
    os.remove(os.path.join(tmp_dataset["image_root"], first_path))

    import logging

    with caplog.at_level(logging.WARNING):
        ds = CheXpertDataset(
            csv_path=tmp_dataset["csv_path"],
            image_root=tmp_dataset["image_root"],
            label_set=_LABEL_SET,
            uncertain_policy="zeros",
            view_policy="all",
        )

    assert len(ds) == 9
    assert "not found" in caplog.text.lower()
