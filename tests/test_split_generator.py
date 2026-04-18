"""Unit tests for src/data/split_generator.py.

Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8
"""

import json

import numpy as np
import pytest

from src.data.split_generator import (
    SplitGenerationError,
    SplitMetadata,
    _load_split,
    _save_split,
    _validate_split,
    generate_split,
)


@pytest.fixture
def labels_array():
    """A small multi-label matrix with at least 1 positive per class."""
    rng = np.random.RandomState(0)
    # 100 samples, 6 classes, ~30% positive rate
    labels = (rng.rand(100, 6) > 0.7).astype(float)
    # Ensure every class has at least a few positives
    for c in range(6):
        labels[c, c] = 1.0
    return labels


class TestGenerateSplit:
    """Core generate_split behaviour."""

    def test_produces_valid_partition(self, labels_array, tmp_path):
        meta = generate_split(labels_array, 0.2, 42, str(tmp_path))
        full = set(range(100))
        assert set(meta.labeled_indices) | set(meta.unlabeled_indices) == full
        assert set(meta.labeled_indices) & set(meta.unlabeled_indices) == set()

    def test_labeled_size_matches_ratio(self, labels_array, tmp_path):
        meta = generate_split(labels_array, 0.2, 42, str(tmp_path))
        assert meta.labeled_size == len(meta.labeled_indices)
        assert meta.unlabeled_size == len(meta.unlabeled_indices)
        assert meta.labeled_size + meta.unlabeled_size == 100

    def test_deterministic_with_same_seed(self, labels_array, tmp_path):
        dir1 = str(tmp_path / "a")
        dir2 = str(tmp_path / "b")
        m1 = generate_split(labels_array, 0.2, 42, dir1)
        m2 = generate_split(labels_array, 0.2, 42, dir2)
        assert m1.labeled_indices == m2.labeled_indices
        assert m1.unlabeled_indices == m2.unlabeled_indices

    def test_different_seeds_produce_different_splits(self, labels_array, tmp_path):
        dir1 = str(tmp_path / "a")
        dir2 = str(tmp_path / "b")
        m1 = generate_split(labels_array, 0.2, 42, dir1)
        m2 = generate_split(labels_array, 0.2, 99, dir2)
        assert m1.labeled_indices != m2.labeled_indices

    def test_persists_and_reloads(self, labels_array, tmp_path):
        meta1 = generate_split(labels_array, 0.2, 42, str(tmp_path))
        meta2 = generate_split(labels_array, 0.2, 42, str(tmp_path))
        assert meta1.labeled_indices == meta2.labeled_indices
        assert meta1.unlabeled_indices == meta2.unlabeled_indices

    def test_json_file_created(self, labels_array, tmp_path):
        generate_split(labels_array, 0.2, 42, str(tmp_path))
        path = tmp_path / "split_r0.2_s42.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert "labeled_indices" in data
        assert "generation_timestamp" in data

    def test_per_class_positive_counts(self, labels_array, tmp_path):
        meta = generate_split(
            labels_array, 0.2, 42, str(tmp_path),
            class_names=[f"c{i}" for i in range(6)],
        )
        for name, count in meta.per_class_positive_counts.items():
            assert count >= 1

    def test_accepts_dataset_with_labels_attr(self, labels_array, tmp_path):
        """Accepts objects with a .labels attribute (like CheXpertDataset)."""

        class FakeDataset:
            def __init__(self, lbl):
                self.labels = lbl

        ds = FakeDataset(labels_array)
        meta = generate_split(ds, 0.2, 42, str(tmp_path))
        assert meta.total_train_size == 100


class TestRetryAndError:
    """Retry logic and SplitGenerationError."""

    def test_raises_on_impossible_split(self, tmp_path):
        """All-zero labels for one class → should exhaust retries."""
        labels = np.zeros((20, 3))
        # Only class 0 has positives
        labels[:, 0] = 1.0
        with pytest.raises(SplitGenerationError, match="Failed to generate"):
            generate_split(labels, 0.1, 0, str(tmp_path), min_positive_per_class=1)


class TestValidation:
    """_validate_split catches inconsistencies."""

    def test_rejects_wrong_total_size(self):
        meta = SplitMetadata(
            seed=0, labeled_ratio=0.5, total_train_size=10,
            labeled_size=5, unlabeled_size=5,
            labeled_indices=[0, 1, 2, 3, 4],
            unlabeled_indices=[5, 6, 7, 8, 9],
            per_class_positive_counts={}, generation_timestamp="",
        )
        with pytest.raises(ValueError, match="does not match dataset size"):
            _validate_split(meta, 20)

    def test_rejects_overlapping_indices(self):
        meta = SplitMetadata(
            seed=0, labeled_ratio=0.5, total_train_size=4,
            labeled_size=3, unlabeled_size=2,
            labeled_indices=[0, 1, 2],
            unlabeled_indices=[2, 3],
            per_class_positive_counts={}, generation_timestamp="",
        )
        with pytest.raises(ValueError, match="overlap"):
            _validate_split(meta, 4)

    def test_rejects_incomplete_coverage(self):
        meta = SplitMetadata(
            seed=0, labeled_ratio=0.5, total_train_size=4,
            labeled_size=2, unlabeled_size=1,
            labeled_indices=[0, 1],
            unlabeled_indices=[3],
            per_class_positive_counts={}, generation_timestamp="",
        )
        with pytest.raises(ValueError, match="do not cover"):
            _validate_split(meta, 4)


class TestSaveLoadRoundTrip:
    """JSON persistence round-trip."""

    def test_round_trip(self, tmp_path):
        meta = SplitMetadata(
            seed=42, labeled_ratio=0.1, total_train_size=100,
            labeled_size=10, unlabeled_size=90,
            labeled_indices=list(range(10)),
            unlabeled_indices=list(range(10, 100)),
            per_class_positive_counts={"a": 3, "b": 7},
            generation_timestamp="2025-01-01T00:00:00+00:00",
        )
        path = str(tmp_path / "test.json")
        _save_split(meta, path)
        loaded = _load_split(path)
        assert loaded == meta
