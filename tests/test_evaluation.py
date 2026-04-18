"""Unit tests for the evaluation module (metrics, evaluator, reporting)."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import compute_auroc, compute_optional_metrics
from src.evaluation.reporting import (
    generate_comparison_bar_chart,
    generate_per_class_auroc_plot,
    generate_results_table,
)


# ---------------------------------------------------------------------------
# metrics.py tests
# ---------------------------------------------------------------------------


class TestComputeAuroc:
    def test_perfect_predictions(self):
        """Perfect scores should yield AUROC = 1.0 for each class."""
        y_true = np.array([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=np.float64)
        y_score = np.array(
            [[0.9, 0.1], [0.1, 0.9], [0.8, 0.8], [0.2, 0.2]], dtype=np.float64
        )
        result = compute_auroc(y_true, y_score, ["A", "B"])
        assert result["macro_auroc"] == pytest.approx(1.0)
        assert result["per_class_auroc"]["A"] == pytest.approx(1.0)
        assert result["per_class_auroc"]["B"] == pytest.approx(1.0)

    def test_single_class_value_returns_nan(self):
        """If a class has only positives, its AUROC should be NaN."""
        y_true = np.array([[1, 0], [1, 1]], dtype=np.float64)
        y_score = np.array([[0.9, 0.1], [0.8, 0.9]], dtype=np.float64)
        result = compute_auroc(y_true, y_score, ["A", "B"])
        assert np.isnan(result["per_class_auroc"]["A"])
        # B has both 0 and 1
        assert not np.isnan(result["per_class_auroc"]["B"])

    def test_macro_excludes_nan_classes(self):
        """Macro AUROC should only average over classes with valid AUROC."""
        y_true = np.array([[1, 0], [1, 1], [1, 0]], dtype=np.float64)
        y_score = np.array(
            [[0.9, 0.1], [0.8, 0.9], [0.7, 0.2]], dtype=np.float64
        )
        result = compute_auroc(y_true, y_score, ["A", "B"])
        # A is all-positive → NaN, macro should equal B's AUROC
        assert result["macro_auroc"] == pytest.approx(
            result["per_class_auroc"]["B"]
        )

    def test_returns_correct_keys(self):
        y_true = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.float64)
        y_score = np.array([[0.8, 0.2, 0.7], [0.3, 0.9, 0.4]], dtype=np.float64)
        names = ["X", "Y", "Z"]
        result = compute_auroc(y_true, y_score, names)
        assert "macro_auroc" in result
        assert "per_class_auroc" in result
        assert set(result["per_class_auroc"].keys()) == set(names)


class TestComputeOptionalMetrics:
    def test_macro_f1(self):
        y_true = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float64)
        y_pred = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float64)
        y_score = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.8]], dtype=np.float64)
        result = compute_optional_metrics(
            y_true, y_pred, y_score, ["A", "B"], ["macro_f1"]
        )
        assert "macro_f1" in result
        assert result["macro_f1"] == pytest.approx(1.0)

    def test_map(self):
        y_true = np.array([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=np.float64)
        y_score = np.array(
            [[0.9, 0.1], [0.1, 0.9], [0.8, 0.8], [0.2, 0.2]], dtype=np.float64
        )
        y_pred = (y_score >= 0.5).astype(np.float64)
        result = compute_optional_metrics(
            y_true, y_pred, y_score, ["A", "B"], ["map"]
        )
        assert "map" in result
        assert 0.0 <= result["map"] <= 1.0

    def test_empty_metrics_list(self):
        y_true = np.array([[1, 0]], dtype=np.float64)
        y_pred = np.array([[1, 0]], dtype=np.float64)
        y_score = np.array([[0.9, 0.1]], dtype=np.float64)
        result = compute_optional_metrics(
            y_true, y_pred, y_score, ["A", "B"], []
        )
        assert result == {}

    def test_both_metrics(self):
        y_true = np.array([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=np.float64)
        y_score = np.array(
            [[0.9, 0.1], [0.1, 0.9], [0.8, 0.8], [0.2, 0.2]], dtype=np.float64
        )
        y_pred = (y_score >= 0.5).astype(np.float64)
        result = compute_optional_metrics(
            y_true, y_pred, y_score, ["A", "B"], ["macro_f1", "map"]
        )
        assert "macro_f1" in result
        assert "map" in result


# ---------------------------------------------------------------------------
# reporting.py tests
# ---------------------------------------------------------------------------


def _make_results_df():
    """Create a small results DataFrame for testing."""
    rows = []
    for setting in ["supervised", "lvmmed"]:
        for ratio in [0.05, 0.10]:
            rows.append(
                {
                    "setting": setting,
                    "labeled_ratio": ratio,
                    "seed": 42,
                    "macro_auroc": np.random.uniform(0.6, 0.9),
                    "auroc_Cardiomegaly": np.random.uniform(0.5, 0.95),
                    "auroc_Edema": np.random.uniform(0.5, 0.95),
                }
            )
    return pd.DataFrame(rows)


class TestReporting:
    def test_comparison_bar_chart_creates_png(self):
        df = _make_results_df()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = generate_comparison_bar_chart(df, tmpdir)
            assert os.path.isfile(path)
            assert path.endswith(".png")

    def test_per_class_auroc_plot_creates_png(self):
        df = _make_results_df()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = generate_per_class_auroc_plot(df, tmpdir)
            assert os.path.isfile(path)
            assert path.endswith(".png")

    def test_results_table_markdown(self):
        df = _make_results_df()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = generate_results_table(df, tmpdir, fmt="markdown")
            assert os.path.isfile(path)
            assert path.endswith(".md")
            content = open(path).read()
            assert "|" in content
            assert "---" in content

    def test_results_table_latex(self):
        df = _make_results_df()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = generate_results_table(df, tmpdir, fmt="latex")
            assert os.path.isfile(path)
            assert path.endswith(".tex")
            content = open(path).read()
            assert "\\begin{tabular}" in content
            assert "\\end{tabular}" in content

    def test_results_table_invalid_format_raises(self):
        df = _make_results_df()
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Unsupported format"):
                generate_results_table(df, tmpdir, fmt="html")
