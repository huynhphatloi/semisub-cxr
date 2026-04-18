"""Results visualization and reporting utilities.

Generates publication-quality bar charts, per-class AUROC plots, and
Markdown/LaTeX summary tables from evaluation results.

Validates: Requirements 7.5, 8.1, 8.2, 8.3, 8.4
"""

import logging
import os
from typing import List, Optional

import matplotlib  # noqa: E402
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Consistent colour palette for the 4 experimental settings
_SETTING_COLORS = {
    "supervised": "#4C72B0",
    "lvmmed": "#55A868",
    "pseudo_label": "#C44E52",
    "uncertainty_filter": "#8172B2",
}

_SETTING_LABELS = {
    "supervised": "Supervised Baseline",
    "lvmmed": "LVM-Med",
    "pseudo_label": "Pseudo-Label",
    "uncertainty_filter": "Uncertainty Filter",
}


def generate_comparison_bar_chart(
    results_df: pd.DataFrame,
    output_dir: str,
    figsize: tuple = (10, 6),
) -> str:
    """Bar chart of macro-AUROC across 4 settings per labeled ratio.

    Parameters
    ----------
    results_df : pd.DataFrame
        Must contain columns: ``setting``, ``labeled_ratio``, ``macro_auroc``.
        If multiple seeds exist per (setting, ratio), the mean is plotted.
    output_dir : str
        Directory where the PNG is saved.
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    str
        Path to the saved PNG file.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Aggregate over seeds
    agg = (
        results_df.groupby(["labeled_ratio", "setting"])["macro_auroc"]
        .mean()
        .reset_index()
    )

    ratios = sorted(agg["labeled_ratio"].unique())
    settings = sorted(agg["setting"].unique())
    n_settings = len(settings)

    x = np.arange(len(ratios))
    width = 0.8 / max(n_settings, 1)

    fig, ax = plt.subplots(figsize=figsize)
    for idx, setting in enumerate(settings):
        subset = agg[agg["setting"] == setting]
        vals = [
            float(subset[subset["labeled_ratio"] == r]["macro_auroc"].values[0])
            if r in subset["labeled_ratio"].values
            else 0.0
            for r in ratios
        ]
        color = _SETTING_COLORS.get(setting, None)
        label = _SETTING_LABELS.get(setting, setting)
        ax.bar(x + idx * width, vals, width, label=label, color=color)

    ax.set_xlabel("Labeled Ratio")
    ax.set_ylabel("Macro AUROC")
    ax.set_title("Macro AUROC Comparison Across Settings")
    ax.set_xticks(x + width * (n_settings - 1) / 2)
    ax.set_xticklabels([f"{r:.0%}" for r in ratios])
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()

    path = os.path.join(output_dir, "macro_auroc_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved comparison bar chart to %s", path)
    return path


def generate_per_class_auroc_plot(
    results_df: pd.DataFrame,
    output_dir: str,
    label_names: Optional[List[str]] = None,
    figsize: tuple = (12, 6),
) -> str:
    """Per-class AUROC comparison across settings.

    Parameters
    ----------
    results_df : pd.DataFrame
        Must contain ``setting`` and ``auroc_{class}`` columns.
        If multiple seeds/ratios exist, the mean is plotted.
    output_dir : str
        Directory where the PNG is saved.
    label_names : list[str] or None
        Class names. If ``None``, auto-detected from ``auroc_*`` columns.
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    str
        Path to the saved PNG file.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Auto-detect class names from columns
    if label_names is None:
        label_names = [
            col.replace("auroc_", "")
            for col in results_df.columns
            if col.startswith("auroc_")
        ]

    settings = sorted(results_df["setting"].unique())
    n_settings = len(settings)

    x = np.arange(len(label_names))
    width = 0.8 / max(n_settings, 1)

    fig, ax = plt.subplots(figsize=figsize)
    for idx, setting in enumerate(settings):
        subset = results_df[results_df["setting"] == setting]
        vals = []
        for name in label_names:
            col = f"auroc_{name}"
            if col in subset.columns:
                vals.append(float(subset[col].mean()))
            else:
                vals.append(0.0)
        color = _SETTING_COLORS.get(setting, None)
        label = _SETTING_LABELS.get(setting, setting)
        ax.bar(x + idx * width, vals, width, label=label, color=color)

    ax.set_xlabel("Pathology Class")
    ax.set_ylabel("AUROC")
    ax.set_title("Per-Class AUROC Comparison")
    ax.set_xticks(x + width * (n_settings - 1) / 2)
    ax.set_xticklabels(label_names, rotation=30, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()

    path = os.path.join(output_dir, "per_class_auroc_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved per-class AUROC plot to %s", path)
    return path


def generate_results_table(
    results_df: pd.DataFrame,
    output_dir: str,
    fmt: str = "markdown",
    label_names: Optional[List[str]] = None,
) -> str:
    """Generate a summary table of results in Markdown or LaTeX format.

    Parameters
    ----------
    results_df : pd.DataFrame
        Must contain ``setting``, ``labeled_ratio``, ``macro_auroc``, and
        ``auroc_{class}`` columns.
    output_dir : str
        Directory where the table file is saved.
    fmt : str
        ``"markdown"`` or ``"latex"``.
    label_names : list[str] or None
        Class names. If ``None``, auto-detected from ``auroc_*`` columns.

    Returns
    -------
    str
        Path to the saved table file.
    """
    os.makedirs(output_dir, exist_ok=True)

    if label_names is None:
        label_names = [
            col.replace("auroc_", "")
            for col in results_df.columns
            if col.startswith("auroc_")
        ]

    # Aggregate over seeds
    group_cols = ["setting", "labeled_ratio"]
    metric_cols = ["macro_auroc"] + [
        f"auroc_{n}"
        for n in label_names
        if f"auroc_{n}" in results_df.columns
    ]
    # Add optional metric columns if present
    for opt in ("macro_f1", "map"):
        if opt in results_df.columns:
            metric_cols.append(opt)

    agg = results_df.groupby(group_cols)[metric_cols].mean().reset_index()
    agg = agg.sort_values(["labeled_ratio", "setting"]).reset_index(drop=True)

    # Rename setting for display
    agg["setting"] = agg["setting"].map(lambda s: _SETTING_LABELS.get(s, s))

    if fmt == "markdown":
        table_str = _to_markdown(agg)
        ext = "md"
    elif fmt == "latex":
        table_str = _to_latex(agg)
        ext = "tex"
    else:
        raise ValueError(f"Unsupported format: {fmt}. Use 'markdown' or 'latex'.")

    path = os.path.join(output_dir, f"results_table.{ext}")
    with open(path, "w") as f:
        f.write(table_str)

    logger.info("Saved results table (%s) to %s", fmt, path)
    return path


def _to_markdown(df: pd.DataFrame) -> str:
    """Convert a DataFrame to a Markdown table string."""
    cols = df.columns.tolist()
    header = "| " + " | ".join(cols) + " |"
    separator = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, row in df.iterrows():
        cells = []
        for col in cols:
            val = row[col]
            if isinstance(val, float):
                cells.append(f"{val:.4f}")
            else:
                cells.append(str(val))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, separator] + rows) + "\n"


def _to_latex(df: pd.DataFrame) -> str:
    """Convert a DataFrame to a LaTeX tabular string."""
    cols = df.columns.tolist()
    col_fmt = "l" * len(cols)
    lines = [
        f"\\begin{{tabular}}{{{col_fmt}}}",
        "\\toprule",
        " & ".join(cols) + " \\\\",
        "\\midrule",
    ]
    for _, row in df.iterrows():
        cells = []
        for col in cols:
            val = row[col]
            if isinstance(val, float):
                cells.append(f"{val:.4f}")
            else:
                cells.append(str(val))
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines) + "\n"
