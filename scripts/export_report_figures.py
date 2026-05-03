#!/usr/bin/env python3
"""Export publication-quality figures for LaTeX report.

Generates:
  fig2 - Macro AUROC grouped bar chart (3 methods x 3 ratios)
  fig3 - Delta Macro AUROC vs supervised baseline
  fig4 - Per-class AUROC at 5% labeled ratio
  fig5 - Per-class AUROC at 10% labeled ratio
  fig6 - Per-class AUROC at 20% labeled ratio
  fig7 - Pseudo-label acceptance rate at threshold=0.7

Run from repo root:
  python scripts/export_report_figures.py
"""

import os
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Constants ──────────────────────────────────────────────────────

PLOTS_DIR = os.path.join("outputs", "plots")
RESULTS_DIR = os.path.join("outputs", "results")
PSEUDO_DIR = os.path.join("outputs", "pseudo_labels")

SETTING_ORDER = ["supervised", "pseudo_label", "uncertainty_filter"]
SETTING_LABELS = {
    "supervised": "Supervised baseline",
    "pseudo_label": "Pseudo-labeling",
    "uncertainty_filter": "Pseudo-labeling + MC Dropout uncertainty",
}
COLORS = {
    "supervised": "#4C72B0",
    "pseudo_label": "#C44E52",
    "uncertainty_filter": "#8172B2",
}

RATIOS = [0.05, 0.10, 0.20]
RATIO_LABELS = ["5%", "10%", "20%"]

CLASS_NAMES_FULL = [
    "Cardiomegaly", "Pleural Effusion", "Pneumothorax",
    "Consolidation", "Atelectasis", "Edema",
]
CLASS_NAMES_SHORT = [
    "Cardio.", "Effusion", "Pneumo.",
    "Consol.", "Atelec.", "Edema",
]

# Fallback macro AUROC (from checkpoint val_metrics)
FALLBACK_MACRO = {
    ("supervised", 0.05): 0.8429, ("supervised", 0.10): 0.8325,
    ("supervised", 0.20): 0.8497,
    ("pseudo_label", 0.05): 0.8414, ("pseudo_label", 0.10): 0.8514,
    ("pseudo_label", 0.20): 0.8508,
    ("uncertainty_filter", 0.05): 0.8377, ("uncertainty_filter", 0.10): 0.8458,
    ("uncertainty_filter", 0.20): 0.8519,
}

# Fallback per-class AUROC (from checkpoint val_metrics)
FALLBACK_PERCLASS = {
    ("supervised", 0.05): [0.8338, 0.9181, 0.7788, 0.9162, 0.7122, 0.8982],
    ("pseudo_label", 0.05): [0.8073, 0.8892, 0.7516, 0.8862, 0.8279, 0.8862],
    ("uncertainty_filter", 0.05): [0.7818, 0.8941, 0.7810, 0.8528, 0.8293, 0.8874],
    ("supervised", 0.10): [0.8479, 0.8972, 0.7150, 0.9103, 0.7289, 0.8958],
    ("pseudo_label", 0.10): [0.7917, 0.9046, 0.8158, 0.8954, 0.8072, 0.8938],
    ("uncertainty_filter", 0.10): [0.8044, 0.8929, 0.7916, 0.9005, 0.7646, 0.9212],
    ("supervised", 0.20): [0.7625, 0.9194, 0.8520, 0.8323, 0.8182, 0.9135],
    ("pseudo_label", 0.20): [0.8197, 0.9231, 0.7751, 0.9120, 0.7610, 0.9138],
    ("uncertainty_filter", 0.20): [0.8127, 0.9301, 0.8648, 0.8715, 0.7181, 0.9144],
}

# Fallback acceptance rates (%)
FALLBACK_ACCEPTANCE = [7.26, 23.13, 2.14, 0.01, 0.03, 7.84]



# ── Data loading ───────────────────────────────────────────────────

def _normalize_ratio(r):
    """Round ratio to nearest standard value."""
    for std in RATIOS:
        if abs(float(r) - std) < 0.005:
            return std
    return float(r)


def load_macro_data():
    """Load macro AUROC from CSV, JSON, or fallback."""
    data = {}

    # Try CSV
    csv_path = os.path.join(RESULTS_DIR, "evaluation_results.csv")
    if os.path.isfile(csv_path):
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                s = str(row.get("setting", "")).strip()
                for key in SETTING_ORDER:
                    if key in s.lower().replace(" ", "_").replace("-", "_"):
                        s = key
                        break
                r = _normalize_ratio(row.get("labeled_ratio", 0))
                a = float(row.get("macro_auroc", 0))
                if s in SETTING_ORDER and r in RATIOS:
                    data[(s, r)] = a
            if len(data) >= 9:
                print(f"Macro AUROC: loaded {len(data)} entries from CSV")
                return data
        except Exception as e:
            print(f"CSV load failed: {e}")

    # Try JSON
    for setting in SETTING_ORDER:
        for ratio in RATIOS:
            for rstr in [f"{ratio}", f"{ratio:.2f}"]:
                jf = os.path.join(RESULTS_DIR, f"eval_{setting}_{rstr}_42.json")
                if os.path.isfile(jf):
                    with open(jf) as f:
                        obj = json.load(f)
                    data[(setting, ratio)] = float(obj.get("macro_auroc", 0))
    if len(data) >= 9:
        print(f"Macro AUROC: loaded {len(data)} entries from JSON")
        return data

    print("Macro AUROC: using fallback values")
    return dict(FALLBACK_MACRO)


def load_perclass_data():
    """Load per-class AUROC from JSON files or fallback."""
    data = {}

    for setting in SETTING_ORDER:
        for ratio in RATIOS:
            # Try multiple filename patterns
            for rstr in [f"{ratio}", f"{ratio:.2f}", f"{ratio:g}"]:
                jf = os.path.join(RESULTS_DIR, f"eval_{setting}_{rstr}_42.json")
                if os.path.isfile(jf):
                    with open(jf) as f:
                        obj = json.load(f)
                    # Extract per-class values
                    pc = obj.get("per_class_auroc", {})
                    if pc and isinstance(pc, dict):
                        vals = [float(pc.get(cn, 0)) for cn in CLASS_NAMES_FULL]
                        data[(setting, ratio)] = vals
                        break
                    # Try flat auroc_* keys
                    vals = []
                    for cn in CLASS_NAMES_FULL:
                        v = obj.get(f"auroc_{cn}", 0)
                        vals.append(float(v))
                    if any(v > 0 for v in vals):
                        data[(setting, ratio)] = vals
                        break

    if len(data) >= 9:
        print(f"Per-class AUROC: loaded {len(data)} entries from JSON")
        return data

    print(f"Per-class AUROC: loaded {len(data)} from JSON, using fallback for rest")
    for key, val in FALLBACK_PERCLASS.items():
        if key not in data:
            data[key] = val
    return data


def load_acceptance_rates():
    """Load pseudo-label acceptance rates from CSV or fallback."""
    csv_path = os.path.join(PSEUDO_DIR, "pseudo_label", "0.05_42", "pseudo_labels.csv")
    if os.path.isfile(csv_path):
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            n = len(df)
            rates = []
            for cn in CLASS_NAMES_FULL:
                col = f"rejected_{cn}"
                if col in df.columns:
                    rejected = df[col].sum()
                    accepted = n - rejected
                    rates.append(100.0 * accepted / n)
                else:
                    rates.append(0.0)
            print(f"Acceptance rates: computed from {csv_path} ({n} samples)")
            return rates
        except Exception as e:
            print(f"Acceptance CSV load failed: {e}")

    print("Acceptance rates: using fallback values")
    return list(FALLBACK_ACCEPTANCE)



# ── Figure functions ───────────────────────────────────────────────

def _save(fig, base):
    """Save PNG (300 dpi) and PDF, print paths."""
    fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  {base}.png")
    print(f"  {base}.pdf")


def fig2_macro_auroc(data):
    """Grouped bar chart: Macro AUROC by method and labeled ratio."""
    x = np.arange(len(RATIOS))
    n = len(SETTING_ORDER)
    width = 0.22
    offsets = np.linspace(-(n - 1) / 2 * width, (n - 1) / 2 * width, n)

    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, setting in enumerate(SETTING_ORDER):
        vals = [data.get((setting, r), 0) for r in RATIOS]
        bars = ax.bar(
            x + offsets[idx], vals, width,
            label=SETTING_LABELS[setting],
            color=COLORS[setting], edgecolor="black", linewidth=0.6,
        )
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.0008,
                    f"{v:.4f}", ha="center", va="bottom",
                    fontsize=7.5, fontweight="bold")

    ax.set_xlabel("Labeled ratio", fontsize=11)
    ax.set_ylabel("Macro AUROC", fontsize=11)
    ax.set_title("So s\u00e1nh Macro AUROC theo method v\u00e0 labeled ratio",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(RATIO_LABELS, fontsize=10)
    ax.set_ylim(0.830, 0.860)
    ax.legend(fontsize=8.5, loc="upper left")
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.tick_params(axis="both", which="major", labelsize=10)
    fig.tight_layout()
    _save(fig, os.path.join(PLOTS_DIR, "fig2_macro_auroc_by_method_ratio"))


def fig3_delta_auroc(data):
    """Grouped bar chart: Delta Macro AUROC vs supervised baseline."""
    x = np.arange(len(RATIOS))
    compare = ["pseudo_label", "uncertainty_filter"]
    n = len(compare)
    width = 0.30
    offsets = np.linspace(-(n - 1) / 2 * width, (n - 1) / 2 * width, n)

    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, setting in enumerate(compare):
        deltas = [data.get((setting, r), 0) - data.get(("supervised", r), 0)
                  for r in RATIOS]
        bars = ax.bar(
            x + offsets[idx], deltas, width,
            label=SETTING_LABELS[setting],
            color=COLORS[setting], edgecolor="black", linewidth=0.6,
        )
        for bar, d in zip(bars, deltas):
            va = "bottom" if d >= 0 else "top"
            offset = 0.0005 if d >= 0 else -0.0005
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + offset,
                    f"{d:+.4f}", ha="center", va=va,
                    fontsize=8, fontweight="bold")

    ax.axhline(y=0, color="black", linewidth=1.0, linestyle="-")
    ax.set_xlabel("Labeled ratio", fontsize=11)
    ax.set_ylabel("\u0394 Macro AUROC", fontsize=11)
    ax.set_title("Delta Macro AUROC so v\u1edbi supervised baseline",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(RATIO_LABELS, fontsize=10)
    ax.legend(fontsize=8.5, loc="upper left")
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.tick_params(axis="both", which="major", labelsize=10)
    fig.tight_layout()
    _save(fig, os.path.join(PLOTS_DIR, "fig3_delta_macro_auroc_vs_supervised"))


def _fig_perclass(perclass_data, ratio, fig_num, title):
    """Grouped bar chart of per-class AUROC for one labeled ratio."""
    x = np.arange(len(CLASS_NAMES_SHORT))
    n = len(SETTING_ORDER)
    width = 0.24
    offsets = np.linspace(-(n - 1) / 2 * width, (n - 1) / 2 * width, n)

    fig, ax = plt.subplots(figsize=(9, 5))
    for idx, setting in enumerate(SETTING_ORDER):
        vals = perclass_data.get((setting, ratio), [0] * 6)
        bars = ax.bar(
            x + offsets[idx], vals, width,
            label=SETTING_LABELS[setting],
            color=COLORS[setting], edgecolor="black", linewidth=0.5,
        )
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.003,
                    f"{v:.4f}", ha="center", va="bottom",
                    fontsize=6.5, fontweight="bold", rotation=90)

    ax.set_xlabel("Pathology class", fontsize=11)
    ax.set_ylabel("AUROC", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES_SHORT, fontsize=10)

    # Auto y-range with margin
    all_vals = []
    for setting in SETTING_ORDER:
        all_vals.extend(perclass_data.get((setting, ratio), [0] * 6))
    ymin = max(0.65, min(all_vals) - 0.05)
    ymax = min(1.0, max(all_vals) + 0.06)
    ax.set_ylim(ymin, ymax)

    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.tick_params(axis="both", which="major", labelsize=10)
    fig.tight_layout()

    ratio_str = f"{int(ratio*100)}pct"
    _save(fig, os.path.join(PLOTS_DIR, f"fig{fig_num}_per_class_auroc_{ratio_str}"))


def fig7_acceptance_rate(rates):
    """Bar chart of pseudo-label acceptance rates."""
    x = np.arange(len(CLASS_NAMES_SHORT))
    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(x, rates, 0.55, color="#ED7D31", edgecolor="black", linewidth=0.6)
    for bar, r in zip(bars, rates):
        label = f"{r:.2f}%" if r >= 0.1 else f"{r:.2f}%"
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                label, ha="center", va="bottom",
                fontsize=9, fontweight="bold")

    ax.set_xlabel("Pathology class", fontsize=11)
    ax.set_ylabel("Acceptance rate (%)", fontsize=11)
    ax.set_title("Pseudo-label acceptance rate t\u1ea1i threshold = 0.7",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES_SHORT, fontsize=10)
    ax.set_ylim(0, max(rates) * 1.25)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.tick_params(axis="both", which="major", labelsize=10)
    fig.tight_layout()
    _save(fig, os.path.join(PLOTS_DIR, "fig7_pseudo_label_acceptance_rate"))



# ── Main ───────────────────────────────────────────────────────────

def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Load data
    macro_data = load_macro_data()
    perclass_data = load_perclass_data()
    acceptance_rates = load_acceptance_rates()

    # Print loaded data summary
    print("\nMacro AUROC:")
    for setting in SETTING_ORDER:
        vals = [f"{macro_data.get((setting, r), 0):.4f}" for r in RATIOS]
        print(f"  {SETTING_LABELS[setting]:48s} {vals}")

    print("\nPer-class AUROC (5%):")
    for setting in SETTING_ORDER:
        vals = perclass_data.get((setting, 0.05), [0]*6)
        vstr = [f"{v:.4f}" for v in vals]
        print(f"  {SETTING_LABELS[setting]:48s} {vstr}")

    print(f"\nAcceptance rates: {[f'{r:.2f}%' for r in acceptance_rates]}")

    # Generate figures
    print("\nSaving figures:")
    fig2_macro_auroc(macro_data)
    fig3_delta_auroc(macro_data)
    _fig_perclass(perclass_data, 0.05, 4,
                  "Per-class AUROC t\u1ea1i 5% labeled ratio")
    _fig_perclass(perclass_data, 0.10, 5,
                  "Per-class AUROC t\u1ea1i 10% labeled ratio")
    _fig_perclass(perclass_data, 0.20, 6,
                  "Per-class AUROC t\u1ea1i 20% labeled ratio")
    fig7_acceptance_rate(acceptance_rates)

    print("\nDone. All figures exported.")


if __name__ == "__main__":
    main()
