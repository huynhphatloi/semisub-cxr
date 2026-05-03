"""Regenerate outputs/results/ from checkpoint val_metrics.

The checkpoints store val_metrics from training. This script extracts
those metrics and writes consistent JSON + CSV files to outputs/results/,
fixing any stale evaluation artifacts without needing the CheXpert dataset.

Usage:
    python -m scripts.regenerate_results_from_checkpoints
"""

import json
import os

import pandas as pd
import torch


CHECKPOINT_BASE = "outputs/checkpoints"
RESULTS_DIR = "outputs/results"

RUNS = [
    ("supervised",          "0.05_42"),
    ("supervised",          "0.1_42"),
    ("supervised",          "0.2_42"),
    ("pseudo_label",        "0.05_42"),
    ("pseudo_label",        "0.1_42"),
    ("pseudo_label",        "0.2_42"),
    ("uncertainty_filter",  "0.05_42"),
    ("uncertainty_filter",  "0.1_42"),
    ("uncertainty_filter",  "0.2_42"),
]

LABEL_NAMES = [
    "Cardiomegaly", "Pleural Effusion", "Pneumothorax",
    "Consolidation", "Atelectasis", "Edema",
]


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_rows = []

    for setting, ratio_seed in RUNS:
        ckpt_path = os.path.join(
            CHECKPOINT_BASE, setting, ratio_seed, "best_checkpoint.pt"
        )
        config_path = os.path.join(
            CHECKPOINT_BASE, setting, ratio_seed, "config.json"
        )

        if not os.path.isfile(ckpt_path):
            print(f"  SKIP (no checkpoint): {setting}/{ratio_seed}")
            continue

        # Load checkpoint
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        val_metrics = ckpt.get("val_metrics", {})
        epoch = ckpt.get("epoch", -1)

        # Load config for experiment_name
        config_name = f"{setting}_{ratio_seed}"
        labeled_ratio = float(ratio_seed.split("_")[0])
        seed = int(ratio_seed.split("_")[1])

        if os.path.isfile(config_path):
            with open(config_path) as f:
                cfg = json.load(f)
            config_name = cfg.get("experiment_name", config_name)

        macro_auroc = val_metrics.get("macro_auroc", 0.0)
        per_class = val_metrics.get("per_class_auroc", {})

        # Build CSV row
        row = {
            "setting": setting,
            "labeled_ratio": labeled_ratio,
            "seed": seed,
            "macro_auroc": macro_auroc,
        }
        for name in LABEL_NAMES:
            row[f"auroc_{name}"] = per_class.get(name, float("nan"))

        all_rows.append(row)

        # Write individual JSON
        json_path = os.path.join(
            RESULTS_DIR, f"eval_{setting}_{labeled_ratio}_{seed}.json"
        )
        payload = {
            "config_name": config_name,
            "setting": setting,
            "labeled_ratio": labeled_ratio,
            "seed": seed,
            "macro_auroc": macro_auroc,
            "best_epoch": epoch,
        }
        for name in LABEL_NAMES:
            payload[f"auroc_{name}"] = per_class.get(name, float("nan"))
        payload["per_class_auroc"] = {
            name: per_class.get(name, float("nan")) for name in LABEL_NAMES
        }

        with open(json_path, "w") as f:
            json.dump(payload, f, indent=2)

        print(
            f"  ✅ {setting:25s} {ratio_seed:8s}  "
            f"macro={macro_auroc:.4f}  epoch={epoch}"
        )

    # Write CSV
    csv_path = os.path.join(RESULTS_DIR, "evaluation_results.csv")
    df = pd.DataFrame(all_rows)
    df.to_csv(csv_path, index=False)

    print(f"\nWrote {len(all_rows)} rows to {csv_path}")
    print(f"Wrote {len(all_rows)} JSON files to {RESULTS_DIR}/")
    print("Done — results now match checkpoints.")


if __name__ == "__main__":
    main()
