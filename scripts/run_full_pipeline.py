#!/usr/bin/env python3
"""
Full pipeline runner on local CheXpert data.

Runs all 3 experimental settings end-to-end:
  1. Supervised Baseline (ImageNet pretrained, 5% labeled)
  2. Pseudo-Labeling (teacher → student with confidence thresholding)
  3. Uncertainty-Filtered Pseudo-Labeling (MC Dropout filtering)

Usage:
    python -m scripts.run_full_pipeline

Outputs go to ./outputs/ (checkpoints, splits, pseudo_labels, results, plots).
"""

import os
import subprocess
import sys
import time


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Config files
CONFIG_SUPERVISED = os.path.join(PROJECT_DIR, "configs/local_supervised_baseline.yaml")
CONFIG_PSEUDO = os.path.join(PROJECT_DIR, "configs/local_pseudo_label.yaml")
CONFIG_UNCERTAINTY = os.path.join(PROJECT_DIR, "configs/local_uncertainty_filter.yaml")

# Output paths
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
LABELED_RATIO = 0.05
SEED = 42


def best_ckpt(setting):
    return os.path.join(
        OUTPUT_DIR, "checkpoints", setting,
        f"{LABELED_RATIO}_{SEED}", "best_checkpoint.pt"
    )


def run_step(description, cmd):
    """Run a subprocess step with timing."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    start = time.time()
    result = subprocess.run(cmd, cwd=PROJECT_DIR)
    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"\n  ✗ FAILED (exit code {result.returncode}) after {elapsed:.1f}s")
        sys.exit(1)
    print(f"\n  ✓ Done in {elapsed:.1f}s")
    return elapsed


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timings = {}

    print("="*60)
    print("  FULL PIPELINE: Semi-Supervised Multi-Label CXR")
    print("  Dataset: CheXpert (full Kaggle download)")
    print(f"  Labeled ratio: {LABELED_RATIO}, Seed: {SEED}")
    print("="*60)

    # Step 1: Generate splits
    t = run_step(
        "Step 1/7: Generate labeled/unlabeled splits",
        [sys.executable, "-m", "scripts.generate_splits",
         "--config", CONFIG_SUPERVISED]
    )
    timings["generate_splits"] = t

    # Step 2: Train supervised baseline
    t = run_step(
        "Step 2/7: Train Supervised Baseline (ImageNet, 30 epochs)",
        [sys.executable, "-m", "scripts.train",
         "--config", CONFIG_SUPERVISED]
    )
    timings["train_supervised"] = t

    # Step 3: Generate pseudo-labels (confidence thresholding)
    teacher_ckpt = best_ckpt("supervised")
    t = run_step(
        "Step 3/7: Generate pseudo-labels (confidence thresholding)",
        [sys.executable, "-m", "scripts.generate_pseudo_labels",
         "--config", CONFIG_PSEUDO,
         "--teacher-checkpoint", teacher_ckpt]
    )
    timings["generate_pseudo_labels"] = t

    # Step 4: Train pseudo-label student
    t = run_step(
        "Step 4/7: Train Pseudo-Label Student (30 epochs)",
        [sys.executable, "-m", "scripts.train",
         "--config", CONFIG_PSEUDO]
    )
    timings["train_pseudo_label"] = t

    # Step 5: Generate uncertainty-filtered pseudo-labels
    t = run_step(
        "Step 5/7: Generate uncertainty-filtered pseudo-labels (MC Dropout)",
        [sys.executable, "-m", "scripts.generate_pseudo_labels",
         "--config", CONFIG_UNCERTAINTY,
         "--teacher-checkpoint", teacher_ckpt]
    )
    timings["generate_uncertainty_labels"] = t

    # Step 6: Train uncertainty-filtered student
    t = run_step(
        "Step 6/7: Train Uncertainty-Filtered Student (30 epochs)",
        [sys.executable, "-m", "scripts.train",
         "--config", CONFIG_UNCERTAINTY]
    )
    timings["train_uncertainty"] = t

    # Step 7: Evaluate all checkpoints
    print(f"\n{'='*60}")
    print("  Step 7/7: Evaluate all checkpoints")
    print(f"{'='*60}")

    eval_start = time.time()
    settings_configs = [
        ("supervised", CONFIG_SUPERVISED),
        ("pseudo_label", CONFIG_PSEUDO),
        ("uncertainty_filter", CONFIG_UNCERTAINTY),
    ]

    for setting, cfg in settings_configs:
        ckpt = best_ckpt(setting)
        if not os.path.isfile(ckpt):
            print(f"  [SKIP] {setting}: no checkpoint at {ckpt}")
            continue
        print(f"\n  Evaluating {setting}...")
        subprocess.run([
            sys.executable, "-m", "scripts.evaluate",
            "--config", cfg,
            "--checkpoint", ckpt
        ], cwd=PROJECT_DIR, check=True)

    timings["evaluate_all"] = time.time() - eval_start

    # Summary
    total = sum(timings.values())
    print(f"\n{'='*60}")
    print("  PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"\n  Timing breakdown:")
    for step, t in timings.items():
        print(f"    {step:35s} {t:8.1f}s")
    print(f"    {'TOTAL':35s} {total:8.1f}s")
    print(f"\n  Results: {OUTPUT_DIR}/results/evaluation_results.csv")
    print(f"  Plots:   {OUTPUT_DIR}/plots/")
    print(f"  Checkpoints: {OUTPUT_DIR}/checkpoints/")
    print()


if __name__ == "__main__":
    main()
