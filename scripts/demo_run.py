"""
Full pipeline demo using synthetic data — no CheXpert required.
Run: python -m scripts.demo_run
"""
import os, sys, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import (
    DataConfig, ExperimentConfig, ModelConfig, PseudoLabelConfig,
    SplitConfig, TrainingConfig, UncertaintyConfig,
)
from src.training.trainer import Trainer
from src.pseudo_labeling.teacher_inference import run_teacher_inference
from src.pseudo_labeling.mc_dropout import run_mc_dropout_inference
from src.pseudo_labeling.threshold_strategy import (
    PositiveOnlyStrategy, UncertaintyFilteredStrategy,
)
from src.pseudo_labeling.artifact_io import (
    save_pseudo_label_artifact, load_pseudo_label_artifact,
)
from src.evaluation.metrics import compute_auroc, compute_optional_metrics
from src.evaluation.reporting import (
    generate_comparison_bar_chart,
    generate_per_class_auroc_plot,
    generate_results_table,
)
from src.utils.sanity_checks import (
    check_data_loading, check_training_step, check_evaluation,
)

# ── Constants ──────────────────────────────────────────────────────────────
LABEL_NAMES = [
    "Cardiomegaly", "Pleural Effusion", "Pneumothorax",
    "Consolidation", "Atelectasis", "Edema",
]
N_TRAIN, N_VAL, N_UNLAB = 60, 20, 40
DIM, C = 8, 6
DEVICE = torch.device("cpu")
OUTPUT_DIR = "/tmp/semisup_demo"


# ── Helpers ────────────────────────────────────────────────────────────────
def make_config(setting, num_epochs=5, pseudo=False, unc=False):
    ct = {n: 0.5 for n in LABEL_NAMES} if pseudo else {}
    ut = {n: 0.6 for n in LABEL_NAMES} if unc else {}
    return ExperimentConfig(
        experiment_name=f"demo_{setting}",
        setting=setting,
        data=DataConfig("/tmp", "train.csv", "valid.csv", LABEL_NAMES,
                        "zeros", "frontal_only", 32, 0),
        split=SplitConfig(0.1, 42, "/tmp/splits"),
        model=ModelConfig("resnet50_imagenet", None, C, 0.5, [0.5]*3, [0.5]*3),
        training=TrainingConfig(0.005, 8, num_epochs, "adam", 0.0, "cosine"),
        pseudo_label=PseudoLabelConfig(pseudo, ct, "positive_only", "from_scratch"),
        uncertainty=UncertaintyConfig(unc, 5, "predictive_entropy", ut),
        output_dir=OUTPUT_DIR,
    )


def make_data(n, seed=0):
    torch.manual_seed(seed)
    X = torch.randn(n, DIM)
    y = torch.zeros(n, C)
    for c in range(C):
        y[:n//2, c] = 1.0
    perm = torch.randperm(n)
    return X, y[perm]


def make_loader(X, y, bs=8):
    return DataLoader(
        TensorDataset(X, y, torch.arange(len(X))),
        batch_size=bs, shuffle=False,
    )


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(DIM, 32), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(32, C),
        )
    def forward(self, x):
        return self.net(x)


def evaluate_model(model, loader):
    model.eval()
    all_y, all_p = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch[0])
            all_y.append(batch[1].numpy())
            all_p.append(torch.sigmoid(logits).numpy())
    y_true  = np.concatenate(all_y)
    y_score = np.concatenate(all_p)
    metrics = compute_auroc(y_true, y_score, LABEL_NAMES)
    opt = compute_optional_metrics(
        y_true, (y_score >= 0.5).astype(float), y_score,
        LABEL_NAMES, ["macro_f1", "map"],
    )
    return metrics, opt


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    X_train, y_train = make_data(N_TRAIN, seed=0)
    X_val,   y_val   = make_data(N_VAL,   seed=1)
    X_unlab, y_unlab = make_data(N_UNLAB, seed=2)

    train_loader = make_loader(X_train, y_train)
    val_loader   = make_loader(X_val,   y_val)
    unlab_loader = make_loader(X_unlab, y_unlab)

    print("=" * 60)
    print("DEMO: Semi-Supervised Multi-Label CXR Pipeline")
    print("=" * 60)

    # ── Sanity checks ──────────────────────────────────────────────────────
    print("\n[1] Sanity checks")
    ds_check = TensorDataset(X_train, y_train, torch.arange(N_TRAIN))
    check_data_loading(ds_check)
    m_tmp = Model()
    check_training_step(m_tmp, (X_train[:8], y_train[:8]), DEVICE)
    check_evaluation(None, m_tmp, val_loader)
    print("    ✓ data loading, training step, evaluation — all passed")

    # ── Setting 1: Supervised Baseline ────────────────────────────────────
    print("\n[2] Supervised Baseline (5 epochs)")
    m1 = Model()
    r1 = Trainer(make_config("supervised"), m1, train_loader, val_loader, DEVICE).train()
    print(f"    best_epoch={r1.best_epoch}  macro_auroc={r1.best_macro_auroc:.4f}")
    for name, auc in r1.per_class_auroc.items():
        print(f"      {name}: {auc:.4f}")

    # ── Setting 2: Pseudo-Labeling ────────────────────────────────────────
    print("\n[3] Pseudo-Labeling Pipeline")
    print("    Teacher inference on unlabeled set...")
    m1.eval()
    indices, probs = run_teacher_inference(m1, unlab_loader, DEVICE)
    print(f"    Samples={len(indices)}  prob_range=[{probs.min():.3f}, {probs.max():.3f}]")

    strategy = PositiveOnlyStrategy({n: 0.5 for n in LABEL_NAMES})
    pseudo_labels, rej_mask = strategy.accept(probs, None, LABEL_NAMES)
    print("    Per-class accepted pseudo-labels:")
    for j, name in enumerate(LABEL_NAMES):
        print(f"      {name}: {(~rej_mask[:,j]).sum()}/{N_UNLAB} accepted")

    art_dir  = os.path.join(OUTPUT_DIR, "pseudo_labels", "pseudo_label", "0.1_42")
    csv_path = save_pseudo_label_artifact(
        art_dir, indices, probs, pseudo_labels, rej_mask,
        None, {"setting": "pseudo_label"}, LABEL_NAMES,
    )
    df_art = load_pseudo_label_artifact(csv_path)
    print(f"    Artifact: {csv_path}  shape={df_art.shape}")

    pseudo_lbl_t = torch.from_numpy(np.nan_to_num(pseudo_labels, nan=0.0)).float()
    pseudo_mask_t = torch.from_numpy((~rej_mask).astype(np.float32))
    cX = torch.cat([X_train, X_unlab])
    cy = torch.cat([y_train, pseudo_lbl_t])
    cm = torch.cat([torch.ones(N_TRAIN, C), pseudo_mask_t])
    ci = torch.cat([torch.zeros(N_TRAIN), torch.ones(N_UNLAB)])
    comb_loader = DataLoader(TensorDataset(cX, cy, cm, ci), batch_size=8, shuffle=False)

    m2 = Model()
    r2 = Trainer(make_config("pseudo_label", pseudo=True), m2, comb_loader, val_loader, DEVICE).train()
    print(f"    Student: best_epoch={r2.best_epoch}  macro_auroc={r2.best_macro_auroc:.4f}")

    # ── Setting 3: Uncertainty Filtering ──────────────────────────────────
    print("\n[4] Uncertainty Filtering (MC Dropout, 5 passes)")
    idx_mc, mean_probs, uncertainties = run_mc_dropout_inference(
        m1, unlab_loader, DEVICE, num_passes=5,
    )
    print(f"    mean_prob_range=[{mean_probs.min():.3f}, {mean_probs.max():.3f}]")
    print(f"    uncertainty_range=[{uncertainties.min():.4f}, {uncertainties.max():.4f}]")

    unc_strategy = UncertaintyFilteredStrategy(
        {n: 0.5 for n in LABEL_NAMES}, {n: 0.6 for n in LABEL_NAMES},
    )
    unc_pseudo, unc_mask = unc_strategy.accept(mean_probs, uncertainties, LABEL_NAMES)
    print("    Per-class accepted (conf + uncertainty filter):")
    for j, name in enumerate(LABEL_NAMES):
        print(f"      {name}: {(~unc_mask[:,j]).sum()}/{N_UNLAB} accepted")

    unc_lbl_t  = torch.from_numpy(np.nan_to_num(unc_pseudo, nan=0.0)).float()
    unc_mask_t = torch.from_numpy((~unc_mask).astype(np.float32))
    uX = torch.cat([X_train, X_unlab])
    uy = torch.cat([y_train, unc_lbl_t])
    um = torch.cat([torch.ones(N_TRAIN, C), unc_mask_t])
    ui = torch.cat([torch.zeros(N_TRAIN), torch.ones(N_UNLAB)])
    unc_loader2 = DataLoader(TensorDataset(uX, uy, um, ui), batch_size=8, shuffle=False)

    m3 = Model()
    r3 = Trainer(make_config("uncertainty_filter", pseudo=True, unc=True), m3, unc_loader2, val_loader, DEVICE).train()
    print(f"    Student: best_epoch={r3.best_epoch}  macro_auroc={r3.best_macro_auroc:.4f}")

    # ── Evaluation & Reporting ─────────────────────────────────────────────
    print("\n[5] Evaluation & Reporting")
    rows = []
    for setting, model in [("supervised", m1), ("pseudo_label", m2), ("uncertainty_filter", m3)]:
        metrics, opt = evaluate_model(model, val_loader)
        row = {
            "setting": setting, "labeled_ratio": 0.1, "seed": 42,
            "macro_auroc": metrics["macro_auroc"],
            "macro_f1": opt["macro_f1"], "map": opt["map"],
        }
        for name, auc in metrics["per_class_auroc"].items():
            row[f"auroc_{name}"] = auc
        rows.append(row)

    results_df = pd.DataFrame(rows)
    results_dir = os.path.join(OUTPUT_DIR, "results")
    plots_dir   = os.path.join(OUTPUT_DIR, "plots")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    results_df.to_csv(os.path.join(results_dir, "evaluation_results.csv"), index=False)

    print("\n    ┌─ Results summary ─────────────────────────────────────┐")
    print(results_df[["setting", "macro_auroc", "macro_f1", "map"]].to_string(index=False))
    print("    └───────────────────────────────────────────────────────┘")

    generate_comparison_bar_chart(results_df, plots_dir)
    generate_per_class_auroc_plot(results_df, plots_dir, label_names=LABEL_NAMES)
    md_path  = generate_results_table(results_df, plots_dir, fmt="markdown")
    tex_path = generate_results_table(results_df, plots_dir, fmt="latex")

    print(f"\n    Plots:   {plots_dir}/macro_auroc_comparison.png")
    print(f"             {plots_dir}/per_class_auroc_comparison.png")
    print(f"    Tables:  {md_path}")
    print(f"             {tex_path}")

    print("\n[6] Markdown results table:")
    print(open(md_path).read())

    print("[7] Output directory tree:")
    for root, dirs, files in os.walk(OUTPUT_DIR):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        level = root.replace(OUTPUT_DIR, "").count(os.sep)
        indent = "  " * level
        print(f"{indent}{os.path.basename(root) or 'semisup_demo'}/")
        for f in sorted(files):
            size = os.path.getsize(os.path.join(root, f))
            print(f"{indent}  {f}  ({size:,} bytes)")

    print("\n" + "=" * 60)
    print("DEMO COMPLETE — all outputs in", OUTPUT_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
