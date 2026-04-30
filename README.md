# Semi-Supervised Multi-Label Classification of Chest X-Ray Images

A systematic comparison of supervised and semi-supervised approaches for multi-label chest X-ray classification on the [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/) dataset, using pseudo-labeling with MC Dropout uncertainty estimation.

## Motivation

Labeling medical images requires board-certified radiologists and is expensive. Hospitals have millions of unlabeled X-rays but only a fraction carry verified labels. This project investigates whether semi-supervised learning can close the performance gap when only **5-20%** of training data is labeled.

## Method Overview

We implement and compare three training strategies under identical conditions:

| Setting | Training Data | Pseudo-Labels | Uncertainty Filter |
|---|---|---|---|
| **Supervised Baseline** | Labeled only | No | No |
| **Pseudo-Label** | Labeled + Pseudo | Confidence > 0.7 | No |
| **Uncertainty Filter** | Labeled + Pseudo | Confidence > 0.7 | Entropy < 0.5 |

All settings use **ResNet-50** (ImageNet pretrained) as the backbone with a dropout + linear classification head (2048 -> 6 classes).

**Pipeline:**

```
CheXpert Dataset -> Stratified Split -> Teacher Training (Supervised)
    -> Teacher Inference on Unlabeled -> Threshold / MC Dropout Filtering
    -> Student Training on Labeled + Pseudo-Labeled -> Evaluation
```

**Key techniques:**
- **Transfer learning**: ImageNet pretrained ResNet-50 backbone
- **Iterative stratified sampling**: Ensures class balance in low-label splits
- **Class-wise pseudo-labeling**: Per-class confidence thresholds (positive-only strategy)
- **MC Dropout uncertainty**: T=20 stochastic forward passes, predictive entropy filtering
- **Masked BCE loss**: Only accepted pseudo-labels contribute to gradient

## Results

Macro AUROC across 3 settings and 3 labeled ratios (seed=42):

| Setting | 5% Labeled | 10% Labeled | 20% Labeled |
|---|---|---|---|
| Supervised Baseline | 0.8429 | 0.8325 | 0.8497 |
| Pseudo-Label | 0.8414 | **0.8514** | 0.8508 |
| Uncertainty Filter | 0.8377 | 0.8458 | **0.8519** |

**Key findings:**
- All methods achieve 0.83-0.85 macro AUROC even with only 5% labeled data
- Pseudo-labeling is most effective at 10% (+1.9 points over supervised)
- Uncertainty filtering shows the most consistent improvement across ratios
- Pseudo-labeling improves Atelectasis by +11.6 AUROC points at 5%

Per-class AUROC at 5% labeled ratio:

| Class | Supervised | Pseudo-Label | Uncertainty Filter |
|---|---|---|---|
| Cardiomegaly | 0.8338 | 0.8073 | 0.7818 |
| Pleural Effusion | 0.9181 | 0.8892 | 0.8941 |
| Pneumothorax | 0.7788 | 0.7516 | 0.7810 |
| Consolidation | 0.9162 | 0.8862 | 0.8528 |
| Atelectasis | 0.7122 | **0.8279** | **0.8293** |
| Edema | 0.8982 | 0.8862 | 0.8874 |

## Dataset Setup

### Option A: Kaggle Download (Colab)

The dataset is automatically downloaded and unzipped in the Colab notebook from Google Drive. Place `archive.zip` (10.7 GB) from [Kaggle CheXpert](https://www.kaggle.com/datasets/ashery/chexpert) on your Drive.

### Option B: Local Setup

1. Download CheXpert from Kaggle
2. Extract to a directory containing `train/`, `valid/`, `train.csv`, `valid.csv`
3. Create a symlink for CSV path resolution:
```bash
mkdir -p /path/to/data/CheXpert-v1.0-small
ln -s /path/to/data/train /path/to/data/CheXpert-v1.0-small/train
ln -s /path/to/data/valid /path/to/data/CheXpert-v1.0-small/valid
```
4. Update `dataset_path` in the YAML config files

## How to Run

### Quick Demo (No CheXpert Required)

```bash
python -m scripts.demo_run
```

Runs the full pipeline on synthetic data to verify everything works.

### Google Colab (Recommended)

Open `notebooks/colab_runner.ipynb` in Colab and run cells sequentially. All training cells support `--resume` for automatic recovery after disconnects.

### Local Training

```bash
# Install dependencies
pip install -r requirements.txt

# Step 1: Generate splits
python -m scripts.generate_splits --config configs/supervised_baseline.yaml

# Step 2: Train supervised baseline
python -m scripts.train --config configs/supervised_baseline.yaml --resume

# Step 3: Generate pseudo-labels
python -m scripts.generate_pseudo_labels \
    --config configs/pseudo_label.yaml \
    --teacher-checkpoint outputs/checkpoints/supervised/0.05_42/best_checkpoint.pt

# Step 4: Train pseudo-label student
python -m scripts.train --config configs/pseudo_label.yaml --resume

# Step 5: Generate uncertainty-filtered pseudo-labels
python -m scripts.generate_pseudo_labels \
    --config configs/uncertainty_filter.yaml \
    --teacher-checkpoint outputs/checkpoints/supervised/0.05_42/best_checkpoint.pt

# Step 6: Train uncertainty-filtered student
python -m scripts.train --config configs/uncertainty_filter.yaml --resume

# Step 7: Evaluate all checkpoints
python -m scripts.evaluate \
    --config configs/supervised_baseline.yaml \
    --checkpoint outputs/checkpoints/supervised/0.05_42/best_checkpoint.pt
```

The `--resume` flag auto-detects `last_checkpoint.pt` and continues from the last saved epoch. If training is already complete, it skips entirely.

## Experiment Settings

All experiments use these hyperparameters:

| Parameter | Value |
|---|---|
| Backbone | ResNet-50 (ImageNet pretrained) |
| Input size | 224 x 224 |
| Batch size | 32 |
| Optimizer | Adam (lr=1e-4, weight_decay=1e-4) |
| Scheduler | Cosine Annealing (T_max=30) |
| Epochs | 30 |
| Dropout | 0.5 |
| Labeled ratios | 5%, 10%, 20% |
| Confidence threshold | 0.7 (all classes) |
| MC Dropout passes | 20 |
| Uncertainty threshold | 0.5 (all classes) |
| Seed | 42 |
| Uncertain label policy | U-Zeros (map -1 to 0) |
| View filter | Frontal only |

## Project Structure

```
semisub-cxr/
  configs/                    # YAML experiment configs
    supervised_baseline.yaml
    pseudo_label.yaml
    uncertainty_filter.yaml
  scripts/                    # CLI entry points
    train.py                  # Train with --resume support
    evaluate.py               # Evaluate checkpoint
    generate_splits.py        # Create labeled/unlabeled split
    generate_pseudo_labels.py # Generate pseudo-labels from teacher
    demo_run.py               # Full pipeline demo (synthetic data)
  src/
    config.py                 # Dataclass configs + YAML loader
    data/
      chexpert_dataset.py     # CheXpert PyTorch Dataset
      combined_dataset.py     # Labeled + pseudo-labeled merger
      split_generator.py      # Stratified split with persistence
      transforms.py           # Train/eval augmentation pipelines
    models/
      backbone_factory.py     # Registry pattern for backbones
      classifier.py           # ResNet-50 + Dropout + Linear head
    training/
      trainer.py              # Training loop + checkpointing
      losses.py               # Masked BCE loss
    pseudo_labeling/
      teacher_inference.py    # Teacher forward pass
      mc_dropout.py           # MC Dropout uncertainty estimation
      threshold_strategy.py   # Accept/reject strategies
      artifact_io.py          # Save/load pseudo-label CSV
    evaluation/
      metrics.py              # AUROC, F1, MAP computation
      evaluator.py            # Checkpoint evaluation
      reporting.py            # Charts + tables generation
    utils/
      seed.py                 # Reproducibility (Python/NumPy/PyTorch/CUDA)
      checkpoint.py           # Save/load with metadata
      sanity_checks.py        # Pipeline validation at each stage
  outputs/                    # Generated artifacts
    splits/                   # Split JSON files
    checkpoints/              # Model checkpoints (best + last)
    pseudo_labels/            # Pseudo-label CSV artifacts
    results/                  # Evaluation CSV/JSON
    plots/                    # Charts (PNG)
  notebooks/
    colab_runner.ipynb        # Google Colab notebook
  tests/                      # Unit + property-based tests
```

## Expected Outputs

After a full run, `outputs/` contains:

```
outputs/
  splits/split_r0.05_s42.json           # Labeled/unlabeled indices
  checkpoints/
    supervised/0.05_42/
      best_checkpoint.pt                 # Best model weights
      last_checkpoint.pt                 # Last epoch (for resume)
      config.json                        # Experiment config copy
    pseudo_label/0.05_42/...
    uncertainty_filter/0.05_42/...
  pseudo_labels/
    pseudo_label/0.05_42/
      pseudo_labels.csv                  # Per-sample pseudo-labels
      config.json
  results/evaluation_results.csv         # All AUROC results
  plots/
    macro_auroc_comparison.png
    per_class_auroc_comparison.png
    results_table.md
    results_table.tex
```

## Known Limitations

- **Small validation set**: CheXpert validation has only 202 frontal views, introducing variance in AUROC estimates
- **Fixed thresholds**: Confidence (0.7) and uncertainty (0.5) thresholds are the same for all classes; adaptive per-class thresholds could improve rare class detection
- **No LVM-Med**: Domain-specific pretrained weights were not available; all experiments use ImageNet backbone
- **Single seed**: Results are from seed=42 only; multiple seeds would provide confidence intervals
- **No iterative pseudo-labeling**: Only one round of teacher-student training; multiple rounds could progressively improve quality

## References

1. Irvin, J., et al. (2019). CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels. *AAAI*, 33(01), 590-597.
2. Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian Approximation. *ICML*, 48, 1050-1059.
3. Lee, D. H. (2013). Pseudo-Label: The Simple and Efficient Semi-Supervised Learning Method. *ICML Workshop*.
4. He, K., et al. (2016). Deep Residual Learning for Image Recognition. *CVPR*, 770-778.
5. Sechidis, K., et al. (2011). On the Stratification of Multi-label Data. *ECML PKDD*, 145-158.
