"""CLI entry point for model evaluation and report generation.

Usage::

    python -m scripts.evaluate --config configs/supervised_baseline.yaml \\
                               --checkpoint outputs/checkpoints/.../best_checkpoint.pt

Validates: Requirements 7.1, 7.2, 7.3, 8.1
"""

import argparse
import os

import pandas as pd
import torch

from src.config import load_config
from src.evaluation.evaluator import evaluate_checkpoint, save_evaluation_results
from src.evaluation.reporting import (
    generate_comparison_bar_chart,
    generate_per_class_auroc_plot,
    generate_results_table,
)
from src.utils.logging_utils import setup_logging
from src.utils.seed import set_all_seeds


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained checkpoint and generate reports."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML experiment config.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint to evaluate.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (e.g. 'cpu', 'cuda:0'). Auto-detected if omitted.",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Setup logging
    logger = setup_logging(config.output_dir, config.experiment_name)
    logger.info("Starting evaluation for %s", config.experiment_name)

    # Set seeds for reproducibility
    set_all_seeds(config.split.seed)

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Evaluate
    metrics = evaluate_checkpoint(args.checkpoint, config, device)
    logger.info("Macro AUROC: %.4f", metrics["macro_auroc"])
    for name, auc in metrics.get("per_class_auroc", {}).items():
        logger.info(
            "  %s: %.4f", name, auc
        )

    # Save results
    results_dir = os.path.join(config.output_dir, "results")
    csv_path = save_evaluation_results(metrics, config, results_dir)

    # Generate plots and tables if results CSV has data
    plots_dir = os.path.join(config.output_dir, "plots")
    results_df = pd.read_csv(csv_path)

    if len(results_df) > 0:
        generate_comparison_bar_chart(results_df, plots_dir)
        generate_per_class_auroc_plot(
            results_df, plots_dir, label_names=config.data.label_set
        )
        generate_results_table(results_df, plots_dir, fmt="markdown")
        generate_results_table(results_df, plots_dir, fmt="latex")

    logger.info(
        "Evaluation complete. Results in %s, plots in %s",
        results_dir,
        plots_dir,
    )


if __name__ == "__main__":
    main()
