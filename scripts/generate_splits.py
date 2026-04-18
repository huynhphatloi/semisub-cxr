"""CLI script for generating low-label splits.

Loads an experiment config, creates the CheXpert training dataset,
generates a labeled/unlabeled split, and runs data-loading and
split-validity sanity checks.

Usage::

    python -m scripts.generate_splits --config configs/supervised_baseline.yaml

Validates: Requirements 2.1, 15.1
"""

import argparse
import logging
import os

from src.config import load_config
from src.data.chexpert_dataset import CheXpertDataset
from src.data.split_generator import generate_split
from src.data.transforms import get_eval_transform
from src.utils.sanity_checks import check_data_loading, check_split_validity
from src.utils.seed import set_all_seeds

logger = logging.getLogger(__name__)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate a labeled/unlabeled split for CheXpert."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML experiment config file.",
    )
    args = parser.parse_args(argv)

    # --- Load config ---------------------------------------------------------
    config = load_config(args.config)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    logger.info("Loaded config: %s", config.experiment_name)

    # --- Set seeds -----------------------------------------------------------
    set_all_seeds(config.split.seed)

    # --- Build dataset -------------------------------------------------------
    csv_path = os.path.join(config.data.dataset_path, config.data.train_csv)
    transform = get_eval_transform(
        image_size=config.data.image_size,
        normalization_mean=config.model.normalization_mean,
        normalization_std=config.model.normalization_std,
    )

    dataset = CheXpertDataset(
        csv_path=csv_path,
        image_root=config.data.dataset_path,
        label_set=config.data.label_set,
        uncertain_policy=config.data.uncertain_policy,
        view_policy=config.data.view_policy,
        transform=transform,
    )

    logger.info("Dataset loaded: %d samples.", len(dataset))

    # --- Sanity check: data loading ------------------------------------------
    check_data_loading(dataset)

    # --- Generate split ------------------------------------------------------
    split_meta = generate_split(
        dataset=dataset,
        labeled_ratio=config.split.labeled_ratio,
        seed=config.split.seed,
        splits_dir=config.split.splits_dir,
        min_positive_per_class=config.split.min_positive_per_class,
        class_names=config.data.label_set,
    )

    # --- Sanity check: split validity ----------------------------------------
    check_split_validity(split_meta, len(dataset), config.split.labeled_ratio)

    # --- Summary -------------------------------------------------------------
    print("\n=== Split Generation Summary ===")
    print(f"  Experiment : {config.experiment_name}")
    print(f"  Seed       : {split_meta.seed}")
    print(f"  Ratio      : {split_meta.labeled_ratio}")
    print(f"  Total      : {split_meta.total_train_size}")
    print(f"  Labeled    : {split_meta.labeled_size}")
    print(f"  Unlabeled  : {split_meta.unlabeled_size}")
    print(f"  Timestamp  : {split_meta.generation_timestamp}")
    print("  Per-class positive counts (labeled subset):")
    for cls, cnt in split_meta.per_class_positive_counts.items():
        print(f"    {cls}: {cnt}")
    print("================================\n")


if __name__ == "__main__":
    main()
