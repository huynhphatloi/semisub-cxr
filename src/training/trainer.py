"""Core training loop with epoch-level logging and checkpointing.

Implements the ``Trainer`` class that handles forward/backward passes,
validation AUROC computation, best/last checkpoint management, and
optional resume from a saved checkpoint.

Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3A.1–3A.6, 12.4, 14.3
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from src.config import ExperimentConfig
from src.training.losses import masked_bce_with_logits
from src.utils.checkpoint import load_checkpoint, save_checkpoint

logger = logging.getLogger(__name__)


@dataclass
class TrainResult:
    """Summary of a completed training run."""

    best_epoch: int
    best_macro_auroc: float
    per_class_auroc: Dict[str, float] = field(default_factory=dict)


class Trainer:
    """Multi-label classifier trainer with checkpointing and validation.

    Args:
        config: Full experiment configuration.
        model: The ``MultiLabelClassifier`` to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        device: Device to run on (e.g. ``torch.device("cpu")``).
    """

    def __init__(
        self,
        config: ExperimentConfig,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
    ) -> None:
        self.config = config
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Optimizer
        self.optimizer = self._create_optimizer()
        # Scheduler (may be None)
        self.scheduler = self._create_scheduler()

        # Best-checkpoint tracking
        self.best_macro_auroc: float = -1.0
        self.best_epoch: int = -1
        self.best_per_class_auroc: Dict[str, float] = {}

        # Resume state
        self.start_epoch: int = 0
        if config.training.resume_checkpoint:
            self._resume(config.training.resume_checkpoint)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> TrainResult:
        """Run the full training loop with epoch-level logging and checkpointing."""
        num_epochs = self.config.training.num_epochs

        for epoch in range(self.start_epoch, num_epochs):
            train_loss = self._train_epoch(epoch)
            val_metrics = self._validate(epoch)
            self._log_epoch(epoch, train_loss, val_metrics)
            self._save_checkpoint(epoch, val_metrics)

            if self.scheduler is not None:
                self.scheduler.step()

        return TrainResult(
            best_epoch=self.best_epoch,
            best_macro_auroc=self.best_macro_auroc,
            per_class_auroc=self.best_per_class_auroc,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _create_optimizer(self) -> torch.optim.Optimizer:
        tc = self.config.training
        if tc.optimizer == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=tc.learning_rate,
                weight_decay=tc.weight_decay,
            )
        elif tc.optimizer == "sgd":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=tc.learning_rate,
                weight_decay=tc.weight_decay,
                momentum=0.9,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {tc.optimizer}")

    def _create_scheduler(self) -> Optional[Any]:
        tc = self.config.training
        if tc.scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=tc.num_epochs
            )
        elif tc.scheduler == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=10, gamma=0.1
            )
        elif tc.scheduler is None:
            return None
        else:
            raise ValueError(f"Unsupported scheduler: {tc.scheduler}")

    def _resume(self, checkpoint_path: str) -> None:
        """Load model/optimizer state and resume from the saved epoch."""
        logger.info("Resuming from checkpoint: %s", checkpoint_path)
        meta = load_checkpoint(
            checkpoint_path,
            self.model,
            self.optimizer,
            device=str(self.device),
        )
        self.start_epoch = meta["epoch"] + 1
        val_metrics = meta.get("val_metrics", {})
        if "macro_auroc" in val_metrics:
            self.best_macro_auroc = val_metrics["macro_auroc"]
            self.best_epoch = meta["epoch"]
            self.best_per_class_auroc = val_metrics.get("per_class_auroc", {})
        logger.info("Resumed — will start from epoch %d", self.start_epoch)

    def _train_epoch(self, epoch: int) -> float:
        """Single training epoch: forward, masked BCE loss, backward."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in self.train_loader:
            images, labels, loss_mask = self._unpack_batch(batch)
            images = images.to(self.device)
            labels = labels.to(self.device)
            if loss_mask is not None:
                loss_mask = loss_mask.to(self.device)

            logits = self.model(images)
            loss = masked_bce_with_logits(logits, labels, loss_mask)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    @staticmethod
    def _unpack_batch(batch):
        """Handle both 3-tuple and 4-tuple batch formats.

        Standard datasets return ``(images, labels, indices)``.
        CombinedDataset returns ``(images, labels, loss_mask, is_pseudo)``.
        """
        if len(batch) == 4:
            images, labels, loss_mask, _is_pseudo = batch
            return images, labels, loss_mask
        # 3-tuple: standard dataset — no mask needed
        images, labels = batch[0], batch[1]
        return images, labels, None

    def _validate(self, epoch: int) -> Dict[str, Any]:
        """Compute macro-AUROC and per-class AUROC on the validation set."""
        self.model.eval()
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch in self.val_loader:
                images, labels = batch[0], batch[1]
                images = images.to(self.device)
                logits = self.model(images)
                probs = torch.sigmoid(logits)
                all_labels.append(labels.cpu().numpy())
                all_probs.append(probs.cpu().numpy())

        y_true = np.concatenate(all_labels, axis=0)
        y_score = np.concatenate(all_probs, axis=0)

        label_names = self.config.data.label_set
        per_class_auroc: Dict[str, float] = {}
        valid_aurocs = []

        for i, name in enumerate(label_names):
            unique = np.unique(y_true[:, i])
            if len(unique) >= 2:
                auc = float(roc_auc_score(y_true[:, i], y_score[:, i]))
                per_class_auroc[name] = auc
                valid_aurocs.append(auc)
            else:
                per_class_auroc[name] = float("nan")
                logger.warning(
                    "Epoch %d: class '%s' has only one label value — "
                    "AUROC undefined.",
                    epoch,
                    name,
                )

        macro_auroc = float(np.mean(valid_aurocs)) if valid_aurocs else 0.0

        return {"macro_auroc": macro_auroc, "per_class_auroc": per_class_auroc}

    def _log_epoch(
        self, epoch: int, train_loss: float, val_metrics: Dict[str, Any]
    ) -> None:
        logger.info(
            "Epoch %d — train_loss=%.6f  val_macro_auroc=%.4f",
            epoch,
            train_loss,
            val_metrics["macro_auroc"],
        )

    def _save_checkpoint(
        self, epoch: int, val_metrics: Dict[str, Any]
    ) -> None:
        """Save best_checkpoint.pt and last_checkpoint.pt."""
        ckpt_dir = os.path.join(
            self.config.output_dir,
            "checkpoints",
            self.config.setting,
            f"{self.config.split.labeled_ratio}_{self.config.split.seed}",
        )

        # Always save last
        last_path = os.path.join(ckpt_dir, "last_checkpoint.pt")
        save_checkpoint(
            path=last_path,
            model=self.model,
            optimizer=self.optimizer,
            epoch=epoch,
            config=self.config,
            val_metrics=val_metrics,
        )

        # Save best if new best (>= for latest-wins-ties)
        macro = val_metrics["macro_auroc"]
        if macro >= self.best_macro_auroc:
            self.best_macro_auroc = macro
            self.best_epoch = epoch
            self.best_per_class_auroc = val_metrics.get("per_class_auroc", {})

            best_path = os.path.join(ckpt_dir, "best_checkpoint.pt")
            save_checkpoint(
                path=best_path,
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch,
                config=self.config,
                val_metrics=val_metrics,
            )
            logger.info(
                "New best checkpoint at epoch %d (macro_auroc=%.4f)",
                epoch,
                macro,
            )
