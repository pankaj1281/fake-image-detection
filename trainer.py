from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """Focal Loss for handling imbalanced and hard-to-classify samples.

    Focuses training on hard negatives (fake images that look real) by
    downweighting easy examples. Improves accuracy on edge cases.

    Reference: Lin et al. "Focal Loss for Dense Object Detection"
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: Model output, shape (batch_size, num_classes)
            targets: Ground truth labels, shape (batch_size,)

        Returns:
            Focal loss (scalar)
        """
        ce_loss = nn.functional.cross_entropy(logits, targets, reduction="none")
        probs = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1.0 - probs) ** self.gamma) * ce_loss
        return focal_loss.mean()


@dataclass(slots=True)
class TrainerConfig:
    learning_rate: float = 3e-4
    epochs: int = 20
    early_stopping_patience: int = 4
    checkpoint_dir: Path = Path("models/checkpoints")
    use_focal_loss: bool = True  # Use Focal Loss instead of CrossEntropyLoss


class Trainer:
    """Training loop with transfer learning, mixed precision, early stopping, and logging."""

    def __init__(self, model: nn.Module, config: TrainerConfig) -> None:
        self.model = model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = AdamW(model.parameters(), lr=config.learning_rate)

        # Use Focal Loss for better handling of hard negatives
        if config.use_focal_loss:
            self.criterion = FocalLoss(alpha=0.25, gamma=2.0)
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
        self.writer = SummaryWriter(log_dir="runs/fake_image_detector")

    def _save_checkpoint(self, epoch: int, val_loss: float) -> Path:
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ckpt = self.config.checkpoint_dir / f"epoch_{epoch}_loss_{val_loss:.4f}.pt"
        torch.save(self.model.state_dict(), ckpt)
        return ckpt

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        best_val_loss = float("inf")
        stale_epochs = 0
        self.model.to(self.device)

        for epoch in range(self.config.epochs):
            # --- Training phase ---
            self.model.train()
            train_loss = 0.0
            correct = 0
            total = 0
            num_batches = len(train_loader)
            for batch_idx, (images, labels) in enumerate(train_loader, 1):
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                train_loss += float(loss.item())
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                print(f"\r  Batch {batch_idx}/{num_batches} — loss: {loss.item():.4f}", end="", flush=True)
            print()  # newline after batch progress
            train_loss /= max(1, len(train_loader))
            train_acc = correct / max(1, total)

            # --- Validation phase ---
            val_loss, val_acc = self._evaluate_loader(val_loader)

            self.writer.add_scalars(
                "loss",
                {"train": train_loss, "val": val_loss},
                global_step=epoch,
            )
            self.writer.add_scalars(
                "accuracy",
                {"train": train_acc, "val": val_acc},
                global_step=epoch,
            )

            checkpoint = self._save_checkpoint(epoch, val_loss)
            logger.info(
                "Epoch %d/%d — train_loss: %.4f, train_acc: %.4f, val_loss: %.4f, val_acc: %.4f | checkpoint: %s",
                epoch + 1,
                self.config.epochs,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
                checkpoint,
            )
            print(
                f"Epoch {epoch + 1}/{self.config.epochs} — "
                f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, "
                f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.config.early_stopping_patience:
                    logger.info("Early stopping triggered at epoch %d", epoch + 1)
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    break

    def _evaluate_loader(self, loader: DataLoader) -> tuple[float, float]:
        """Compute average loss and accuracy on a DataLoader."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.inference_mode():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                logits = self.model(images)
                total_loss += float(self.criterion(logits, labels).item())
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        avg_loss = total_loss / max(1, len(loader))
        accuracy = correct / max(1, total)
        return avg_loss, accuracy

    def evaluate(self, test_loader: DataLoader) -> dict[str, float]:
        """Run full evaluation on a test DataLoader and return metrics."""
        import numpy as np
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
        )

        self.model.to(self.device)
        self.model.eval()
        all_preds: list[int] = []
        all_labels: list[int] = []
        all_probs: list[list[float]] = []

        with torch.inference_mode():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                logits = self.model(images)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = logits.argmax(dim=1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().tolist())
                all_probs.extend(probs.tolist())

        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        cm = confusion_matrix(y_true, y_pred)

        report_str = classification_report(
            y_true, y_pred, target_names=["fake", "real"], zero_division=0
        )

        print("\n" + "=" * 60)
        print("  TEST SET EVALUATION RESULTS")
        print("=" * 60)
        print(f"  Accuracy : {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall   : {rec:.4f}")
        print(f"  F1-Score : {f1:.4f}")
        print("-" * 60)
        print("  Classification Report:")
        print(report_str)
        print("-" * 60)
        print("  Confusion Matrix:")
        print(cm)
        print("=" * 60 + "\n")

        return {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
        }
