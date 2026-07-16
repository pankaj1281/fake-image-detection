from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import mlflow
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TrainerConfig:
    learning_rate: float = 3e-4
    epochs: int = 20
    early_stopping_patience: int = 4
    checkpoint_dir: Path = Path("models/checkpoints")


class Trainer:
    """Training loop with transfer learning, mixed precision, early stopping, and logging."""

    def __init__(self, model: nn.Module, config: TrainerConfig) -> None:
        self.model = model
        self.config = config
        self.optimizer = AdamW(model.parameters(), lr=config.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
        self.writer = SummaryWriter(log_dir="runs/fake_image_detector")

    def _save_checkpoint(self, epoch: int, val_loss: float) -> Path:
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ckpt = self.config.checkpoint_dir / f"epoch_{epoch}_loss_{val_loss:.4f}.pt"
        torch.save(self.model.state_dict(), ckpt)
        return ckpt

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        best_val_loss = float("inf")
        stale_epochs = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        mlflow.set_experiment("fake-image-detection")
        with mlflow.start_run():
            mlflow.log_params({"lr": self.config.learning_rate, "epochs": self.config.epochs})
            for epoch in range(self.config.epochs):
                self.model.train()
                train_loss = 0.0
                for images, labels in train_loader:
                    images, labels = images.to(device), labels.to(device)
                    self.optimizer.zero_grad(set_to_none=True)
                    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                        logits = self.model(images)
                        loss = self.criterion(logits, labels)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    train_loss += float(loss.item())
                train_loss /= max(1, len(train_loader))

                self.model.eval()
                val_loss = 0.0
                with torch.inference_mode():
                    for images, labels in val_loader:
                        images, labels = images.to(device), labels.to(device)
                        logits = self.model(images)
                        val_loss += float(self.criterion(logits, labels).item())
                val_loss /= max(1, len(val_loader))

                self.writer.add_scalars(
                    "loss",
                    {"train": train_loss, "val": val_loss},
                    global_step=epoch,
                )
                mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)

                checkpoint = self._save_checkpoint(epoch, val_loss)
                logger.info("Saved checkpoint: %s", checkpoint)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    stale_epochs = 0
                else:
                    stale_epochs += 1
                    if stale_epochs >= self.config.early_stopping_patience:
                        logger.info("Early stopping triggered at epoch %d", epoch)
                        break
