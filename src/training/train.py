from __future__ import annotations

from pathlib import Path

from src.dataset.loader import build_dataloaders
from src.inference.models import EnsembleModel
from src.training.trainer import Trainer, TrainerConfig
from src.utils.config import settings


def main() -> None:
    train_dir = Path("data/train")
    val_dir = Path("data/val")
    if not train_dir.exists() or not val_dir.exists():
        print("Create data/train and data/val with class folders before training.")
        return

    model = EnsembleModel(num_classes=settings.num_classes, pretrained=True)
    config = TrainerConfig()
    trainer = Trainer(model=model, config=config)
    train_loader, val_loader = build_dataloaders(train_dir=train_dir, val_dir=val_dir)
    trainer.fit(train_loader=train_loader, val_loader=val_loader)


if __name__ == "__main__":
    main()
