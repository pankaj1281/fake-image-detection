from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.config import settings


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train fake-image detection model")
    parser.add_argument(
        "--profile",
        choices=("fast", "accurate"),
        default="fast",
        help="fast: quicker iteration, accurate: full ensemble for best quality",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--backbones",
        type=str,
        default=None,
        help="Comma-separated subset of: efficientnet_b4,convnext_tiny,vit_b_16",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    train_dir = Path("data/train")
    val_dir = Path("data/val")
    if not train_dir.exists() or not val_dir.exists():
        print("Create data/train and data/val with class folders before training.")
        return
    try:
        from src.dataset.loader import build_dataloaders
        from src.inference.models import EnsembleModel
        from src.training.trainer import Trainer, TrainerConfig
    except ModuleNotFoundError as error:
        missing_module = (error.name or "").split(".")[0]
        if missing_module in {"torch", "torchvision"}:
            print(
                "Missing training dependency. Install requirements first:\n"
                "pip install -r requirements.txt"
            )
            return
        raise

    profile_defaults = {
        "fast": {"image_size": 192, "batch_size": 24, "backbones": ["convnext_tiny"], "epochs": 6},
        "accurate": {
            "image_size": 380,
            "batch_size": 16,
            "backbones": settings.model_backbones,
            "epochs": 20,
        },
    }
    selected = profile_defaults[args.profile]
    selected_backbones = (
        [name.strip() for name in args.backbones.split(",") if name.strip()]
        if args.backbones
        else selected["backbones"]
    )

    model = EnsembleModel(
        num_classes=settings.num_classes,
        pretrained=True,
        backbones=selected_backbones,
    )
    config = TrainerConfig(
        learning_rate=args.learning_rate or 3e-4,
        epochs=args.epochs or selected["epochs"],
    )
    trainer = Trainer(model=model, config=config)
    train_loader, val_loader = build_dataloaders(
        train_dir=train_dir,
        val_dir=val_dir,
        image_size=args.image_size or selected["image_size"],
        batch_size=args.batch_size or selected["batch_size"],
        num_workers=args.num_workers,
    )
    trainer.fit(train_loader=train_loader, val_loader=val_loader)


if __name__ == "__main__":
    main()
