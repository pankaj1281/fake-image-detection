from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


SUPPORTED_DATASETS: set[str] = {"cifake", "faceforensics++", "dfdc", "custom"}


@dataclass(slots=True)
class DatasetConfig:
    name: str
    root: Path

    def validate(self) -> None:
        if self.name.lower() not in SUPPORTED_DATASETS:
            supported = ", ".join(sorted(SUPPORTED_DATASETS))
            raise ValueError(f"Unsupported dataset '{self.name}'. Supported: {supported}")
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset path not found: {self.root}")


class DatasetRegistry:
    """Simple dataset registry to normalize dataset setup across sources."""

    @staticmethod
    def build(name: str, root: str | Path) -> DatasetConfig:
        config = DatasetConfig(name=name, root=Path(root))
        config.validate()
        return config


def build_dataloaders(
    train_dir: str | Path,
    val_dir: str | Path,
    test_dir: str | Path | None = None,
    image_size: int = 224,
    batch_size: int = 16,
    num_workers: int = 2,
) -> tuple[Any, Any, Any | None]:
    """Build train, val, and optionally test DataLoaders with augmentations.

    Returns:
        (train_loader, val_loader, test_loader) — test_loader is None when
        test_dir is not provided or does not exist.

    Args:
        batch_size: Default increased to 16 for GPU efficiency (was 4).
        num_workers: Default increased to 2 for faster data loading (was 0).
    """
    import torch
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder
    from torchvision.transforms import v2

    pin = torch.cuda.is_available()

    # Training transforms with augmentation
    train_transform = v2.Compose(
        [
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomRotation(degrees=15),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            v2.Resize((image_size, image_size)),
            v2.ToImage(),
            v2.ToDtype(dtype=torch.float32, scale=True),
            v2.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    # Validation transforms (no augmentation)
    val_transform = v2.Compose(
        [
            v2.Resize((image_size, image_size)),
            v2.ToImage(),
            v2.ToDtype(dtype=torch.float32, scale=True),
            v2.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    train_dataset = ImageFolder(train_dir, transform=train_transform)
    val_dataset = ImageFolder(val_dir, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )

    test_loader = None
    if test_dir is not None and Path(test_dir).exists():
        test_dataset = ImageFolder(test_dir, transform=val_transform)
        if len(test_dataset) > 0:
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin,
            )

    return train_loader, val_loader, test_loader
