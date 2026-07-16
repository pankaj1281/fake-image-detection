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
    image_size: int = 380,
    batch_size: int = 16,
    num_workers: int = 2,
) -> tuple[Any, Any]:
    import torch
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder
    from torchvision.transforms import v2

    transform = v2.Compose(
        [
            v2.Resize((image_size, image_size)),
            v2.ToImage(),
            v2.ToDtype(dtype=torch.float32, scale=True),
        ]
    )
    train_dataset = ImageFolder(train_dir, transform=transform)
    val_dataset = ImageFolder(val_dir, transform=transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
