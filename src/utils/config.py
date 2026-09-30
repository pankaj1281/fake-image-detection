from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path


@dataclass(slots=True)
class Settings:
    """Application settings used by training and inference services."""

    project_name: str = "Fake Image Detection System"
    model_dir: Path = Path("models")
    dataset_mode: str = field(default_factory=lambda: os.getenv("DATASET_MODE", "binary").strip().lower())
    class_names: list[str] = field(default_factory=list)
    model_backbones: list[str] = field(
        default_factory=lambda: ["efficientnet_b4", "convnext_tiny", "vit_b_16"]
    )
    binary_fake_aliases: list[str] = field(
        default_factory=lambda: [
            "fake",
            "ai_generated",
            "deepfake",
            "gan_generated",
            "diffusion_generated",
            "manipulated",
            "synthetic",
        ]
    )

    def __post_init__(self) -> None:
        if self.dataset_mode == "multiclass":
            self.class_names = [
                "ai_generated",
                "deepfake",
                "gan_generated",
                "diffusion_generated",
                "manipulated",
                "real",
            ]
            return
        self.dataset_mode = "binary"
        self.class_names = ["fake", "real"]

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


settings = Settings()
