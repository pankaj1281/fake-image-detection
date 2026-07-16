from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class Settings:
    """Application settings used by training and inference services."""

    project_name: str = "Fake Image Detection System"
    model_dir: Path = Path("models")
    num_classes: int = 6
    class_names: list[str] = field(
        default_factory=lambda: [
            "ai_generated",
            "deepfake",
            "gan_generated",
            "diffusion_generated",
            "manipulated",
            "real",
        ]
    )
    model_backbones: list[str] = field(
        default_factory=lambda: ["efficientnet_b4", "convnext_tiny", "vit_b_16"]
    )


settings = Settings()
