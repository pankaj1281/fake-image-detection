from __future__ import annotations

import torch
from torch import nn
from torchvision import models


class EnsembleModel(nn.Module):
    SUPPORTED_BACKBONES: tuple[str, ...] = (
        "efficientnet_b4",
        "convnext_tiny",
        "vit_b_16",
    )

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = False,
        backbones: list[str] | None = None,
    ) -> None:
        super().__init__()
        selected_backbones = backbones or list(self.SUPPORTED_BACKBONES)
        if not selected_backbones:
            raise ValueError("At least one backbone must be selected.")
        unsupported = sorted(set(selected_backbones) - set(self.SUPPORTED_BACKBONES))
        if unsupported:
            supported = ", ".join(self.SUPPORTED_BACKBONES)
            raise ValueError(
                f"Unsupported backbones: {', '.join(unsupported)}. Supported: {supported}"
            )

        self.backbone_names = selected_backbones
        self.backbones = nn.ModuleList()
        for backbone_name in self.backbone_names:
            self.backbones.append(
                self._build_backbone(
                    backbone_name=backbone_name,
                    num_classes=num_classes,
                    pretrained=pretrained,
                )
            )

    @staticmethod
    def _build_backbone(backbone_name: str, num_classes: int, pretrained: bool) -> nn.Module:
        if backbone_name == "efficientnet_b4":
            weights = models.EfficientNet_B4_Weights.DEFAULT if pretrained else None
            model = models.efficientnet_b4(weights=weights)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
            return model

        if backbone_name == "convnext_tiny":
            weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            model = models.convnext_tiny(weights=weights)
            model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
            return model

        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        model = models.vit_b_16(weights=weights)
        model.heads[0] = nn.Linear(model.heads[0].in_features, num_classes)
        return model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = torch.stack([model(x) for model in self.backbones], dim=0)
        return logits.mean(dim=0)
