from __future__ import annotations

import torch
from torch import nn
from torchvision import models


class EnsembleModel(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = False) -> None:
        super().__init__()
        efficientnet_weights = (
            models.EfficientNet_B4_Weights.DEFAULT if pretrained else None
        )
        convnext_weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        vit_weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None

        self.efficientnet = models.efficientnet_b4(weights=efficientnet_weights)
        self.convnext = models.convnext_tiny(weights=convnext_weights)
        self.vit = models.vit_b_16(weights=vit_weights)

        self.efficientnet.classifier[1] = nn.Linear(
            self.efficientnet.classifier[1].in_features, num_classes
        )
        self.convnext.classifier[2] = nn.Linear(
            self.convnext.classifier[2].in_features, num_classes
        )
        self.vit.heads[0] = nn.Linear(self.vit.heads[0].in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = torch.stack(
            [
                self.efficientnet(x),
                self.convnext(x),
                self.vit(x),
            ],
            dim=0,
        )
        return logits.mean(dim=0)
