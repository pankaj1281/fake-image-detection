from __future__ import annotations

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def build_train_augmentations(image_size: int = 380) -> A.Compose:
    return A.Compose(
        [
            A.RandomResizedCrop(size=(image_size, image_size), scale=(0.7, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.ImageCompression(quality_range=(60, 100), p=0.3),
            A.ColorJitter(p=0.4),
            A.Normalize(),
            ToTensorV2(),
        ]
    )


def build_eval_augmentations(image_size: int = 380) -> A.Compose:
    return A.Compose([A.Resize(image_size, image_size), A.Normalize(), ToTensorV2()])
