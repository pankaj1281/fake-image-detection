from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.inference.explainability import ExplainabilityOutput, build_explainability_maps
from src.inference.models import EnsembleModel
from src.utils.config import settings


@dataclass(slots=True)
class PredictionResult:
    label: str
    confidence: float
    probabilities: dict[str, float]
    explainability: ExplainabilityOutput


class FakeImagePredictor:
    def __init__(self, model_path: str | Path | None = None) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EnsembleModel(num_classes=settings.num_classes).to(self.device)
        self.model.eval()
        if model_path is not None and Path(model_path).exists():
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)

    @staticmethod
    def _to_tensor(image: Image.Image) -> torch.Tensor:
        arr = np.array(image.resize((380, 380))).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        return tensor

    def predict(self, image: Image.Image) -> PredictionResult:
        tensor = self._to_tensor(image).to(self.device)
        with torch.inference_mode():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        best_idx = int(np.argmax(probs))
        class_prob = {label: float(prob) for label, prob in zip(settings.class_names, probs, strict=True)}
        explainability = build_explainability_maps(np.array(image.resize((224, 224))))
        return PredictionResult(
            label=settings.class_names[best_idx],
            confidence=float(probs[best_idx]),
            probabilities=class_prob,
            explainability=explainability,
        )
