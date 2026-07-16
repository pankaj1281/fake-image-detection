from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class ExplainabilityOutput:
    grad_cam: list[list[float]]
    attention_map: list[list[float]]
    heatmap: list[list[float]]


def _normalize_map(arr: np.ndarray) -> np.ndarray:
    arr = arr - arr.min()
    if arr.max() > 0:
        arr = arr / arr.max()
    return arr


def build_explainability_maps(image: np.ndarray) -> ExplainabilityOutput:
    """Lightweight placeholder explainability maps for API responses."""

    grayscale = image.mean(axis=2)
    grad_cam = _normalize_map(grayscale)
    attention = _normalize_map(np.flipud(grayscale))
    heatmap = _normalize_map((grad_cam + attention) / 2)
    return ExplainabilityOutput(
        grad_cam=grad_cam.tolist(),
        attention_map=attention.tolist(),
        heatmap=heatmap.tolist(),
    )
