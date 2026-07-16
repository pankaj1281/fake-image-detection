from __future__ import annotations

import io
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from src.inference.predictor import FakeImagePredictor
from src.utils.config import settings
from src.utils.logging import configure_logging

configure_logging()
app = FastAPI(title=settings.project_name, version="1.0.0")
predictor = FakeImagePredictor()


async def _read_image(file: UploadFile) -> Image.Image:
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file uploaded")
    try:
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Invalid image file") from exc
    return image


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/model_info")
def model_info() -> dict[str, object]:
    return {
        "backbones": settings.model_backbones,
        "classes": settings.class_names,
        "ensemble": True,
    }


@app.post("/predict")
async def predict(file: Annotated[UploadFile, File(...)]) -> dict[str, object]:
    image = await _read_image(file)
    prediction = predictor.predict(image)
    return {
        "label": prediction.label,
        "confidence": prediction.confidence,
        "probabilities": prediction.probabilities,
        "explainability": {
            "grad_cam": prediction.explainability.grad_cam,
            "attention_map": prediction.explainability.attention_map,
            "heatmap": prediction.explainability.heatmap,
        },
    }


@app.post("/batch_predict")
async def batch_predict(files: Annotated[list[UploadFile], File(...)]) -> dict[str, object]:
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    results: list[dict[str, object]] = []
    for uploaded in files:
        image = await _read_image(uploaded)
        prediction = predictor.predict(image)
        results.append(
            {
                "filename": uploaded.filename,
                "label": prediction.label,
                "confidence": prediction.confidence,
                "probabilities": prediction.probabilities,
            }
        )
    return {"results": results}
