from __future__ import annotations

from pathlib import Path

from PIL import Image

from src.inference.predictor import FakeImagePredictor


def main(image_path: str) -> None:
    predictor = FakeImagePredictor()
    image = Image.open(image_path).convert("RGB")
    result = predictor.predict(image)
    print(result)


if __name__ == "__main__":
    sample = Path("data/sample.jpg")
    if sample.exists():
        main(str(sample))
    else:
        print("No sample image found at data/sample.jpg")
