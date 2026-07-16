from __future__ import annotations

import numpy as np

from src.evaluation.metrics import evaluate_predictions
from src.utils.config import settings


def main() -> None:
    y_true = [0, 1, 2, 3, 4, 5]
    y_pred = [0, 1, 2, 2, 4, 5]
    y_score = np.eye(settings.num_classes)
    report = evaluate_predictions(y_true, y_pred, y_score, settings.num_classes)
    print(report)


if __name__ == "__main__":
    main()
