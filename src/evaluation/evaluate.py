from __future__ import annotations

import numpy as np

from src.evaluation.metrics import evaluate_predictions
from src.utils.config import settings


def main() -> None:
    y_true = list(range(settings.num_classes))
    y_pred = y_true.copy()
    if settings.num_classes > 1:
        y_pred[-1] = settings.num_classes - 2
    y_score = np.eye(settings.num_classes)
    report = evaluate_predictions(y_true, y_pred, y_score, settings.num_classes)
    print(report)


if __name__ == "__main__":
    main()
