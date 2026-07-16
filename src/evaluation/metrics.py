from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(slots=True)
class EvaluationReport:
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    confusion_matrix: list[list[int]]


def evaluate_predictions(
    y_true: list[int], y_pred: list[int], y_score: np.ndarray, num_classes: int
) -> EvaluationReport:
    return EvaluationReport(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        recall=float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        f1_score=float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        roc_auc=float(roc_auc_score(np.eye(num_classes)[y_true], y_score, multi_class="ovr")),
        confusion_matrix=confusion_matrix(y_true, y_pred).tolist(),
    )
