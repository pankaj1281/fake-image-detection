import importlib.util
import unittest


class MetricsTests(unittest.TestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("numpy") and importlib.util.find_spec("sklearn"),
        "numpy and scikit-learn are required for metrics test",
    )
    def test_metrics_report_contains_expected_fields(self) -> None:
        import numpy as np

        from src.evaluation.metrics import evaluate_predictions

        y_true = [0, 1, 2, 3, 4, 5]
        y_pred = [0, 1, 2, 3, 4, 5]
        y_score = np.eye(6)
        report = evaluate_predictions(y_true, y_pred, y_score, 6)
        self.assertEqual(report.accuracy, 1.0)
        self.assertEqual(len(report.confusion_matrix), 6)


if __name__ == "__main__":
    unittest.main()
