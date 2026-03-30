from pathlib import Path
import sys
import unittest

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pdm_learn.modeling import KFold_PR, LOOCV, core_predict, ks_pvalue


class ModelingTests(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(0)
        self.positive = np.array(
            [
                [1.0, 1.1, 1.2],
                [1.2, 1.0, 1.1],
                [0.9, 1.2, 1.0],
            ]
        )
        self.negative = np.array(
            [
                [0.0, 0.2, 0.1],
                [0.1, 0.0, 0.2],
                [0.2, 0.1, 0.0],
                [0.3, 0.2, 0.1],
                [0.1, 0.3, 0.2],
                [0.2, 0.3, 0.1],
            ]
        )

    def test_core_predict_returns_rank_scores(self) -> None:
        scores = core_predict(self.positive, self.negative, 3, model="LR")
        self.assertEqual(scores.shape, (len(self.negative),))
        self.assertTrue(np.isfinite(scores).all())

    def test_loocv_returns_curve_when_requested(self) -> None:
        area, x_values, y_values = LOOCV(
            self.positive,
            self.negative,
            3,
            model="LR",
            equation=True,
        )
        self.assertGreaterEqual(area, 0.0)
        self.assertLessEqual(area, 1.0)
        self.assertEqual(len(x_values), len(y_values))

    def test_kfold_pr_returns_average_precision(self) -> None:
        area, recall, precision = KFold_PR(
            self.positive,
            self.negative,
            3,
            model="LR",
            n_splits=3,
        )
        self.assertGreaterEqual(area, 0.0)
        self.assertLessEqual(area, 1.0)
        self.assertEqual(recall.ndim, 1)
        self.assertEqual(precision.ndim, 1)

    def test_ks_pvalue_matches_feature_count(self) -> None:
        p_values = ks_pvalue(self.positive, self.negative)
        self.assertEqual(len(p_values), self.positive.shape[1])


if __name__ == "__main__":
    unittest.main()
