import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pdm_learn.modeling import (
    evaluate_labeled_features,
    feature_ks_pvalues,
    rank_candidates,
    select_features,
)


class ModelingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.table = pd.DataFrame(
            {
                "entity": ["A", "B", "C", "D", "E", "F"],
                "metadata": ["left", "left", "left", "right", "right", "right"],
                "signal": [1.0, 1.1, 0.9, 0.0, 0.2, 0.1],
                "secondary": [0.8, 1.0, 0.9, 0.1, 0.0, 0.2],
            }
        )
        self.labels = {"A", "B", "C"}

    def test_feature_selection_prefers_separating_columns(self) -> None:
        positive = np.asarray([[1.0, 0.1], [1.2, 0.2], [0.9, 0.3]])
        negative = np.asarray([[0.0, 0.2], [0.1, 0.1], [0.2, 0.3]])

        pvalues = feature_ks_pvalues(positive, negative)
        selected = select_features(positive, negative, n_features=1)

        self.assertEqual(pvalues.shape, (2,))
        self.assertEqual(selected.tolist(), [0])

    def test_ranking_ignores_non_numeric_metadata(self) -> None:
        ranking = rank_candidates(self.table, self.labels, trials=3, random_state=7)

        self.assertEqual(set(ranking["entity"]), {"D", "E", "F"})
        self.assertTrue(ranking["score"].between(0, 1).all())

    def test_evaluation_returns_cross_validated_scores(self) -> None:
        result = evaluate_labeled_features(
            {"first": self.table, "second": self.table},
            (label for label in self.labels),
            n_splits=3,
        )

        self.assertEqual(result["positive_rows"].tolist(), [3, 3])
        self.assertEqual(result["negative_rows"].tolist(), [3, 3])
        self.assertTrue(np.isfinite(result["roc_auc"]).all())
        self.assertTrue(np.isfinite(result["average_precision"]).all())


if __name__ == "__main__":
    unittest.main()
