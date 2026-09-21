import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pdm_learn.modeling import (
    evaluate_labeled_features,
    feature_ks_pvalues,
    loocv_rank_recovery,
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
        ranking = rank_candidates(self.table, self.labels, trials=20, random_state=7)

        self.assertEqual(set(ranking["entity"]), {"D", "E", "F"})
        self.assertTrue(ranking["score"].between(0, 1).all())
        self.assertTrue((ranking["score_trials"] >= 1).all())
        pd.testing.assert_frame_equal(
            ranking, rank_candidates(self.table, self.labels, trials=20, random_state=7)
        )

    def test_ranking_fits_once_and_averages_only_held_out_predictions(self) -> None:
        table = pd.DataFrame(
            {"entity": [f"id{i}" for i in range(7)], "row": range(7), "extra": range(10, 17)}
        )
        predictions = {i: [] for i in range(2, 7)}
        selection_inputs = []
        models = []
        test = self

        def select_training_features(positive, negative, *, n_features):
            selection_inputs.append((positive.copy(), negative.copy()))
            test.assertEqual(n_features, 1)
            return np.array([0])

        class RecordingClassifier:
            classes_ = np.array([0, 1])

            def fit(self, values, labels):
                positive, negative = selection_inputs[-1]
                test.assertEqual(set(positive[:, 0]), {0, 1})
                test.assertEqual(len(negative), 2)
                test.assertTrue(set(negative[:, 0]).issubset(predictions))
                np.testing.assert_array_equal(values[:, 0], np.r_[positive[:, 0], negative[:, 0]])
                np.testing.assert_array_equal(labels, [1, 1, 0, 0])
                self.training_rows = set(values[:, 0])
                self.probability = (len(models) + 1) / 10
                models.append(self)
                return self

            def predict_proba(self, values):
                test.assertEqual(set(values[:, 0]), set(predictions) - self.training_rows)
                test.assertFalse(set(values[:, 0]) & self.training_rows)
                for row in values[:, 0]:
                    predictions[int(row)].append(self.probability)
                return np.tile([1 - self.probability, self.probability], (len(values), 1))

        with (
            patch(
                "pdm_learn.modeling.select_features", side_effect=select_training_features
            ) as select,
            patch(
                "pdm_learn.modeling._build_classifier",
                side_effect=lambda *args: RecordingClassifier(),
            ) as build,
        ):
            ranking = rank_candidates(table, {"id0", "id1"}, trials=5, n_features=1, random_state=7)

        self.assertEqual(build.call_count, 5)
        self.assertEqual(select.call_count, 5)
        self.assertEqual([call.args[1] for call in build.call_args_list], list(range(7, 12)))
        self.assertGreater(ranking["score_trials"].nunique(), 1)
        for row in ranking.itertuples(index=False):
            expected = predictions[int(row.entity[2:])]
            self.assertEqual(row.score_trials, len(expected))
            if expected:
                self.assertAlmostEqual(row.score, np.mean(expected))
            else:
                self.assertTrue(np.isnan(row.score))

    def test_unscored_candidates_have_missing_scores_and_sort_last(self) -> None:
        ranking = rank_candidates(
            self.table, self.labels, trials=1, feature_selection=False, random_state=7
        )

        self.assertEqual(set(ranking["entity"]), {"D", "E", "F"})
        self.assertEqual(ranking["score_trials"].tolist(), [1, 0, 0])
        self.assertTrue(np.isfinite(ranking.iloc[0]["score"]))
        self.assertTrue(ranking.iloc[1:]["score"].isna().all())

    def test_grouped_ranking_caps_overlaps_and_keeps_all_positives_out_of_negatives(self) -> None:
        table = pd.DataFrame({"entity": [f"id{i}" for i in range(19)], "row": range(19)})
        groups = {"large": range(6), "overlap": range(5, 8), "small": [8]}
        labels = {f"id{i}" for i in range(9)}
        predictions = {i: [] for i in range(9, 19)}
        fits = []
        test = self

        def select_training_features(positive, negative, *, n_features):
            test.assertTrue(set(positive[:, 0]).issubset(range(9)))
            test.assertTrue(set(negative[:, 0]).issubset(predictions))
            return np.array([0])

        class RecordingClassifier:
            classes_ = np.array([0, 1])

            def fit(self, values, targets):
                positive = set(values[targets == 1, 0])
                negative = set(values[targets == 0, 0])
                test.assertEqual(len(positive), len(negative))
                test.assertEqual(len(values), len(positive) + len(negative))
                for members in groups.values():
                    test.assertLessEqual(len(positive.intersection(members)), 2)
                test.assertIn(8, positive)
                test.assertTrue(negative.issubset(predictions))
                fits.append((positive, negative))
                self.negative = negative
                self.probability = (len(fits) % 7 + 1) / 10
                return self

            def predict_proba(self, values):
                test.assertEqual(set(values[:, 0]), set(predictions) - self.negative)
                for row in values[:, 0]:
                    predictions[int(row)].append(self.probability)
                return np.tile([1 - self.probability, self.probability], (len(values), 1))

        kwargs = {
            "positive_groups": {name: {f"id{i}" for i in members}
                                for name, members in groups.items()},
            "max_positive_per_group": 2, "trials": 30, "random_state": 7,
        }
        with (
            patch(
                "pdm_learn.modeling._build_classifier", side_effect=lambda *a: RecordingClassifier()
            ),
            patch("pdm_learn.modeling.select_features", side_effect=select_training_features),
        ):
            ranking = rank_candidates(table, labels, **kwargs)
            first_fits = fits.copy()
            for row in ranking.itertuples(index=False):
                expected = predictions[int(row.entity[2:])]
                self.assertEqual(row.score_trials, len(expected))
                self.assertAlmostEqual(row.score, np.mean(expected))
            fits.clear()
            for scores in predictions.values():
                scores.clear()
            repeated = rank_candidates(table, labels, **kwargs)

        self.assertEqual(fits, first_fits)
        pd.testing.assert_frame_equal(ranking, repeated)
        self.assertEqual(set(ranking["entity"]), {f"id{i}" for i in predictions})
        self.assertGreater(len({frozenset(positive) for positive, _ in fits}), 1)
        self.assertEqual(set.union(*(positive for positive, _ in fits)), set(range(9)))

    def test_grouped_ranking_uses_only_complete_positives_and_leaves_candidates_to_score(self):
        table = self.table.copy()
        table.loc[0, "signal"] = np.nan
        for source, eligible in ((table, {"B", "C"}), (self.table, self.labels)):
            with self.subTest(eligible=eligible):
                ranking = rank_candidates(
                    source, self.labels,
                    positive_groups={"complete": eligible, "absent": {"missing"}},
                    max_positive_per_group=10, trials=30, random_state=7,
                )
                self.assertEqual(set(ranking["entity"]), {"D", "E", "F"})
                self.assertTrue(ranking["score"].notna().all())
                self.assertEqual(ranking["score_trials"].sum(), 30)

    def test_grouped_ranking_validates_cap_and_coverage(self) -> None:
        for cap in (0, -1, True, 1.5):
            with self.subTest(cap=cap), self.assertRaisesRegex(ValueError, "positive integer"):
                rank_candidates(
                    self.table, self.labels,
                    positive_groups={"complex": self.labels}, max_positive_per_group=cap,
                )
        with self.assertRaisesRegex(ValueError, "both"):
            rank_candidates(self.table, self.labels, max_positive_per_group=2)
        with self.assertRaisesRegex(ValueError, "Every eligible positive"):
            rank_candidates(
                self.table, self.labels,
                positive_groups={"incomplete": {"A"}}, max_positive_per_group=2,
            )

    def test_ranking_rejects_duplicate_identifiers_before_fitting(self) -> None:
        table = pd.concat([self.table, self.table.iloc[[-1]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "unique identifiers"):
            rank_candidates(table, self.labels)

    def test_loocv_rank_recovery_returns_curve_and_each_held_out_positive(self) -> None:
        curve, held_out = loocv_rank_recovery(
            self.table,
            self.labels,
            model="logistic_regression",
            trials=20,
            random_state=7,
        )

        self.assertEqual(set(held_out["entity"]), self.labels)
        self.assertEqual(curve["rank_threshold"].tolist(), [0, 1, 2, 3])
        self.assertEqual(int(curve["recovered_positives"].iloc[-1]), len(self.labels))
        self.assertTrue(curve["loocv_area"].between(0, 1).all())
        self.assertTrue(np.isfinite(held_out["score"]).all())

    def test_loocv_rejects_incomplete_score_coverage(self) -> None:
        with self.assertRaisesRegex(ValueError, "Increase trials"):
            loocv_rank_recovery(self.table, self.labels, trials=1, random_state=7)

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

    def test_evaluation_uses_explicit_labels_and_grouped_folds(self) -> None:
        result = evaluate_labeled_features(
            self.table,
            {"A", "B", "C"},
            negative_labels={"D", "E", "F"},
            groups=["p1", "p2", "p3", "n1", "n2", "n3"],
            n_splits=3,
        )

        self.assertEqual(result["positive_rows"].tolist(), [3])
        self.assertEqual(result["negative_rows"].tolist(), [3])
        self.assertEqual(result["cv_strategy"].tolist(), ["stratified_group_kfold"])
        self.assertTrue(np.isfinite(result["roc_auc"]).all())


if __name__ == "__main__":
    unittest.main()
