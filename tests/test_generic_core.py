"""Regression coverage for the documented, domain-neutral public API."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pdm_learn import (
    DatasetSpec,
    FeatureConfig,
    FeaturePair,
    build_features,
    evaluate_labeled_features,
    rank_candidates,
    rank_unlabeled,
)


class GenericCoreTests(unittest.TestCase):
    def test_mixed_dataset_kinds_and_custom_pair_list(self) -> None:
        datasets = {
            "score": pd.DataFrame(
                {
                    "record": ["A", "B", "C", "D"],
                    "S1": [0.0, 1.0, 2.0, 3.0],
                    "S2": [1.0, 2.0, 3.0, 4.0],
                    "S3": [2.0, 3.0, 4.0, 5.0],
                    "S4": [3.0, 4.0, 5.0, 6.0],
                }
            ),
            "state": pd.DataFrame(
                {
                    "state id": ["A", "B", "C", "D"],
                    "S1": [-2.5, 0.0, 3.5, -2.5],
                    "S2": [0.0, 3.5, -2.5, 0.0],
                    "S3": [3.5, -2.5, 0.0, 3.5],
                    "S4": [0.0, 0.0, 3.5, -2.5],
                }
            ),
            "event": pd.DataFrame(
                {
                    "event id": ["A", "B", "C", "D"],
                    "S1": [0, 1, 0, 1],
                    "S2": [1, 1, 0, 0],
                    "S3": [0, 0, 1, 1],
                    "S4": [1, 0, 1, 0],
                }
            ),
        }

        tables = build_features(
            datasets,
            specs=[
                DatasetSpec("score", id_column="record"),
                DatasetSpec(
                    "state",
                    kind="discrete",
                    levels=(-2.5, 0.0, 3.5),
                    normalize=None,
                    id_column="state id",
                ),
                DatasetSpec("event", kind="binary", normalize=None, id_column="event id"),
            ],
            methods=("pdm", "pearson", "spearman", "mi", "bicor"),
            config=FeatureConfig(
                pairs=[FeaturePair("score", "state"), FeaturePair("event", "score")],
                density_bins=2,
            ),
        )

        self.assertEqual(set(tables), {"pdm", "pearson", "spearman", "mi", "bicor"})
        self.assertEqual(tables["pdm"].columns[0], "record")
        self.assertEqual(tables["pdm"].shape, (4, 11))  # 1 ID + 2x3 and 2x2 PDM blocks
        self.assertEqual(tables["mi"].shape, (4, 3))
        self.assertTrue(np.isfinite(tables["pdm"].iloc[:, 1:].to_numpy(dtype=float)).all())

    def test_missing_binary_entities_and_unshared_samples_are_handled(self) -> None:
        continuous = pd.DataFrame(
            {
                "target": ["A", "B", "C"],
                "S1": [0.0, 1.0, 2.0],
                "S2": [1.0, 2.0, 3.0],
                "S3": [2.0, 3.0, 4.0],
            }
        )
        events = pd.DataFrame(
            {
                "event": ["A"],
                "S1": [1.0],
                "S2": [0.0],
            }
        )

        table = build_features(
            {"events": events, "response": continuous},
            specs=[
                DatasetSpec("events", kind="binary", normalize=None, id_column="event"),
                DatasetSpec("response", normalize=None, id_column="target"),
            ],
            pairs=[FeaturePair("events", "response")],
            methods=("pdm",),
            config=FeatureConfig(density_bins=2, drop_missing=True),
        )["pdm"]

        self.assertEqual(table.iloc[:, 0].tolist(), ["A", "B", "C"])
        self.assertFalse(table.iloc[:, 1:].isna().any().any())

    def test_generic_ranking_and_evaluation_accept_positive_labels(self) -> None:
        table = pd.DataFrame(
            {
                "entity": ["A", "B", "C", "D", "E", "F"],
                "feature one": [1.0, 1.1, 0.9, 0.0, 0.1, 0.2],
                "feature two": [1.0, 0.8, 1.2, 0.1, 0.0, 0.2],
            }
        )
        labels = {"A", "B", "C"}

        ranking = rank_candidates(table, labels, trials=2)
        alias_ranking = rank_unlabeled(table, labels, trials=2)
        evaluation = evaluate_labeled_features(table, labels, n_splits=3)

        self.assertEqual(set(ranking["entity"]), {"D", "E", "F"})
        self.assertEqual(set(alias_ranking["entity"]), {"D", "E", "F"})
        self.assertTrue(ranking["score"].between(0, 1).all())
        self.assertEqual(evaluation.loc[0, "positive_rows"], 3)
        self.assertEqual(evaluation.loc[0, "negative_rows"], 3)
        self.assertTrue(np.isfinite(evaluation.loc[0, "roc_auc"]))


if __name__ == "__main__":
    unittest.main()
