from dataclasses import dataclass
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pdm_learn.datasets import DatasetSpec
from pdm_learn.pair_features import (
    PairFeatureConfig,
    build_pair_density_features,
    build_pair_features,
)


@dataclass(frozen=True)
class LocalPair:
    """A local pair object verifies the builder's structural pair API."""

    left: str
    right: str


class PairFeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        # Identifier headers vary and are not always the first matrix column.
        # Sample header whitespace also varies between matrices intentionally.
        self.datasets = {
            "response": pd.DataFrame(
                {
                    "sample 1": [1.0, 4.0],
                    "compound id": ["Drug A", "Drug B"],
                    " sample 2 ": [2.0, 2.0],
                    "sample 3": [3.0, 5.0],
                    "sample 4": [4.0, 1.0],
                }
            ),
            "state": pd.DataFrame(
                {
                    "protein code": ["P1", "P2"],
                    " sample 1 ": [0.0, 1.0],
                    "sample 2": [1.0, 1.0],
                    "sample 3": [0.0, 0.0],
                    "sample 4": [1.0, 0.0],
                }
            ),
            "membership": pd.DataFrame(
                {
                    "sample 1": [1.0, 0.0],
                    " sample 2 ": [0.0, 1.0],
                    "member id": ["P1", "P2"],
                    "sample 3": [1.0, 0.0],
                    "sample 4": [0.0, 1.0],
                }
            ),
        }
        self.specs = [
            DatasetSpec("response", id_column="compound id", normalize=None),
            DatasetSpec("state", kind="discrete", levels=(0.0, 1.0), normalize=None, id_column="protein code"),
            DatasetSpec("membership", kind="binary", normalize=None, id_column="member id"),
        ]
        self.pairs = pd.DataFrame(
            {
                "source drug": [" Drug A ", "Drug B", "missing"],
                "target protein": [" P1", "P2 ", "P1"],
                # This missing value must not be used to drop a valid feature row.
                "optional metadata": ["first", np.nan, "unresolved"],
            },
            index=[10, 20, 30],
        )

    def test_builds_pdm_and_statistics_for_continuous_discrete_binary_inputs(self) -> None:
        tables = build_pair_features(
            self.datasets,
            self.pairs,
            left_id_column="source drug",
            right_id_column="target protein",
            specs=self.specs,
            pairs=[LocalPair("response", "state"), LocalPair("response", "membership")],
            methods=("pdm", "pearson", "spearman", "mi", "bicor"),
            config=PairFeatureConfig(density_bins=2, drop_missing=False),
        )

        self.assertEqual(set(tables), {"pdm", "pearson", "spearman", "mi", "bicor"})
        self.assertEqual(tables["pdm"].index.tolist(), [10, 20, 30])
        self.assertEqual(
            tables["pdm"].columns[:3].tolist(),
            ["source drug", "target protein", "optional metadata"],
        )
        self.assertEqual(tables["pdm"].shape[1], 11)  # 3 metadata + two 2-by-2 PDM blocks
        self.assertEqual(tables["pearson"].shape[1], 5)  # 3 metadata + one feature per matrix pair
        self.assertTrue(np.isfinite(tables["pdm"].iloc[:2, 3:].to_numpy(dtype=float)).all())
        self.assertTrue(tables["pdm"].iloc[2, 3:].isna().all())
        self.assertTrue(tables["pearson"].iloc[2, 3:].isna().all())

    def test_config_pairs_and_drop_missing_only_consider_generated_features(self) -> None:
        features = build_pair_density_features(
            self.datasets,
            self.pairs,
            left_id_column="source drug",
            right_id_column="target protein",
            dataset_specs=self.specs,
            config=PairFeatureConfig(
                pairs=[("response", "state")],
                density_bins=2,
                drop_missing=True,
            ),
        )

        self.assertEqual(features.index.tolist(), [10, 20])
        self.assertTrue(pd.isna(features.loc[20, "optional metadata"]))
        self.assertEqual(len([column for column in features if column.startswith("response.state.")]), 4)

    def test_reports_pair_matrices_without_shared_samples(self) -> None:
        no_overlap = dict(self.datasets)
        no_overlap["state"] = self.datasets["state"].rename(
            columns={" sample 1 ": "other 1", "sample 2": "other 2", "sample 3": "other 3", "sample 4": "other 4"}
        )

        with self.assertRaisesRegex(ValueError, "do not share any sample columns"):
            build_pair_features(
                no_overlap,
                self.pairs,
                left_id_column="source drug",
                right_id_column="target protein",
                dataset_specs=self.specs,
                pairs=[("response", "state")],
            )


if __name__ == "__main__":
    unittest.main()
