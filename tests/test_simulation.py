from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pdm_learn.preprocessing import densitymap
from pdm_learn.simulation import (
    build_heatmap_dataset,
    build_metric_dataset,
    iter_simulated_pairs,
    perturb_pair,
)


class SimulationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.positive = pd.DataFrame(
            [
                [0.0, 0.5, 1.0, 1.5],
                [1.0, 1.5, 2.0, 2.5],
                [0.2, 0.4, 0.6, 0.8],
                [0.8, 0.6, 0.4, 0.2],
            ]
        )

    def test_perturb_pair_standardizes_and_clips(self) -> None:
        rng = np.random.default_rng(0)
        x, y = perturb_pair(
            np.array([0.0, 1.0, 2.0]),
            np.array([2.0, 3.0, 4.0]),
            epsilon_std=1.0,
            centers=[-1.0, 0.0, 1.0],
            rng=rng,
        )
        self.assertLessEqual(np.max(x), 1.0)
        self.assertGreaterEqual(np.min(x), -1.0)
        self.assertLessEqual(np.max(y), 1.0)
        self.assertGreaterEqual(np.min(y), -1.0)

    def test_iter_simulated_pairs_respects_repeat_count(self) -> None:
        pairs = list(
            iter_simulated_pairs(
                self.positive,
                repeats=3,
                epsilon_std=0.25,
                rng=np.random.default_rng(1),
            )
        )
        self.assertEqual(len(pairs), 6)

    def test_build_metric_dataset_returns_expected_rows(self) -> None:
        output = build_metric_dataset(
            self.positive,
            lambda x, y: float(np.corrcoef(x, y)[0, 1]),
            repeats=2,
            epsilon_std=0.5,
            shuffle_y=True,
            rng=np.random.default_rng(2),
        )
        self.assertEqual(output.shape, (4, 1))

    def test_build_heatmap_dataset_returns_logged_features(self) -> None:
        output = build_heatmap_dataset(
            self.positive,
            densitymap,
            centers=np.linspace(-2, 2, 5),
            repeats=1,
            epsilon_std=0.5,
            sigma=0.1,
            rng=np.random.default_rng(3),
        )
        self.assertEqual(output.shape[0], 2)
        self.assertTrue(np.isfinite(output.to_numpy()).all())


if __name__ == "__main__":
    unittest.main()
