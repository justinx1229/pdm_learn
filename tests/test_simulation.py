import sys
import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pdm_learn.simulation import (
    build_density_dataset,
    build_metric_dataset,
    iter_simulated_pairs,
    perturb_pair,
)


def _simulation_feature_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "simulation" / "make_features.py"
    spec = spec_from_file_location("simulation_make_features", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SimulationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.seed_pairs = pd.DataFrame(
            [
                [0.0, 0.5, 1.0, 1.5],
                [1.0, 1.5, 2.0, 2.5],
                [0.2, 0.4, 0.6, 0.8],
                [0.8, 0.6, 0.4, 0.2],
            ]
        )

    def test_perturb_pair_standardizes_and_clips(self) -> None:
        x, y = perturb_pair(
            np.array([0.0, 1.0, 2.0]),
            np.array([2.0, 3.0, 4.0]),
            noise_scale=1.0,
            centers=[-1.0, 0.0, 1.0],
            rng=np.random.default_rng(0),
        )
        self.assertLessEqual(np.max(x), 1.0)
        self.assertGreaterEqual(np.min(y), -1.0)

    def test_metric_and_density_builders_return_expected_shapes(self) -> None:
        pairs = list(
            iter_simulated_pairs(
                self.seed_pairs,
                repeats=2,
                noise_scale=0.25,
                rng=np.random.default_rng(1),
            )
        )
        metric = build_metric_dataset(
            self.seed_pairs,
            lambda x, y: float(np.corrcoef(x, y)[0, 1]),
            repeats=2,
            noise_scale=0.25,
            simulated_pairs=pairs,
        )
        density = build_density_dataset(
            self.seed_pairs,
            centers=np.linspace(-2, 2, 5),
            repeats=2,
            noise_scale=0.25,
            sigma=0.5,
            simulated_pairs=pairs,
        )

        self.assertEqual(metric.shape, (4, 1))
        self.assertEqual(density.shape, (4, 25))
        self.assertTrue(np.isfinite(density.to_numpy()).all())

    def test_simulation_seed_pairs_retain_standardized_benchmark_design(self) -> None:
        seed_pairs = _simulation_feature_script().build_positive_base()

        self.assertEqual(seed_pairs.shape, (8, 500))
        self.assertTrue(np.allclose(seed_pairs.mean(axis=1), 0.0))
        self.assertTrue(np.allclose(seed_pairs.std(axis=1, ddof=0), 1.0))


if __name__ == "__main__":
    unittest.main()
