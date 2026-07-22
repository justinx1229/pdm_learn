import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pdm_learn.features import density_map, drop_missing_observations


class DensityMapTests(unittest.TestCase):
    def test_continuous_density_map_is_normalized(self) -> None:
        density = density_map(
            [0.0, 0.5, 1.0],
            [1.0, 1.5, 2.0],
            [-1.0, 0.0, 1.0],
            [0.0, 1.0, 2.0],
            sigma=0.5,
        )

        self.assertEqual(density.shape, (3, 3))
        self.assertAlmostEqual(float(density.sum()), 1.0, places=6)

    def test_discrete_density_map_counts_observations(self) -> None:
        density = density_map(
            [0, 1, 1, 0],
            [1, 1, 0, 0],
            [0, 1],
            [0, 1],
            x_discrete=True,
            y_discrete=True,
        )

        np.testing.assert_allclose(density, [[0.25, 0.25], [0.25, 0.25]])

    def test_missing_observations_are_removed_in_pairs(self) -> None:
        x, y = drop_missing_observations([1.0, np.nan, 3.0], [4.0, 5.0, np.nan])
        np.testing.assert_array_equal(x, [1.0])
        np.testing.assert_array_equal(y, [4.0])


if __name__ == "__main__":
    unittest.main()
