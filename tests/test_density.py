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

    def test_underflow_assigns_mass_to_nearest_center(self) -> None:
        cases = [
            (1000, 1000, False, False, (2, 2)),
            (-1000, 1000, False, False, (2, 0)),
            (1, 1000, True, False, (2, 1)),
            (1, -1000, True, False, (0, 1)),
            (1000, 1, False, True, (1, 2)),
            (-1000, 1, False, True, (1, 0)),
        ]
        for x, y, x_discrete, y_discrete, cell in cases:
            with self.subTest(x=x, y=y, x_discrete=x_discrete, y_discrete=y_discrete):
                density = density_map(
                    [x], [y], [0, 1, 2], [0, 1, 2],
                    x_discrete=x_discrete, y_discrete=y_discrete, sigma=0.1,
                )
                expected = np.zeros((3, 3))
                expected[cell] = 1.0
                np.testing.assert_array_equal(density, expected)
                self.assertEqual(float(density.sum()), 1.0)

    def test_continuous_density_preserves_gaussian_weights(self) -> None:
        density = density_map([0], [0], [-1, 0, 1], [-1, 0, 1], sigma=1.0)
        weights = np.array([np.exp(-0.5), 1.0, np.exp(-0.5)])
        expected = np.outer(weights, weights) / weights.sum() ** 2
        np.testing.assert_allclose(density, expected)

    def test_missing_observations_are_removed_in_pairs(self) -> None:
        x, y = drop_missing_observations([1.0, np.nan, 3.0], [4.0, 5.0, np.nan])
        np.testing.assert_array_equal(x, [1.0])
        np.testing.assert_array_equal(y, [4.0])


if __name__ == "__main__":
    unittest.main()
