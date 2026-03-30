from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pdm_learn.preprocessing import build_density_map, densitymap, drop_nan, extract


class PreprocessingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.df1 = pd.DataFrame(
            {
                "gene": ["A", "B"],
                "c1": [0.0, 1.0],
                "c2": [0.5, 1.5],
                "c3": [1.0, np.nan],
            }
        )
        self.df1.name = "expr"

        self.df2 = pd.DataFrame(
            {
                "gene": ["A", "B"],
                "c1": [1.0, 0.0],
                "c2": [1.5, 0.5],
                "c3": [2.0, np.nan],
            }
        )
        self.df2.name = "crispr"

    def test_extract_returns_missing_sentinel(self) -> None:
        np.testing.assert_array_equal(extract(self.df1, "A"), np.array([0.0, 0.5, 1.0]))
        self.assertEqual(extract(self.df1, "missing"), -1)

    def test_drop_nan_removes_shared_missing_positions(self) -> None:
        x, y = drop_nan(np.array([1.0, np.nan, 3.0]), np.array([4.0, 5.0, np.nan]))
        np.testing.assert_array_equal(x, np.array([1.0]))
        np.testing.assert_array_equal(y, np.array([4.0]))

    def test_densitymap_normalizes_matrix(self) -> None:
        matrix = densitymap(
            np.array([0.0, 0.5, 1.0]),
            np.array([1.0, 1.5, 2.0]),
            [-1.0, 0.0, 1.0],
            [0.0, 1.0, 2.0],
            sigma=0.5,
        )
        self.assertAlmostEqual(float(np.sum(matrix)), 1.0, places=6)

    def test_build_density_map_supports_both_argument_orders(self) -> None:
        pairs = np.array([["A", "A"], ["B", "B"]], dtype=object)
        density_points = [[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]]
        continuous = [True, True]

        first = build_density_map([self.df1, self.df2], pairs, density_points, continuous)
        second = build_density_map([self.df1, self.df2], pairs, continuous, density_points)

        pd.testing.assert_frame_equal(first, second)


if __name__ == "__main__":
    unittest.main()
