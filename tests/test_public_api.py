import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pdm_learn


class PublicApiTests(unittest.TestCase):
    def test_root_exports_only_generic_library_objects(self) -> None:
        self.assertEqual(pdm_learn.__version__, "1.0.0")
        self.assertIn("DatasetSpec", pdm_learn.__all__)
        self.assertIn("build_features", pdm_learn.__all__)
        self.assertIn("build_pair_features", pdm_learn.__all__)
        self.assertNotIn("oncogene", pdm_learn.__all__)
        self.assertNotIn("ppi", pdm_learn.__all__)
        self.assertFalse(hasattr(pdm_learn, "build_oncogene_density_features"))


if __name__ == "__main__":
    unittest.main()
