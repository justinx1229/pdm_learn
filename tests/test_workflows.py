import sys
import tempfile
import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pdm_learn.workflows import oncogene, ppi


def _ppi_feature_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "ppi" / "make_features.py"
    spec = spec_from_file_location("ppi_make_features", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _matrices() -> dict[str, pd.DataFrame]:
    return {
        "expression": pd.DataFrame(
            {
                "entity": ["A", "B", "C"],
                "S1": [1, 2, 3],
                "S2": [2, 4, 1],
                "S3": [3, 1, 2],
                "S4": [4, 3, 4],
            }
        ),
        "copy_number": pd.DataFrame(
            {
                "entity": ["A", "B", "C"],
                "S1": [0, 1, 0],
                "S2": [1, 0, 1],
                "S3": [0, 1, 1],
                "S4": [1, 1, 0],
            }
        ),
        "shrna": pd.DataFrame(
            {
                "entity": ["A", "B", "C"],
                "S1": [4, 2, 1],
                "S2": [2, 3, 4],
                "S3": [3, 1, 2],
                "S4": [1, 4, 3],
            }
        ),
        "mutation": pd.DataFrame(
            {
                "entity": ["A", "B", "C"],
                "S1": [0, 1, 1],
                "S2": [1, 0, 1],
                "S3": [1, 1, 0],
                "S4": [0, 1, 0],
            }
        ),
        "crispr": pd.DataFrame(
            {
                "entity": ["A", "B", "C"],
                "S1": [3, 1, 4],
                "S2": [1, 4, 2],
                "S3": [4, 2, 3],
                "S4": [2, 3, 1],
            }
        ),
    }


class WorkflowTests(unittest.TestCase):
    def test_oncogene_recipe_loads_preset_paths_and_uses_generic_builder(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {}
            for name, table in _matrices().items():
                path = root / f"{name}.csv"
                table.to_csv(path, index=False)
                paths[name] = path

            datasets = oncogene.load_inputs(root, input_paths=paths)
            features = oncogene.build_features(datasets, methods=("pearson",))

        self.assertEqual(set(datasets), {"gene_exp", "copy_num", "shRNA", "gene_mut", "CRISPR"})
        self.assertEqual(features["pearson"].shape, (3, 11))

    def test_ppi_recipe_builds_pair_features_without_root_legacy_modules(self) -> None:
        datasets = {
            "gene_exp": _matrices()["expression"],
            "copy_num": _matrices()["copy_number"],
            "shRNA": _matrices()["shrna"],
            "gene_mut": _matrices()["mutation"],
            "CRISPR": _matrices()["crispr"],
        }
        pairs = pd.DataFrame({"Gene1": ["A", "B"], "Gene2": ["B", "C"]})

        features = ppi.build_features(datasets, pairs, methods=("pearson",))

        self.assertEqual(features["pearson"].shape, (2, 27))

    def test_ppi_script_symmetrizes_undirected_pairs_and_removes_duplicates(self) -> None:
        datasets = {
            "gene_exp": _matrices()["expression"],
            "copy_num": _matrices()["copy_number"],
            "shRNA": _matrices()["shrna"],
            "gene_mut": _matrices()["mutation"],
            "CRISPR": _matrices()["crispr"],
        }
        script = _ppi_feature_script()
        forward = script.build_symmetric_features(
            datasets,
            pd.DataFrame(
                {
                    "Gene1": ["A", "A"],
                    "Gene2": ["B", "B"],
                    "pair": ["A.B", "A.B"],
                    "canonical_pair": ["A.B", "A.B"],
                }
            ),
        )
        reverse = script.build_symmetric_features(
            datasets,
            pd.DataFrame(
                {
                    "Gene1": ["B"],
                    "Gene2": ["A"],
                    "pair": ["B.A"],
                    "canonical_pair": ["A.B"],
                }
            ),
        )
        columns = ppi.pdm_feature_columns(forward)

        self.assertEqual(len(forward), 1)
        np.testing.assert_allclose(forward.loc[:, columns], reverse.loc[:, columns])

    def test_ppi_negative_controls_are_sampled_only_from_supplied_candidates(self) -> None:
        complexes = pd.DataFrame(
            {
                "Complex Name": ["first"],
                "Representative Genes (Core Members)": ["['A', 'B']"],
            }
        )
        candidates = pd.DataFrame(
            {
                "Gene1": ["A", "A", "C"],
                "Gene2": ["B", "C", "D"],
            }
        )

        controls = ppi.sample_negative_control_pairs(
            complexes,
            gene_universe=["A", "B", "C", "D"],
            candidate_pairs=candidates,
            count=2,
            rng=np.random.default_rng(7),
        )

        self.assertEqual(set(controls["canonical_pair"]), {"A.C", "C.D"})


if __name__ == "__main__":
    unittest.main()
