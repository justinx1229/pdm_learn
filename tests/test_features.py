from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pdm_learn
from pdm_learn.features import (
    DatasetPair,
    DatasetSpec,
    build_feature_tables,
    load_gene_matrices,
    rank_labeled_genes,
    split_labeled_feature_tables,
)


class PublicFeatureApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.datasets = {
            "expression": pd.DataFrame(
                {"gene": ["A", "B", "C"], "S1": [0.0, 1.0, 2.0], "S2": [1.0, 2.0, 3.0], "S3": [2.0, 3.0, 4.0]}
            ),
            "state": pd.DataFrame(
                {"symbol": ["A", "B", "C"], "S1": [0, 1, 0], "S2": [1, 1, 0], "S3": [0, 0, 1]}
            ),
        }

    def test_root_exports_documented_feature_api(self) -> None:
        self.assertEqual(pdm_learn.__version__, "0.3.0")
        self.assertIs(pdm_learn.DatasetSpec, DatasetSpec)
        self.assertIs(pdm_learn.build_feature_tables, build_feature_tables)

    def test_load_gene_matrices_normalizes_names(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "expression.csv"
            pd.DataFrame({"gene": [" A "], "S1": [1.0]}).to_csv(path, index=False)

            datasets = load_gene_matrices({" expression ": path})

        self.assertEqual(set(datasets), {"expression"})
        self.assertEqual(datasets["expression"].iloc[0, 0], "A")

    def test_build_and_split_feature_tables(self) -> None:
        tables = build_feature_tables(
            self.datasets,
            pairs=[DatasetPair("expression", "state")],
            dataset_specs={"state": DatasetSpec("state", "binary", (0.0, 1.0), normalize=False)},
            methods=("pearson", "mi"),
            boxes=3,
        )

        self.assertEqual(set(tables), {"pdm", "pearson", "mi"})
        self.assertEqual(tables["pdm"].shape, (3, 7))
        split = split_labeled_feature_tables(tables, {"A"})
        self.assertEqual(split["pdm"][0].iloc[:, 0].tolist(), ["A"])
        self.assertEqual(split["pdm"][1].iloc[:, 0].tolist(), ["B", "C"])

    def test_rank_labeled_genes_uses_first_column_as_identifier(self) -> None:
        table = pd.DataFrame(
            {
                "gene": ["A", "B", "C", "D", "E", "F"],
                "feature": [1.0, 1.1, 0.9, 0.0, 0.1, 0.2],
            }
        )

        ranking = rank_labeled_genes(table, {"A", "B", "C"}, trials=2, model="LR")

        self.assertEqual(set(ranking["gene"]), {"D", "E", "F"})
        self.assertTrue(ranking["score"].between(0, 1).all())


if __name__ == "__main__":
    unittest.main()
