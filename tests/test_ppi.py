from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pdm_learn.ppi import (
    canonical_pair,
    cancer_complex_pair_set,
    derive_shared_trimmed_inputs,
    parse_gene_list,
    sample_negative_control_pairs,
    sample_positive_control_pairs,
)


class PPITests(unittest.TestCase):
    def setUp(self) -> None:
        self.complexes = pd.DataFrame(
            {
                "Complex Name": ["Complex A", "Complex B"],
                "Representative Genes (Core Members)": ["['A', 'B', 'C', 'D']", ["C", "E", "F"]],
            }
        )
        self.gene_universe = ["A", "B", "C", "D", "E", "F", "G", "H"]

    def test_parse_gene_list_accepts_literal_and_iterable(self) -> None:
        self.assertEqual(parse_gene_list("['A', 'B']"), ["A", "B"])
        self.assertEqual(parse_gene_list(["A", "B"]), ["A", "B"])

    def test_canonical_pair_sorts_pair(self) -> None:
        self.assertEqual(canonical_pair("B", "A"), "A.B")

    def test_cancer_complex_pair_set_contains_within_complex_pairs(self) -> None:
        pairs = cancer_complex_pair_set(self.complexes)
        self.assertIn("A.B", pairs)
        self.assertIn("E.F", pairs)
        self.assertNotIn("A.H", pairs)

    def test_sample_positive_control_pairs_keeps_complex_labels(self) -> None:
        pairs = sample_positive_control_pairs(
            self.complexes,
            gene_universe=self.gene_universe,
            max_pairs_per_complex=1,
            rng=np.random.default_rng(0),
        )
        self.assertEqual(set(pairs.columns), {"Complex", "Gene1", "Gene2", "pair"})
        self.assertEqual(len(pairs), 2)

    def test_sample_negative_control_pairs_excludes_complex_pairs(self) -> None:
        negatives = sample_negative_control_pairs(
            self.complexes,
            gene_universe=self.gene_universe,
            count=3,
            rng=np.random.default_rng(1),
        )
        excluded = cancer_complex_pair_set(self.complexes)
        self.assertTrue(all(canonical_pair(row.Gene1, row.Gene2) not in excluded for row in negatives.itertuples()))

    def test_derive_shared_trimmed_inputs_from_trimmed_wei_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "DepMap_data"
            output = root / "DepMap_Trimmed"
            source.mkdir()

            for filename in [
                "CCLE_gene_expression_trimmed_Wei.csv",
                "shRNA_Broad_Trimmed_Wei.csv",
                "Avana_gene_effect_20Q3_Trimmed_Wei.csv",
            ]:
                pd.DataFrame(
                    {
                        "Unnamed: 0": ["A", "B", "C"],
                        "CL1": [1.0, 2.0, 10.0],
                        "CL2": [3.0, 4.0, 12.0],
                    }
                ).to_csv(source / filename, index=False)
            pd.DataFrame(
                {
                    "Unnamed: 0": ["A", "B", "D"],
                    "CL1": [1.0, 1.5, 2.0],
                    "CL2": [0.5, 2.5, 3.0],
                }
            ).to_csv(source / "CCLE_gene_cn_trimmed_Wei.csv", index=False)
            pd.DataFrame(
                {
                    "Hugo_Symbol": ["A", "B", "C"],
                    "Cell line": ["CL1", "CL2", "CL1"],
                }
            ).to_csv(source / "CCLE_gene_mutation_trimmed_Wei.csv", index=False)

            paths = derive_shared_trimmed_inputs(source, output)

            expression = pd.read_csv(paths["expression"])
            self.assertEqual(expression.iloc[:, 0].tolist(), ["A", "B"])
            self.assertEqual(expression[["CL1", "CL2"]].values.tolist(), [[-1.0, 1.0], [-1.0, 1.0]])

            copy_number = pd.read_csv(paths["copy_number"])
            self.assertEqual(copy_number[["CL1", "CL2"]].values.tolist(), [[2.0, 1.0], [3.0, 4.0]])

            mutation = pd.read_csv(paths["mutation"])
            self.assertEqual(mutation.iloc[:, 0].tolist(), ["A", "B"])
            self.assertEqual(mutation[["CL1", "CL2"]].values.tolist(), [[1.0, 0.0], [0.0, 1.0]])


if __name__ == "__main__":
    unittest.main()
