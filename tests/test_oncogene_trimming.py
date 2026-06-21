from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pdm_learn.oncogene import (
    ONCOGENE_TRIMMED_FILENAMES,
    SHARED_TRIMMED_FILENAMES,
    load_oncogene_inputs,
    standardize_oncogene_matrix,
    standardize_oncogene_mutations,
    trim_oncogene_input_tables,
    _mutation_to_reference,
)


class OncogeneTrimmingTests(unittest.TestCase):
    def test_standardize_genes_as_rows_matrix(self) -> None:
        raw = pd.DataFrame(
            {
                "Gene": ["TP53 (7157)", "KRAS"],
                "CL1": [1.0, 2.0],
                "CL2": [3.0, 4.0],
            }
        )
        output = standardize_oncogene_matrix(raw, orientation="genes-as-rows")
        self.assertEqual(output.columns.tolist(), ["gene name", "CL1", "CL2"])
        self.assertEqual(output["gene name"].tolist(), ["KRAS", "TP53"])

    def test_standardize_samples_as_rows_matrix(self) -> None:
        raw = pd.DataFrame(
            {
                "Cell line": ["CL1", "CL2"],
                "TP53 (7157)": [1.0, 3.0],
                "KRAS": [2.0, 4.0],
            }
        )
        output = standardize_oncogene_matrix(raw, orientation="samples-as-rows")
        self.assertEqual(output.columns.tolist(), ["gene name", "CL1", "CL2"])
        self.assertEqual(output["gene name"].tolist(), ["KRAS", "TP53"])

    def test_standardize_mutations(self) -> None:
        raw = pd.DataFrame({"Gene": ["TP53 (7157)", "KRAS"], "Cell line": ["CL1", "CL2"]})
        output = standardize_oncogene_mutations(raw)
        self.assertEqual(output[["Hugo_Symbol", "Cell line"]].values.tolist(), [["TP53", "CL1"], ["KRAS", "CL2"]])

    def test_trim_oncogene_input_tables_writes_expected_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_paths = {}
            for name in ("expression", "copy_number", "shrna", "crispr"):
                path = root / f"{name}.csv"
                pd.DataFrame({"Gene": ["TP53", "KRAS"], "CL1": [1.0, 2.0], "CL2": [3.0, 4.0]}).to_csv(path, index=False)
                raw_paths[name] = path
            mutation_path = root / "mutation.csv"
            pd.DataFrame({"Gene": ["TP53", "BRAF"], "Cell line": ["CL1", "CL3"]}).to_csv(mutation_path, index=False)
            raw_paths["mutation"] = mutation_path

            output_paths = trim_oncogene_input_tables(raw_paths, root / "out", gene_list=["TP53", "KRAS"])

            self.assertEqual(set(output_paths), set(ONCOGENE_TRIMMED_FILENAMES))
            for path in output_paths.values():
                self.assertTrue(path.exists())
            mutation = pd.read_csv(output_paths["mutation"])
            self.assertEqual(mutation[["Hugo_Symbol", "Cell line"]].values.tolist(), [["TP53", "CL1"]])

    def test_load_oncogene_inputs_reads_shared_trimmed_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            trimmed = root / "DepMap_Trimmed"
            trimmed.mkdir()
            for key in ("expression", "copy_number", "shrna", "crispr"):
                pd.DataFrame({"gene name": ["A", "B"], "CL1": [1.0, 2.0], "CL2": [3.0, 4.0]}).to_csv(
                    trimmed / SHARED_TRIMMED_FILENAMES[key],
                    index=False,
                )
            pd.DataFrame({"gene name": ["A", "B"], "CL1": [1.0, 0.0], "CL2": [0.0, 1.0]}).to_csv(
                trimmed / SHARED_TRIMMED_FILENAMES["mutation"],
                index=False,
            )

            datasets = load_oncogene_inputs(root)

            self.assertEqual(set(datasets), {"gene_exp", "copy_num", "shRNA", "gene_mut", "CRISPR"})
            self.assertEqual(datasets["gene_mut"].shape, (2, 3))

    def test_mutation_to_reference_accepts_binary_matrix(self) -> None:
        mutation = pd.DataFrame({"gene name": ["A", "B"], "CL1": [1, 0], "CL2": [0, 1], "CL3": [1, 1]})
        reference = pd.DataFrame({"gene name": ["A", "B", "C"], "CL2": [0.0, 0.0, 0.0], "CL1": [0.0, 0.0, 0.0]})

        matrix, _ = _mutation_to_reference(mutation, reference)

        self.assertEqual(matrix.columns.tolist(), ["gene name", "CL2", "CL1"])
        self.assertEqual(matrix.values.tolist(), [["A", 0.0, 1.0], ["B", 1.0, 0.0], ["C", 0.0, 0.0]])

    def test_mutation_to_reference_rejects_long_mutation_table(self) -> None:
        mutation = pd.DataFrame({"Hugo_Symbol": ["A"], "Cell line": ["CL1"]})
        reference = pd.DataFrame({"gene name": ["A"], "CL1": [0.0]})

        with self.assertRaises(ValueError):
            _mutation_to_reference(mutation, reference)


if __name__ == "__main__":
    unittest.main()
