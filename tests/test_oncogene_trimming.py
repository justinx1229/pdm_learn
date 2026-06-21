from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pdm_learn.oncogene import (
    ONCOGENE_TRIMMED_FILENAMES,
    standardize_oncogene_matrix,
    standardize_oncogene_mutations,
    trim_oncogene_input_tables,
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


if __name__ == "__main__":
    unittest.main()
