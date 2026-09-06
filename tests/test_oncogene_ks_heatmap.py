import tempfile
import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _heatmap_script():
    path = PROJECT_ROOT / "scripts" / "oncogene" / "make_ks_heatmap.py"
    spec = spec_from_file_location("oncogene_ks_heatmap", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OncogeneKsHeatmapTests(unittest.TestCase):
    def test_writes_png_and_pdf_for_all_pdm_feature_blocks(self) -> None:
        script = _heatmap_script()
        rows = []
        index = 0
        for left, right in script.PAIR_ORDER:
            for feature_index in range(script.AXIS_WIDTHS[left] * script.AXIS_WIDTHS[right]):
                rows.append(
                    {
                        "feature": f"{left}.{right}.{feature_index}",
                        "log10_ks_pvalue": -float(index + 1) / 10,
                    }
                )
                index += 1
        table = pd.DataFrame(rows)
        blocks = script._feature_values(table, "log10_ks_pvalue")
        self.assertEqual(len(blocks), 10)
        self.assertEqual(blocks["gene_mut.gene_exp"].shape, (2, 7))
        self.assertEqual(blocks["CRISPR.copy_num"].shape, (7, 7))

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "oncogene_ks_heatmap.png"
            pdf_output = output.with_suffix(".pdf")
            script.save_ks_heatmap(table, output, pdf_output_path=pdf_output)

            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)
            self.assertTrue(pdf_output.is_file())
            self.assertGreater(pdf_output.stat().st_size, 0)
            np.testing.assert_equal(len(table), 350)
