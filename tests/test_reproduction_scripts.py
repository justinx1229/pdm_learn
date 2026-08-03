import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ReproductionScriptTests(unittest.TestCase):
    def test_smoke_profile_runs_all_workflows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "runs"
            subprocess.run(
                [
                    sys.executable,
                    "scripts/reproduce.py",
                    "all",
                    "--profile",
                    "smoke",
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=PROJECT_ROOT,
                check=True,
            )
            expected = [
                output_dir / "oncogene" / "results" / "oncogene_ranking.csv",
                output_dir / "ppi" / "results" / "ranked_PPI_pairs_cancer.csv",
                output_dir / "ppi" / "results" / "ppi_control_benchmarks.csv",
                output_dir / "simulation" / "results" / "simulation_benchmarks.csv",
            ]
            self.assertTrue(all(path.is_file() for path in expected))


if __name__ == "__main__":
    unittest.main()
