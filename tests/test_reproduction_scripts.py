import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from unittest.mock import patch

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixtures" / "reproduction"
TEST_CONFIG = FIXTURE_DIR / "workflows.json"


def _reproduction_script():
    spec = spec_from_file_location("reproduce", PROJECT_ROOT / "scripts" / "reproduce.py")
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReproductionScriptTests(unittest.TestCase):
    def test_pipeline_uses_custom_data_without_other_workflow_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory) / "custom"
            output_dir = Path(directory) / "runs"
            shutil.copytree(FIXTURE_DIR / "DepMap_Trimmed", data_dir)
            for path in data_dir.glob("*.csv"):
                table = pd.read_csv(path)
                table.iloc[:, 0] = "custom_" + table.iloc[:, 0].astype(str)
                table.to_csv(path, index=False)
            expression = Path(directory) / "my expression.csv"
            (data_dir / "Gene_Expression_Trimmed.csv").rename(expression)
            labels = Path(directory) / "positives.txt"
            labels.write_text(
                "".join(
                    f"custom_{label}\n"
                    for label in (FIXTURE_DIR / "oncogene.txt").read_text().splitlines()
                )
            )
            subprocess.run(
                [
                    sys.executable, "scripts/reproduce.py", "oncogene",
                    "--config", str(TEST_CONFIG),
                    "--data-dir", str(data_dir), "--output-dir", str(output_dir),
                    "--expression", str(expression), "--label-file", str(labels),
                ],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            ranking = pd.read_csv(output_dir / "oncogene" / "results" / "oncogene_ranking.csv")
            self.assertEqual(set(ranking.iloc[:, 0]), {f"custom_{gene}" for gene in "FGHIJ"})

    def test_simulation_does_not_require_study_inputs(self) -> None:
        script = _reproduction_script()
        with tempfile.TemporaryDirectory() as directory:
            argv = ["reproduce.py", "simulation", "--data-dir", str(Path(directory) / "missing")]
            with patch.object(sys, "argv", argv), patch.object(script, "run_simulation") as run:
                script.main()
            run.assert_called_once()

    def test_simulation_forwards_noise_sweep_from_cli(self) -> None:
        script = _reproduction_script()
        argv = ["reproduce.py", "simulation", "--epsilon-std", "0.5", "1", "2"]
        with patch.object(sys, "argv", argv), patch.object(script, "run_simulation") as run:
            script.main()
        settings = run.call_args.args[1]
        self.assertEqual(settings["epsilon_std"], [0.5, 1.0, 2.0])
        with patch.object(script, "_run") as execute:
            script.run_simulation(Path("unused"), settings)
        command = execute.call_args_list[0].args[0]
        index = command.index("--epsilon-std")
        self.assertEqual(command[index + 1:index + 4], ["0.5", "1.0", "2.0"])

    def test_ppi_uses_explicit_inputs_even_when_default_files_exist(self) -> None:
        script = _reproduction_script()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workbook = root / "cancer_complexes_clean.xlsx"
            workbook.touch()
            complexes = root / "custom complexes.csv"
            complexes.touch()
            pairs = root / "custom pairs.csv"
            expression = root / "custom expression.csv"
            with patch.object(script, "_run") as run:
                script.run_ppi(
                    root, root / "output", json.loads(TEST_CONFIG.read_text())["ppi"],
                    matrix_args=("--expression", str(expression)),
                    biogrid=pairs, complexes=complexes,
                )
            complex_paths = []
            for call in run.call_args_list:
                command = call.args[0]
                if "--complexes" in command:
                    complex_paths.append(command[command.index("--complexes") + 1])
            self.assertEqual(complex_paths, [str(complexes), str(complexes)])
            features, controls = [call.args[0] for call in run.call_args_list[:2]]
            for command in (features, controls):
                self.assertEqual(command[command.index("--expression") + 1], str(expression))
            self.assertEqual(features[features.index("--biogrid") + 1], str(pairs))
            self.assertEqual(
                controls[controls.index("--negative-pair-candidates") + 1], str(pairs)
            )
            command = run.call_args_list[-1].args[0]
            self.assertEqual(command[command.index("--max-pairs-per-complex") + 1], "10")

    def test_ppi_ranking_cap_is_independent_of_control_cap(self) -> None:
        script = _reproduction_script()
        settings = json.loads(TEST_CONFIG.read_text())["ppi"]
        settings["ranking_max_pairs_per_complex"] = 2
        settings["max_pairs_per_complex"] = 5
        with patch.object(script, "_run") as run:
            script.run_ppi(FIXTURE_DIR, Path("unused"), settings)
        controls, ranking = [call.args[0] for call in run.call_args_list[1:]]
        self.assertEqual(controls[controls.index("--max-pairs-per-complex") + 1], "5")
        self.assertEqual(ranking[ranking.index("--max-pairs-per-complex") + 1], "2")

    def test_ppi_preserves_default_complex_file_selection(self) -> None:
        script = _reproduction_script()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv = root / "cancer_complexes.csv"
            csv.touch()
            workbook = root / "cancer_complexes_clean.xlsx"
            for expected in (csv, workbook):
                expected.touch()
                with self.subTest(expected=expected.name), patch.object(script, "_run") as run:
                    script.run_ppi(
                        root, root / "output", json.loads(TEST_CONFIG.read_text())["ppi"]
                    )
                    for call in run.call_args_list[1:]:
                        command = call.args[0]
                        self.assertEqual(command[command.index("--complexes") + 1], str(expected))

    def test_pipeline_runs_all_workflows_on_sample_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "runs"
            subprocess.run(
                [
                    sys.executable,
                    "scripts/reproduce.py",
                    "all",
                    "--data-dir",
                    str(FIXTURE_DIR / "DepMap_Trimmed"),
                    "--label-file",
                    str(FIXTURE_DIR / "oncogene.txt"),
                    "--biogrid",
                    str(FIXTURE_DIR / "clean_biogrid_interactions.csv"),
                    "--complexes",
                    str(FIXTURE_DIR / "cancer_complexes.csv"),
                    "--config",
                    str(TEST_CONFIG),
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
