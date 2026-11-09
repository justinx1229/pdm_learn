import json
import subprocess
import sys
import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pdm_learn.simulation import (
    build_density_dataset,
    build_metric_dataset,
    iter_simulated_pairs,
    perturb_pair,
)


def _simulation_feature_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "simulation" / "make_features.py"
    spec = spec_from_file_location("simulation_make_features", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SimulationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.seed_pairs = pd.DataFrame(
            [
                [0.0, 0.5, 1.0, 1.5],
                [1.0, 1.5, 2.0, 2.5],
                [0.2, 0.4, 0.6, 0.8],
                [0.8, 0.6, 0.4, 0.2],
            ]
        )

    def test_perturb_pair_standardizes_and_clips(self) -> None:
        x, y = perturb_pair(
            np.array([0.0, 1.0, 2.0]),
            np.array([2.0, 3.0, 4.0]),
            noise_scale=1.0,
            centers=[-1.0, 0.0, 1.0],
            rng=np.random.default_rng(0),
        )
        self.assertLessEqual(np.max(x), 1.0)
        self.assertGreaterEqual(np.min(y), -1.0)

    def test_metric_and_density_builders_return_expected_shapes(self) -> None:
        pairs = list(
            iter_simulated_pairs(
                self.seed_pairs,
                repeats=2,
                noise_scale=0.25,
                rng=np.random.default_rng(1),
            )
        )
        metric = build_metric_dataset(
            self.seed_pairs,
            lambda x, y: float(np.corrcoef(x, y)[0, 1]),
            repeats=2,
            noise_scale=0.25,
            simulated_pairs=pairs,
        )
        density = build_density_dataset(
            self.seed_pairs,
            centers=np.linspace(-2, 2, 5),
            repeats=2,
            noise_scale=0.25,
            sigma=0.5,
            simulated_pairs=pairs,
        )

        self.assertEqual(metric.shape, (4, 1))
        self.assertEqual(density.shape, (4, 25))
        self.assertTrue(np.isfinite(density.to_numpy()).all())

    def test_simulation_seed_pairs_retain_standardized_benchmark_design(self) -> None:
        seed_pairs = _simulation_feature_script().build_positive_base()

        self.assertEqual(seed_pairs.shape, (8, 500))
        self.assertTrue(np.allclose(seed_pairs.mean(axis=1), 0.0))
        self.assertTrue(np.allclose(seed_pairs.std(axis=1, ddof=0), 1.0))


def run_simulation_script(name, *args, check=True):
    root = Path(__file__).resolve().parents[1]
    return subprocess.run(
        [sys.executable, str(root / "scripts/simulation" / name), *map(str, args)],
        cwd=root, capture_output=True, text=True, check=check,
    )


def test_simulation_exports_matched_predictions_and_plots_without_refitting(tmp_path):
    features, results = tmp_path / "features", tmp_path / "results"
    run_simulation_script("make_features.py", "--output-dir", features,
                          "--positive-repeats", 1, "--negative-repeats", 3)
    run_simulation_script(
        "run_benchmarks.py", "--feature-dir", features, "--results-dir", results,
        "--model", "logistic_regression", "--n-splits", 2, "--models",
        "logistic_regression", "random_forest", "gradient_boosting", "svm",
    )
    assert {path.name for path in results.iterdir()} == {
        f"simulation_{prefix}{suffix}.csv" for prefix in ("", "model_")
        for suffix in ("benchmarks", "cv_predictions", "roc_curve", "pr_curve")
    }
    for prefix, group in (("", "method"), ("model_", "model")):
        summary = pd.read_csv(results / f"simulation_{prefix}benchmarks.csv").set_index(group)
        predicted = pd.read_csv(results / f"simulation_{prefix}cv_predictions.csv")
        assert not predicted.duplicated([group, "entity"]).any()
        assert predicted.groupby("entity").fold.nunique().eq(1).all()
        assert predicted.groupby(group).size().eq(16).all()
        assert len(summary) == (5 if group == "method" else 4)
        assert summary.positive_rows.eq(4).all() and summary.negative_rows.eq(12).all()
        for name, rows in predicted.groupby(group):
            assert summary.loc[name, "roc_auc"] == pytest.approx(
                roc_auc_score(rows.target, rows.score)
            )
            assert summary.loc[name, "average_precision"] == pytest.approx(
                average_precision_score(rows.target, rows.score)
            )
    methods = pd.read_csv(results / "simulation_cv_predictions.csv")
    models = pd.read_csv(results / "simulation_model_cv_predictions.csv")
    pd.testing.assert_frame_equal(
        methods.loc[methods.method == "PDM"].reset_index(drop=True),
        models.loc[models.model == "logistic_regression"].reset_index(drop=True),
        check_dtype=False,  # Scalar methods leave n_features blank in the combined CSV.
    )
    before = {path.name: path.read_bytes() for path in results.iterdir()}
    run_simulation_script("plot_benchmarks.py", "--results-dir", results)
    assert all((results / name).read_bytes() == content for name, content in before.items())
    assert {path.name for path in (results / "figures").iterdir()} == {
        f"simulation_{prefix}{suffix}.{extension}" for prefix in ("", "model_")
        for suffix in ("benchmarks", "roc", "pr") for extension in ("png", "pdf")
    }
    assert all(path.stat().st_size > 0 for path in (results / "figures").iterdir())


def test_simulation_methods_share_complete_rows_and_reject_mismatched_counts(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts/simulation/run_benchmarks.py"
    spec = spec_from_file_location("simulation_run_benchmarks", path)
    script = module_from_spec(spec)
    spec.loader.exec_module(script)
    for label in ("positive", "negative"):
        directory = tmp_path / label
        directory.mkdir()
        for method in ("pdm", "pearson"):
            pd.DataFrame({"signal": [1., 2., 3.]}).to_csv(directory / f"{method}.csv", index=False)
    pd.DataFrame({"signal": [np.inf, 2., 3.]}).to_csv(
        tmp_path / "positive/pearson.csv", index=False
    )
    tables, labels = script.load_tables(tmp_path, ["pdm", "pearson"])
    assert labels == {"positive_0", "positive_1", "positive_2"}
    assert tables["pdm"].simulation_id.tolist() == tables["pearson"].simulation_id.tolist()
    assert len(tables["pdm"]) == 5
    assert "positive_0" not in set(tables["pdm"].simulation_id)
    pd.DataFrame({"signal": [1., 2.]}).to_csv(tmp_path / "negative/pearson.csv", index=False)
    with pytest.raises(ValueError, match="same positive and negative rows"):
        script.load_tables(tmp_path, ["pdm", "pearson"])


def test_simulation_explicit_plot_requires_saved_input(tmp_path):
    result = run_simulation_script("plot_benchmarks.py", "--results-dir", tmp_path,
                                   "--plots", "pr", check=False)
    assert result.returncode != 0
    assert "simulation_pr_curve.csv" in result.stderr
    assert not (tmp_path / "figures").exists()


def test_simulation_noise_sweep_keeps_levels_separate_and_folds_matched(tmp_path):
    features, results = tmp_path / "features", tmp_path / "results"
    generation = ("--positive-repeats", 2, "--negative-repeats", 2, "--seed", 7)
    run_simulation_script("make_features.py", "--output-dir", features,
                          "--epsilon-std", 2, 0.5, 0, 2, *generation)
    manifest = json.loads((features / "simulation_manifest.json").read_text())
    assert manifest["epsilon_stds"] == [0.0, 0.5, 2.0]
    # A level generated in a sweep is identical to generating it alone.
    single = tmp_path / "single"
    run_simulation_script("make_features.py", "--output-dir", single,
                          "--epsilon-std", 2, *generation)
    for path in (features / "epsilon_2.0").rglob("*.csv"):
        counterpart = single / path.relative_to(features / "epsilon_2.0")
        pd.testing.assert_frame_equal(pd.read_csv(path), pd.read_csv(counterpart))
    # An incomplete row at one level must be excluded at every level and for every method.
    path = features / "epsilon_0.5/positive/pdm.csv"
    incomplete = pd.read_csv(path)
    incomplete.iloc[0, 0] = np.nan
    incomplete.to_csv(path, index=False)
    run_simulation_script("run_benchmarks.py", "--feature-dir", features,
                          "--results-dir", results, "--n-splits", 2,
                          "--model", "logistic_regression", "--models",
                          "logistic_regression", "gradient_boosting")
    for prefix, group in (("", "method"), ("model_", "model")):
        summary = pd.read_csv(results / f"simulation_{prefix}benchmarks.csv")
        predictions = pd.read_csv(results / f"simulation_{prefix}cv_predictions.csv")
        assert set(summary.epsilon_std) == {0, 0.5, 2}
        assert len(summary) == (15 if group == "method" else 6)
        assert predictions.groupby("entity").fold.nunique().eq(1).all()
        assert not predictions.duplicated(["epsilon_std", group, "entity"]).any()
        summary = summary.set_index(["epsilon_std", group])
        for key, rows in predictions.groupby(["epsilon_std", group]):
            assert len(rows) == 23
            assert "positive_0" not in set(rows.entity)
            assert summary.loc[key, "roc_auc"] == pytest.approx(
                roc_auc_score(rows.target, rows.score)
            )
            assert summary.loc[key, "average_precision"] == pytest.approx(
                average_precision_score(rows.target, rows.score)
            )
        for suffix in ("pr_curve", "roc_curve"):
            curves = pd.read_csv(results / f"simulation_{prefix}{suffix}.csv")
            assert set(curves.epsilon_std) == {0, 0.5, 2}
    before = {path.name: path.read_bytes() for path in results.iterdir()}
    run_simulation_script("plot_benchmarks.py", "--results-dir", results)
    assert len(list((results / "figures").glob("*.png"))) == 6
    assert len(list((results / "figures").glob("*.pdf"))) == 6
    assert all((results / name).read_bytes() == value for name, value in before.items())


@pytest.mark.parametrize("epsilon", ["-1", "nan", "inf"])
def test_simulation_rejects_invalid_noise_levels(tmp_path, epsilon):
    output = tmp_path / "features"
    result = run_simulation_script("make_features.py", "--output-dir", output,
                                   "--epsilon-std", epsilon, check=False)
    assert result.returncode != 0
    assert "finite and non-negative" in result.stderr
    assert not output.exists()


if __name__ == "__main__":
    unittest.main()
