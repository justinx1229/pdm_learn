"""Exercise the prediction / benchmark / plotting boundaries on small fixtures."""

import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = PROJECT_ROOT / "tests" / "fixtures" / "reproduction"


def run_script(name, *args, check=True):
    return subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "oncogene" / name), *map(str, args)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=check,
    )


def test_oncogene_stages_write_only_their_own_outputs(tmp_path):
    features, results = tmp_path / "features", tmp_path / "results"
    run_script(
        "make_features.py",
        "--data-dir",
        FIXTURES / "DepMap_Trimmed",
        "--output-dir",
        features,
        "--methods",
        "pearson",
    )
    common = (
        "--feature-dir",
        features,
        "--results-dir",
        results,
        "--label-file",
        FIXTURES / "oncogene.txt",
        "--label-name",
        "test",
        "--model",
        "logistic_regression",
    )
    run_script("run_predictions.py", *common, "--ranking-trials", 25)
    predictions = {path.name: path.read_bytes() for path in results.iterdir()}
    assert set(predictions) == {"test_ranking.csv"}
    assert set(pd.read_csv(results / "test_ranking.csv").iloc[:, 0]) == set("FGHIJ")

    negatives = tmp_path / "negatives.txt"
    negatives.write_text("F\nG\nH\nI\nJ\n")
    run_script(
        "run_benchmarks.py",
        *common,
        "--benchmarks",
        "ks",
        "loocv",
        "labeled",
        "--methods",
        "PDM,Pearson",
        "--loocv-trials",
        50,
        "--negative-label-file",
        negatives,
        "--n-splits",
        2,
    )
    assert {path.name for path in results.iterdir()} == {
        *predictions,
        "test_ks_pvalues.csv",
        "test_loocv_curve.csv",
        "test_loocv_held_out_ranks.csv",
        "test_method_benchmarks.csv",
    }
    assert all((results / name).read_bytes() == content for name, content in predictions.items())
    benchmark = pd.read_csv(results / "test_method_benchmarks.csv")
    assert set(benchmark["method"]) == {"PDM", "Pearson"}
    assert benchmark[["roc_auc", "average_precision"]].notna().all().all()
    ranks = pd.read_csv(results / "test_loocv_held_out_ranks.csv")
    assert len(ranks) == 10  # Five held-out positives for each of two methods.

    before_plot = {path.name: path.read_bytes() for path in results.iterdir()}
    run_script("plot_benchmarks.py", "--results-dir", results, "--label-name", "test")
    assert all((results / name).read_bytes() == content for name, content in before_plot.items())
    plots = results / "figures"
    assert {path.name for path in plots.iterdir()} == {
        f"test_{name}.{extension}"
        for name in ("ks_heatmap", "loocv", "method_benchmarks")
        for extension in ("png", "pdf")
    }
    assert all(path.stat().st_size > 0 for path in plots.iterdir())


def test_benchmark_only_flags_are_rejected_by_predictions():
    result = run_script("run_predictions.py", "--loocv", check=False)
    assert result.returncode != 0
    assert "unrecognized arguments: --loocv" in result.stderr


def test_labeled_benchmark_requires_explicit_negatives(tmp_path):
    result = run_script(
        "run_benchmarks.py",
        "--benchmarks",
        "labeled",
        "--results-dir",
        tmp_path / "results",
        check=False,
    )
    assert result.returncode != 0
    assert "requires --negative-label-file" in result.stderr
    assert not (tmp_path / "results").exists()


def test_plot_selection_requires_its_saved_input(tmp_path):
    result = run_script(
        "plot_benchmarks.py", "--results-dir", tmp_path, "--plots", "loocv", check=False
    )
    assert result.returncode != 0
    assert "oncogene_loocv_curve.csv" in result.stderr
    assert not (tmp_path / "figures").exists()
