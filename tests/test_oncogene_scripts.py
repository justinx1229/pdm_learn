"""Exercise the prediction / benchmark / plotting boundaries on small fixtures."""

import subprocess
import sys
from argparse import Namespace
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

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
        "test_cv_predictions.csv",
        "test_pr_curve.csv",
        "test_roc_curve.csv",
    }
    assert all((results / name).read_bytes() == content for name, content in predictions.items())
    benchmark = pd.read_csv(results / "test_method_benchmarks.csv")
    assert set(benchmark["method"]) == {"PDM", "Pearson"}
    assert benchmark[["roc_auc", "average_precision"]].notna().all().all()
    predictions_table = pd.read_csv(results / "test_cv_predictions.csv")
    for method, predicted in predictions_table.groupby("method"):
        summary = benchmark.set_index("method").loc[method]
        assert predicted.entity.is_unique
        assert set(predicted.entity) == set("ABCDEFGHIJ")
        assert summary.average_precision == pytest.approx(
            average_precision_score(predicted.target, predicted.score)
        )
        assert summary.roc_auc == pytest.approx(roc_auc_score(predicted.target, predicted.score))
    ranks = pd.read_csv(results / "test_loocv_held_out_ranks.csv")
    assert len(ranks) == 10  # Five held-out positives for each of two methods.

    before_plot = {path.name: path.read_bytes() for path in results.iterdir()}
    run_script("plot_benchmarks.py", "--results-dir", results, "--label-name", "test")
    assert all((results / name).read_bytes() == content for name, content in before_plot.items())
    plots = results / "figures"
    assert {path.name for path in plots.iterdir()} == {
        f"test_{name}.{extension}"
        for name in ("ks_heatmap", "loocv", "method_benchmarks", "pr", "roc")
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


def benchmark_script():
    spec = spec_from_file_location(
        "run_benchmarks", PROJECT_ROOT / "scripts/oncogene/run_benchmarks.py"
    )
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_common_genes_are_finite_unique_and_aligned():
    script = benchmark_script()
    first = pd.DataFrame({"gene": [" C ", "B", "A", "D"], "x": [3, np.inf, 1, 4]})
    second = pd.DataFrame({"gene": ["A", "C", "B"], "y": [2, 6, 4]})
    tables = script._common_tables({"first": first, "second": second})
    assert tables["first"].gene.tolist() == tables["second"].gene.tolist() == ["A", "C"]
    assert tables["first"].x.tolist() == [1, 3]
    with pytest.raises(ValueError, match="duplicate"):
        script._common_tables({"first": pd.concat([first, first.iloc[[0]]])})


def test_ablation_removes_all_incident_pairs_and_preserves_genes(monkeypatch):
    script = benchmark_script()
    from itertools import combinations

    pairs = list(combinations(script.ASSAYS, 2))
    table = pd.DataFrame({f"{left}.{right}.0": [1., 2., 3.] for left, right in pairs})
    table.insert(0, "gene", ["A", "B", "C"])
    calls = []

    def recovery(subset, labels, args, model):
        calls.append(subset)
        return (pd.DataFrame({"loocv_area": [0.8], "unlabeled_candidate_rows": [1]}),
                pd.DataFrame({"gene": ["A", "B"], "rank": [0, 1]}))

    monkeypatch.setattr(script, "_pdm_recovery", recovery)
    result = script.run_ablation({"PDM": table}, {"A", "B"}, Namespace(model="svm"))
    summary = result["ablation"]
    assert len(summary) == len(calls) == 16
    for (_, row), subset in zip(summary.iterrows(), calls, strict=True):
        used = {assay for assay in script.ASSAYS if row[assay]}
        expected = [f"{left}.{right}.0" for left, right in pairs if {left, right} <= used]
        assert subset.columns.tolist() == ["gene", *expected]
        assert subset.gene.tolist() == table.gene.tolist()
        assert row.feature_columns == len(expected)


def test_pr_without_negatives_is_marked_as_unlabeled(tmp_path):
    features = tmp_path / "features"
    features.mkdir()
    pd.DataFrame({"gene": list("ABCDEFGH"), "signal": range(8)}).to_csv(
        features / "pearson.csv", index=False
    )
    run_script("run_benchmarks.py", "--feature-dir", features,
               "--results-dir", tmp_path / "results", "--benchmarks", "pr",
               "--label-file", FIXTURES / "oncogene.txt", "--model", "logistic_regression",
               "--n-splits", 2)
    predictions = pd.read_csv(tmp_path / "results/oncogene_cv_predictions.csv")
    assert set(predictions.comparison) == {"unlabeled_candidates"}
    assert predictions.target.sum() == 5
    assert set(predictions.fold) == {0, 1}
    assert set(predictions.entity) == set("ABCDEFGH")
