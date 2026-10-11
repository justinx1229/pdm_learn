"""Rank and cross-validate oncogene feature tables produced by ``make_features``."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pdm_learn.labels import load_label_set
from pdm_learn.modeling import (
    MODEL_NAMES,
    evaluate_labeled_features,
    feature_ks_pvalues,
    loocv_rank_recovery,
    rank_candidates,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank and benchmark gene-level feature tables.")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument(
        "--feature-dir", type=Path, default=PROJECT_ROOT / "runs" / "oncogene" / "features"
    )
    parser.add_argument(
        "--results-dir", type=Path, default=PROJECT_ROOT / "runs" / "oncogene" / "results"
    )
    parser.add_argument("--density-name", default="dataset_trimmed_package.csv")
    parser.add_argument("--model", choices=MODEL_NAMES, default="gradient_boosting")
    parser.add_argument("--ranking-trials", type=int, default=100)
    parser.add_argument(
        "--loocv",
        action="store_true",
        help="Write leave-one-positive-out rank-recovery curves and a PNG chart.",
    )
    parser.add_argument(
        "--loocv-trials",
        type=int,
        default=25,
        help="Positive-unlabeled ranking trials for each held-out positive.",
    )
    parser.add_argument(
        "--loocv-methods",
        default="all",
        help="Comma-separated feature tables for LOOCV, or 'all'.",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--features-left", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--label-file", type=Path, default=None, help="One-column label list.")
    parser.add_argument(
        "--negative-label-file",
        type=Path,
        default=None,
        help="One-column file of explicit negative labels required for ROC AUC/AP benchmarking.",
    )
    parser.add_argument("--label-name", default="oncogene", help="Prefix for output filenames.")
    parser.add_argument(
        "--positive-extra",
        action="append",
        default=[],
        help="Additional positive label to include. Repeat for multiple genes.",
    )
    return parser.parse_args()


def _load_feature_tables(feature_dir: Path, density_name: str) -> dict[str, pd.DataFrame]:
    paths = {
        "PDM": feature_dir / density_name,
        "Pearson": feature_dir / "pearson.csv",
        "Spearman": feature_dir / "spearman.csv",
        "MutualInfo": feature_dir / "mi.csv",
        "Bicor": feature_dir / "bicor.csv",
    }
    tables = {name: pd.read_csv(path) for name, path in paths.items() if path.exists()}
    if "PDM" not in tables:
        raise FileNotFoundError(f"PDM feature table was not found: {paths['PDM']}")
    return tables


def _ks_table(
    feature_table: pd.DataFrame,
    positive_labels: set[str],
    negative_labels: set[str] | None = None,
) -> pd.DataFrame:
    identifier = feature_table.columns[0]
    values = feature_table.drop(columns=identifier).apply(pd.to_numeric, errors="coerce")
    complete = values.notna().all(axis=1)
    labels = feature_table.loc[complete, identifier].astype("string").str.strip()
    positive = labels.isin(positive_labels).to_numpy()
    comparison = (
        labels.isin(negative_labels).to_numpy()
        if negative_labels is not None
        else ~positive
    )
    if not positive.any() or not comparison.any():
        raise ValueError(
            "PDM feature table needs positive labels and a comparison set of complete rows."
        )
    complete_values = values.loc[complete].to_numpy()
    pvalues = feature_ks_pvalues(complete_values[positive], complete_values[comparison])
    return pd.DataFrame(
        {
            "feature": values.columns,
            "comparison": (
                "explicit_negative_controls"
                if negative_labels is not None
                else "unlabeled_candidates"
            ),
            "ks_pvalue": pvalues,
            "log10_ks_pvalue": np.log10(np.clip(pvalues, 1e-300, None)),
        }
    )


def _loocv_method_names(value: str, tables: dict[str, pd.DataFrame]) -> list[str]:
    requested = [name.strip() for name in value.split(",") if name.strip()]
    if not requested:
        raise ValueError("Provide at least one LOOCV method or 'all'.")
    if len(requested) == 1 and requested[0].lower() == "all":
        return list(tables)
    lookup = {name.lower(): name for name in tables}
    unknown = [name for name in requested if name.lower() not in lookup]
    if unknown:
        available = ", ".join(tables)
        raise ValueError(f"Unknown LOOCV methods: {', '.join(unknown)}. Available: {available}.")
    return [lookup[name.lower()] for name in requested]


def _save_loocv_plot(curves: pd.DataFrame, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.5, 5))
    for method, curve in curves.groupby("method", sort=False):
        area = float(curve["loocv_area"].iloc[0])
        candidates = int(curve["unlabeled_candidate_rows"].iloc[0])
        positives = int(curve["positive_rows"].iloc[0])
        axis.step(
            curve["rank_threshold"],
            curve["recovered_fraction"],
            where="post",
            linewidth=2,
            label=(
                f"{method} (area = {area:.3f})\n"
                f"{candidates:,} unlabeled candidates, {positives:,} positives"
            ),
        )
    axis.set_xlabel("Rank threshold (lower is better)")
    axis.set_ylabel("Fraction of held-out positives recovered")
    axis.set_title("Leave-one-positive-out rank recovery")
    axis.set_ylim(0, 1.02)
    axis.legend(title="Feature table", frameon=False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    label_path = args.label_file or args.data_dir / "oncogene.txt"
    labels = load_label_set(label_path, header=None, labels=args.positive_extra)
    negative_labels = (
        load_label_set(args.negative_label_file, header=None)
        if args.negative_label_file is not None
        else None
    )
    tables = _load_feature_tables(args.feature_dir, args.density_name)

    ranking = rank_candidates(
        tables["PDM"],
        labels,
        trials=args.ranking_trials,
        model=args.model,
        n_features=args.features_left,
        random_state=args.random_state,
        progress=True,
    )
    ranking_path = args.results_dir / f"{args.label_name}_ranking.csv"
    ranking.to_csv(ranking_path, index=False)

    benchmark_path = None
    if negative_labels is not None:
        benchmarks = []
        for name, table in tables.items():
            result = evaluate_labeled_features(
                table,
                labels,
                negative_labels=negative_labels,
                model=args.model,
                n_splits=args.n_splits,
                feature_selection=name == "PDM",
                n_features=args.features_left,
                random_state=args.random_state,
            )
            result.loc[:, "method"] = name
            benchmarks.append(result)
        benchmark_path = args.results_dir / f"{args.label_name}_method_benchmarks.csv"
        pd.concat(benchmarks, ignore_index=True).to_csv(benchmark_path, index=False)

    loocv_curve_path = None
    loocv_ranks_path = None
    loocv_plot_path = None
    if args.loocv:
        curves = []
        held_out_ranks = []
        for method in _loocv_method_names(args.loocv_methods, tables):
            curve, ranks = loocv_rank_recovery(
                tables[method],
                labels,
                model=args.model,
                trials=args.loocv_trials,
                feature_selection=method == "PDM",
                n_features=args.features_left,
                random_state=args.random_state,
                progress=True,
            )
            curves.append(curve.assign(method=method, comparison="unlabeled_candidates"))
            held_out_ranks.append(
                ranks.assign(method=method, comparison="unlabeled_candidates")
            )
        loocv_curve_path = args.results_dir / f"{args.label_name}_loocv_curve.csv"
        loocv_ranks_path = args.results_dir / f"{args.label_name}_loocv_held_out_ranks.csv"
        loocv_plot_path = args.results_dir / f"{args.label_name}_loocv.png"
        loocv_curves = pd.concat(curves, ignore_index=True)
        loocv_curves.to_csv(loocv_curve_path, index=False)
        pd.concat(held_out_ranks, ignore_index=True).to_csv(loocv_ranks_path, index=False)
        _save_loocv_plot(loocv_curves, loocv_plot_path)

    pvalue_path = args.results_dir / f"{args.label_name}_ks_pvalues.csv"
    _ks_table(tables["PDM"], labels, negative_labels).to_csv(pvalue_path, index=False)
    print(f"ranking: {ranking_path}")
    if benchmark_path is None:
        print("Skipping ROC AUC/AP: provide --negative-label-file with explicit negative labels.")
    else:
        print(f"benchmarks: {benchmark_path}")
    if loocv_curve_path is not None:
        print(f"LOOCV curve: {loocv_curve_path}")
        print(f"LOOCV held-out ranks: {loocv_ranks_path}")
        print(f"LOOCV chart: {loocv_plot_path}")
    print(f"KS p-values: {pvalue_path}")


if __name__ == "__main__":
    main()
