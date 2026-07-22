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
    parser.add_argument("--density-name", default="dataset_trimmed_v3.csv")
    parser.add_argument("--model", choices=MODEL_NAMES, default="gradient_boosting")
    parser.add_argument("--ranking-trials", type=int, default=100)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--features-left", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--label-file", type=Path, default=None, help="One-column label list.")
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


def _ks_table(feature_table: pd.DataFrame, positive_labels: set[str]) -> pd.DataFrame:
    identifier = feature_table.columns[0]
    values = feature_table.drop(columns=identifier).apply(pd.to_numeric, errors="coerce")
    complete = values.notna().all(axis=1)
    labels = feature_table.loc[complete, identifier].astype("string").str.strip()
    positive = labels.isin(positive_labels).to_numpy()
    if not positive.any() or positive.all():
        raise ValueError("PDM feature table needs both labelled and unlabelled complete rows.")
    complete_values = values.loc[complete].to_numpy()
    pvalues = feature_ks_pvalues(complete_values[positive], complete_values[~positive])
    return pd.DataFrame(
        {
            "feature": values.columns,
            "ks_pvalue": pvalues,
            "log10_ks_pvalue": np.log10(np.clip(pvalues, 1e-300, None)),
        }
    )


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    label_path = args.label_file or args.data_dir / "oncogene.txt"
    labels = load_label_set(label_path, header=None, labels=args.positive_extra)
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

    benchmarks = []
    for name, table in tables.items():
        result = evaluate_labeled_features(
            table,
            labels,
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

    pvalue_path = args.results_dir / f"{args.label_name}_ks_pvalues.csv"
    _ks_table(tables["PDM"], labels).to_csv(pvalue_path, index=False)
    print(f"ranking: {ranking_path}")
    print(f"benchmarks: {benchmark_path}")
    print(f"KS p-values: {pvalue_path}")


if __name__ == "__main__":
    main()
