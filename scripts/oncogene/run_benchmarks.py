"""Run oncogene benchmarks and save CSV results; plotting is a separate step."""

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
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_feature_tables(feature_dir: Path, density_name: str) -> dict[str, pd.DataFrame]:
    paths = {
        "PDM": feature_dir / density_name,
        "Pearson": feature_dir / "pearson.csv",
        "Spearman": feature_dir / "spearman.csv",
        "MutualInfo": feature_dir / "mi.csv",
        "Bicor": feature_dir / "bicor.csv",
    }
    tables = {name: pd.read_csv(path) for name, path in paths.items() if path.exists()}
    if not tables:
        raise FileNotFoundError(f"No feature tables found in {feature_dir}.")
    return tables


def _method_names(value: str, tables: dict[str, pd.DataFrame]) -> list[str]:
    requested = [name.strip() for name in value.split(",") if name.strip()]
    if not requested:
        raise ValueError("Provide at least one feature method or 'all'.")
    if len(requested) == 1 and requested[0].lower() == "all":
        return list(tables)
    lookup = {name.lower(): name for name in tables}
    unknown = [name for name in requested if name.lower() not in lookup]
    if unknown:
        available = ", ".join(tables)
        raise ValueError(f"Unknown feature methods: {', '.join(unknown)}. Available: {available}.")
    return [lookup[name.lower()] for name in requested]


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
        labels.isin(negative_labels).to_numpy() if negative_labels is not None else ~positive
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


def run_ks(
    tables: dict[str, pd.DataFrame], labels: set[str], args: argparse.Namespace
) -> dict[str, pd.DataFrame]:
    """Write the standalone PDM feature-distribution comparison."""
    if "PDM" not in tables:
        raise ValueError("The KS diagnostic requires a PDM feature table.")
    negatives = (
        load_label_set(args.negative_label_file, header=None)
        if args.negative_label_file is not None
        else None
    )
    return {"ks_pvalues": _ks_table(tables["PDM"], labels, negatives)}


def run_loocv(
    tables: dict[str, pd.DataFrame], labels: set[str], args: argparse.Namespace
) -> dict[str, pd.DataFrame]:
    """Return recovery curves and held-out ranks for each selected feature method."""
    curves, ranks = [], []
    for name, table in tables.items():
        curve, held_out = loocv_rank_recovery(
            table,
            labels,
            model=args.model,
            trials=args.loocv_trials,
            feature_selection=name == "PDM",
            n_features=args.features_left,
            random_state=args.random_state,
            progress=True,
        )
        curves.append(curve.assign(method=name, comparison="unlabeled_candidates"))
        ranks.append(held_out.assign(method=name, comparison="unlabeled_candidates"))
    return {
        "loocv_curve": pd.concat(curves, ignore_index=True),
        "loocv_held_out_ranks": pd.concat(ranks, ignore_index=True),
    }


def run_labeled(
    tables: dict[str, pd.DataFrame], labels: set[str], args: argparse.Namespace
) -> dict[str, pd.DataFrame]:
    """Evaluate ROC AUC and average precision against explicit negative controls."""
    negatives = load_label_set(args.negative_label_file, header=None)
    results = []
    for name, table in tables.items():
        result = evaluate_labeled_features(
            table,
            labels,
            negative_labels=negatives,
            model=args.model,
            n_splits=args.n_splits,
            feature_selection=name == "PDM",
            n_features=args.features_left,
            random_state=args.random_state,
        )
        results.append(result.assign(method=name))
    return {"method_benchmarks": pd.concat(results, ignore_index=True)}


# Add a benchmark function above and register it here. Each function returns
# {filename_suffix: dataframe}; main handles the common CSV writing.
BENCHMARKS = {"ks": run_ks, "loocv": run_loocv, "labeled": run_labeled}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument(
        "--feature-dir", type=Path, default=PROJECT_ROOT / "runs" / "oncogene" / "features"
    )
    parser.add_argument(
        "--results-dir", type=Path, default=PROJECT_ROOT / "runs" / "oncogene" / "results"
    )
    parser.add_argument("--density-name", default="dataset_trimmed_package.csv")
    parser.add_argument("--label-file", type=Path, help="Defaults to DATA_DIR/oncogene.txt.")
    parser.add_argument("--label-name", default="oncogene", help="Prefix for output filenames.")
    parser.add_argument("--positive-extra", action="append", default=[])
    parser.add_argument(
        "--negative-label-file",
        type=Path,
        help="Required for labeled evaluation; optional comparison labels for KS.",
    )
    parser.add_argument(
        "--benchmarks", nargs="+", choices=tuple(BENCHMARKS), default=["ks", "loocv"]
    )
    parser.add_argument(
        "--methods",
        default="all",
        help="Comma-separated methods for LOOCV/labeled evaluation, or all. KS always uses PDM.",
    )
    parser.add_argument("--model", choices=MODEL_NAMES, default="gradient_boosting")
    parser.add_argument("--loocv-trials", type=int, default=25)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--features-left", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    if "labeled" in args.benchmarks and args.negative_label_file is None:
        parser.error("The labeled benchmark requires --negative-label-file.")
    return args


def main() -> None:
    args = parse_args()
    labels = load_label_set(
        args.label_file or args.data_dir / "oncogene.txt", header=None, labels=args.positive_extra
    )
    tables = _load_feature_tables(args.feature_dir, args.density_name)
    selected_tables = {name: tables[name] for name in _method_names(args.methods, tables)}
    args.results_dir.mkdir(parents=True, exist_ok=True)
    for name in dict.fromkeys(args.benchmarks):
        inputs = tables if name == "ks" else selected_tables
        for suffix, result in BENCHMARKS[name](inputs, labels, args).items():
            path = args.results_dir / f"{args.label_name}_{suffix}.csv"
            result.to_csv(path, index=False)
            print(f"{name}: {path}")


if __name__ == "__main__":
    main()
