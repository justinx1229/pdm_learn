"""Benchmark simulated PDM and scalar association representations."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pdm_learn.modeling import MODEL_NAMES, evaluate_labeled_features

PROJECT_ROOT = Path(__file__).resolve().parents[2]
METHODS = ("pdm", "pearson", "spearman", "mi", "bicor")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark simulated feature representations.")
    parser.add_argument(
        "--feature-dir", type=Path, default=PROJECT_ROOT / "runs" / "simulation" / "features"
    )
    parser.add_argument(
        "--results-dir", type=Path, default=PROJECT_ROOT / "runs" / "simulation" / "results"
    )
    parser.add_argument("--model", choices=MODEL_NAMES, default="gradient_boosting")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--features-left", type=int, default=48)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def _labeled_table(positive_path: Path, negative_path: Path) -> tuple[pd.DataFrame, set[str]]:
    positive = pd.read_csv(positive_path)
    negative = pd.read_csv(negative_path)
    positive.insert(0, "simulation_id", [f"positive_{index}" for index in range(len(positive))])
    negative.insert(0, "simulation_id", [f"negative_{index}" for index in range(len(negative))])
    return pd.concat([positive, negative], ignore_index=True), set(positive["simulation_id"])


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for method in METHODS:
        table, labels = _labeled_table(
            args.feature_dir / "positive" / f"{method}.csv",
            args.feature_dir / "negative" / f"{method}.csv",
        )
        result = evaluate_labeled_features(
            table,
            labels,
            model=args.model,
            n_splits=args.n_splits,
            feature_selection=method == "pdm",
            n_features=args.features_left if method == "pdm" else None,
            random_state=args.random_state,
        )
        result.loc[:, "method"] = method.upper() if method != "pdm" else "PDM"
        rows.append(result)
    output_path = args.results_dir / "simulation_benchmarks.csv"
    pd.concat(rows, ignore_index=True).to_csv(output_path, index=False)
    print(f"simulation benchmarks: {output_path}")


if __name__ == "__main__":
    main()
