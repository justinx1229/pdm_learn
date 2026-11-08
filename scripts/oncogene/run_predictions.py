"""Rank oncogene candidates and save the ranked gene list."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pdm_learn.labels import load_label_set
from pdm_learn.modeling import (
    MODEL_NAMES,
    rank_candidates,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank gene-level candidates.")
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


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    label_path = args.label_file or args.data_dir / "oncogene.txt"
    labels = load_label_set(label_path, header=None, labels=args.positive_extra)
    table = pd.read_csv(args.feature_dir / args.density_name)

    ranking = rank_candidates(
        table,
        labels,
        trials=args.ranking_trials,
        model=args.model,
        n_features=args.features_left,
        random_state=args.random_state,
        progress=True,
    )
    ranking_path = args.results_dir / f"{args.label_name}_ranking.csv"
    ranking.to_csv(ranking_path, index=False)

    print(f"ranking: {ranking_path}")


if __name__ == "__main__":
    main()
