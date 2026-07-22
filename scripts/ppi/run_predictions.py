"""Rank BioGRID pairs using curated cancer-complex pairs as positive labels."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pdm_learn.modeling import MODEL_NAMES, evaluate_labeled_features, rank_candidates
from pdm_learn.workflows.ppi import cancer_complex_pair_set, load_cancer_complexes

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark and rank BioGRID PPI feature rows.")
    parser.add_argument(
        "--features",
        type=Path,
        default=PROJECT_ROOT / "runs" / "ppi" / "features" / "biogrid_pdm.csv",
    )
    parser.add_argument(
        "--complexes", type=Path, default=PROJECT_ROOT / "data" / "cancer_complexes_clean.xlsx"
    )
    parser.add_argument(
        "--results-dir", type=Path, default=PROJECT_ROOT / "runs" / "ppi" / "results"
    )
    parser.add_argument("--model", choices=MODEL_NAMES, default="gradient_boosting")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--features-left", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    features = pd.read_csv(args.features, low_memory=False)
    if "canonical_pair" not in features.columns:
        raise ValueError(
            "PPI features must contain the canonical_pair column from make_features.py."
        )
    labels = cancer_complex_pair_set(load_cancer_complexes(args.complexes))

    ranking = rank_candidates(
        features,
        labels,
        id_column="canonical_pair",
        trials=args.trials,
        model=args.model,
        n_features=args.features_left,
        random_state=args.random_state,
        progress=True,
    )
    ranking_path = args.results_dir / "ranked_PPI_pairs_cancer.csv"
    ranking.to_csv(ranking_path, index=False)

    benchmark = evaluate_labeled_features(
        features,
        labels,
        id_column="canonical_pair",
        model=args.model,
        n_splits=args.n_splits,
        feature_selection=True,
        n_features=args.features_left,
        random_state=args.random_state,
    )
    benchmark.loc[:, "method"] = "PDM"
    benchmark_path = args.results_dir / "ppi_benchmark.csv"
    benchmark.to_csv(benchmark_path, index=False)
    print(f"PPI ranking: {ranking_path}")
    print(f"PPI benchmark: {benchmark_path}")


if __name__ == "__main__":
    main()
