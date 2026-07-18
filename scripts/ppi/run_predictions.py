from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pdm_learn.ppi import evaluate_ppi_controls, load_cancer_complexes, rank_ppi_pairs_cancer


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PPI control benchmarks and rank BioGRID pairs.")
    parser.add_argument("--artifacts-dir", type=Path, default=PROJECT_ROOT / "artifacts")
    parser.add_argument("--complexes", type=Path, default=PROJECT_ROOT / "data" / "cancer_complexes_clean.xlsx")
    parser.add_argument("--biogrid-features", type=Path, default=PROJECT_ROOT / "artifacts" / "results" / "clean_biogrid_interactions_pdm.csv")
    parser.add_argument("--model", default="GBR")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--features-left", type=int, default=None)
    parser.add_argument("--skip-control-benchmark", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.artifacts_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    complexes = load_cancer_complexes(args.complexes)
    biogrid_features = pd.read_csv(args.biogrid_features, low_memory=False)
    ranking = rank_ppi_pairs_cancer(
        biogrid_features,
        complexes,
        trials=args.trials,
        model=args.model,
        ks_test=True,
        features_left=args.features_left,
        progress=True,
    )
    ranking_path = results_dir / "ranked_PPI_pairs_cancer.csv"
    ranking.to_csv(ranking_path, index=False)
    print(f"PPI ranking: {ranking_path}")

    if not args.skip_control_benchmark:
        positive_path = args.artifacts_dir / "controls" / "positive_controls.pkl"
        negative_path = args.artifacts_dir / "controls" / "negative_controls.pkl"
        if positive_path.exists() and negative_path.exists():
            positive_controls = pd.read_pickle(positive_path)
            negative_controls = pd.read_pickle(negative_path)
            benchmark = evaluate_ppi_controls(
                positive_controls,
                negative_controls,
                model=args.model,
                features_left=args.features_left,
                progress=True,
            )
            benchmark_path = results_dir / "ppi_control_benchmarks.csv"
            benchmark.to_csv(benchmark_path, index=False)
            print(f"PPI control benchmarks: {benchmark_path}")
        else:
            print("Skipping PPI control benchmark because control pickles were not found.")


if __name__ == "__main__":
    main()
