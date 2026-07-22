"""Generate deterministic simulated associations and their feature tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from pdm_learn.features import bicor, compute_mutual_information
from pdm_learn.simulation import (
    build_density_dataset,
    build_metric_dataset,
    collect_simulated_pairs,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate simulated association feature tables.")
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "runs" / "simulation" / "features"
    )
    parser.add_argument("--epsilon-std", type=float, default=3.0)
    parser.add_argument("--negative-repeats", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_positive_base() -> pd.DataFrame:
    """Return paired linear, quadratic, sinusoidal, and constant signals."""
    linear = np.linspace(-1, 1, 500)
    quadratic = np.linspace(-1, 1, 500)
    sinusoid = np.linspace(-np.pi, np.pi, 500)
    constant = np.linspace(-1, 1, 500)
    rows = [
        linear,
        linear,
        quadratic,
        quadratic**2,
        sinusoid,
        np.sin(sinusoid),
        constant,
        np.zeros_like(constant),
    ]
    return pd.DataFrame(rows)


def _scalar_features(
    seed_pairs: pd.DataFrame,
    simulated_pairs: list[tuple[np.ndarray, np.ndarray]],
    *,
    repeats: int,
    noise_scale: float,
    centers: np.ndarray,
    prefix: str,
) -> None:
    metrics = {
        "pearson": lambda x, y: float(pearsonr(x, y).statistic),
        "spearman": lambda x, y: float(spearmanr(x, y).statistic),
        "mi": compute_mutual_information,
        "bicor": bicor,
    }
    for name, metric in metrics.items():
        build_metric_dataset(
            seed_pairs,
            metric,
            repeats=repeats,
            noise_scale=noise_scale,
            centers=centers,
            column_name=name,
            simulated_pairs=simulated_pairs,
        ).to_csv(prefix / f"{name}.csv", index=False)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_pairs = build_positive_base()
    positive = collect_simulated_pairs(
        seed_pairs,
        repeats=1,
        noise_scale=args.epsilon_std,
        centers=np.linspace(-2, 2, 7),
        rng=np.random.default_rng(args.seed),
    )
    negative = collect_simulated_pairs(
        seed_pairs,
        repeats=args.negative_repeats,
        noise_scale=args.epsilon_std,
        shuffle_y=True,
        centers=np.linspace(-2, 2, 7),
        rng=np.random.default_rng(args.seed + 1),
    )
    centers = np.linspace(-2, 2, 7)

    collections = (("positive", positive, 1), ("negative", negative, args.negative_repeats))
    for label, pairs, repeats in collections:
        directory = args.output_dir / label
        directory.mkdir(exist_ok=True)
        build_density_dataset(
            seed_pairs,
            centers=centers,
            repeats=repeats,
            noise_scale=args.epsilon_std,
            sigma=0.1,
            simulated_pairs=pairs,
        ).to_csv(directory / "pdm.csv", index=False)
        _scalar_features(
            seed_pairs,
            pairs,
            repeats=repeats,
            noise_scale=args.epsilon_std,
            centers=centers,
            prefix=directory,
        )
    print(f"simulation features: {args.output_dir}")


if __name__ == "__main__":
    main()
