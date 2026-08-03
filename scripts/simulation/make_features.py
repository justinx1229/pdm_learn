"""Generate deterministic simulated associations and their feature tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.stats import pearsonr, spearmanr

from pdm_learn.features import bicor, compute_mutual_information
from pdm_learn.simulation import (
    build_density_dataset,
    build_metric_dataset,
    collect_simulated_pairs,
    standardize_pair,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate simulated association feature tables.")
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "runs" / "simulation" / "features"
    )
    parser.add_argument("--epsilon-std", type=float, default=3.0)
    parser.add_argument(
        "--positive-repeats",
        type=int,
        default=25,
        help="Number of perturbed positive realizations per seed association.",
    )
    parser.add_argument("--negative-repeats", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _arc_length_axis(
    derivative,
    start: float,
    stop: float,
    *,
    count: int,
) -> np.ndarray:
    """Sample a curve's input axis at approximately equal arc-length intervals."""
    grid = np.linspace(start, stop, max(10_000, count * 20))
    speed = np.sqrt(1.0 + np.square(derivative(grid)))
    arc_length = cumulative_trapezoid(speed, grid, initial=0.0)
    targets = np.linspace(0.0, arc_length[-1], count)
    return np.interp(targets, arc_length, grid)


def build_positive_base() -> pd.DataFrame:
    """Return the standardized seed associations used by the original benchmark."""
    linear = np.linspace(-1.0, 1.0, 500)
    quadratic = _arc_length_axis(lambda values: 2.0 * values, -1.0, 1.0, count=500)
    sinusoid = _arc_length_axis(np.cos, -np.pi, np.pi, count=500)
    constant = np.linspace(0.0, 1.0, 251)
    constant_x = np.append(constant, np.zeros_like(constant[1:-1]))
    constant_y = np.append(np.zeros_like(constant), constant[1:-1])

    pairs = (
        (linear, linear),
        (quadratic, quadratic**2),
        (sinusoid, np.sin(sinusoid)),
        (constant_x, constant_y),
    )
    rows = [values for x, y in pairs for values in standardize_pair(x, y)]
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
    if args.positive_repeats < 1:
        raise ValueError("positive_repeats must be a positive integer.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_pairs = build_positive_base()
    positive = collect_simulated_pairs(
        seed_pairs,
        repeats=args.positive_repeats,
        noise_scale=args.epsilon_std,
        centers=np.linspace(-2, 2, 7),
        rng=np.random.default_rng(args.seed),
    )
    negative = collect_simulated_pairs(
        seed_pairs,
        repeats=args.positive_repeats * args.negative_repeats,
        noise_scale=args.epsilon_std,
        shuffle_y=True,
        centers=np.linspace(-2, 2, 7),
        rng=np.random.default_rng(args.seed + 1),
    )
    centers = np.linspace(-2, 2, 7)

    collections = (
        ("positive", positive, args.positive_repeats),
        ("negative", negative, args.positive_repeats * args.negative_repeats),
    )
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
