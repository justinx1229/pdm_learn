from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sympy import Integer, Symbol, pi, sin
from sklearn.feature_selection import mutual_info_regression

from pdm_learn.oncogene import bicor
from pdm_learn.preprocessing import densitymap
from pdm_learn.simulation import build_heatmap_dataset, build_metric_dataset, collect_simulated_pairs, partition


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate simulated association feature tables.")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--epsilon-std", type=float, default=3.0)
    parser.add_argument("--negative-repeats", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _standardized_pair(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = x.astype(float)
    y = y.astype(float)
    x /= np.std(x)
    y /= np.std(y)
    x -= np.mean(x)
    y -= np.mean(y)
    return x, y


def build_positive_base() -> pd.DataFrame:
    u = Symbol("u")
    equations = [
        (u, (-1, 1), 500),
        (u**2, (-1, 1), 500),
        (sin(u), (-pi, pi), 500),
        (Integer(0), (0, 1), 251),
    ]
    rows = []
    for equation, bounds, count in equations:
        x = partition(u, equation, bounds=bounds, num=count).astype(float)
        y = np.array([equation.subs(u, value).evalf() for value in x]).astype(float)
        if equation == 0:
            temp = x.copy()
            x = np.append(x, y[1:-1])
            y = np.append(y, temp[1:-1])
        x, y = _standardized_pair(x, y)
        rows.extend([x, y])
    return pd.DataFrame(rows)


def compute_mi(x: np.ndarray, y: np.ndarray) -> float:
    return float(mutual_info_regression(x.reshape(-1, 1), y, discrete_features=False)[0])


def main() -> None:
    args = parse_args()
    output_dir = args.data_dir / "simulated"
    output_dir.mkdir(parents=True, exist_ok=True)

    base = build_positive_base()
    positive_rows = []
    for index in range(0, len(base), 2):
        x = base.iloc[index].to_numpy()
        y = base.iloc[index + 1].to_numpy()
        for _ in range(25):
            positive_rows.extend([x, y])
    positive = pd.DataFrame(positive_rows).reset_index(drop=True)
    positive_path = output_dir / "positive.csv"
    positive.to_csv(positive_path, index=False)

    centers = np.linspace(-2, 2, 7)
    positive_pairs = collect_simulated_pairs(
        positive,
        repeats=1,
        epsilon_std=args.epsilon_std,
        centers=centers,
        rng=np.random.default_rng(args.seed),
    )
    negative_pairs = collect_simulated_pairs(
        positive,
        repeats=args.negative_repeats,
        epsilon_std=args.epsilon_std,
        shuffle_y=True,
        centers=centers,
        rng=np.random.default_rng(args.seed + 1),
    )

    build_heatmap_dataset(
        positive,
        densitymap,
        centers=centers,
        repeats=1,
        epsilon_std=args.epsilon_std,
        sigma=0.1,
        simulated_pairs=positive_pairs,
    ).to_csv(output_dir / "positive_heatmap.csv", index=False)
    build_heatmap_dataset(
        positive,
        densitymap,
        centers=centers,
        repeats=args.negative_repeats,
        epsilon_std=args.epsilon_std,
        sigma=0.1,
        shuffle_y=True,
        simulated_pairs=negative_pairs,
    ).to_csv(output_dir / "negative_heatmap.csv", index=False)

    metric_specs = {
        "pearson": lambda x, y: pearsonr(x, y)[0],
        "spearman": lambda x, y: spearmanr(x, y)[0],
        "mi": compute_mi,
        "bicor": bicor,
    }
    for name, metric_fn in metric_specs.items():
        column_name = name.upper() if name != "bicor" else "BiCor"
        build_metric_dataset(
            positive,
            metric_fn,
            repeats=1,
            epsilon_std=args.epsilon_std,
            centers=centers,
            column_name=column_name,
            simulated_pairs=positive_pairs,
        ).to_csv(output_dir / f"positive_{name}.csv", index=False)
        build_metric_dataset(
            positive,
            metric_fn,
            repeats=args.negative_repeats,
            epsilon_std=args.epsilon_std,
            shuffle_y=True,
            centers=centers,
            column_name=column_name,
            simulated_pairs=negative_pairs,
        ).to_csv(output_dir / f"negative_{name}.csv", index=False)

    print(f"simulation features: {output_dir}")


if __name__ == "__main__":
    main()
