from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pdm_learn.oncogene import ONCOGENE_HEATMAP_DIMENSIONS


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate publication figure files from saved results.")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--artifacts-dir", type=Path, default=PROJECT_ROOT / "artifacts")
    parser.add_argument("--figures-dir", type=Path, default=PROJECT_ROOT / "figures")
    return parser.parse_args()


def save_oncogene_ks_heatmaps(pvalue_path: Path, output_path: Path) -> None:
    pvalues = pd.read_csv(pvalue_path)
    values = pvalues["log10_ks_pvalue"].to_numpy()
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    index = 0
    for ax, dimensions in zip(axes.flat, ONCOGENE_HEATMAP_DIMENSIONS):
        size = dimensions[0] * dimensions[1]
        matrix = values[index : index + size].reshape(dimensions)
        sns.heatmap(np.flip(matrix, 0), ax=ax, cmap="gist_heat_r", vmin=vmin, vmax=vmax, cbar=False, xticklabels=False, yticklabels=False)
        ax.set_xlabel("")
        ax.set_ylabel("")
        index += size
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_simulation_heatmap(simulated_dir: Path, output_path: Path) -> None:
    positive_heatmap = pd.read_csv(simulated_dir / "positive_heatmap.csv")
    matrix = positive_heatmap.iloc[0].to_numpy(dtype=float).reshape(7, 7)
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(matrix, ax=ax, cmap="hot", cbar=False, xticklabels=False, yticklabels=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")

    pvalue_path = args.artifacts_dir / "results" / "oncogene_ks_pvalues.csv"
    if pvalue_path.exists():
        output_path = args.figures_dir / "oncogene_ks_heatmaps.png"
        save_oncogene_ks_heatmaps(pvalue_path, output_path)
        print(f"oncogene KS heatmaps: {output_path}")
    else:
        print(f"Skipping oncogene KS heatmaps; missing {pvalue_path}")

    simulated_dir = args.data_dir / "simulated"
    if (simulated_dir / "positive_heatmap.csv").exists():
        output_path = args.figures_dir / "simulation_positive_heatmap.png"
        save_simulation_heatmap(simulated_dir, output_path)
        print(f"simulation heatmap: {output_path}")
    else:
        print(f"Skipping simulation heatmap; missing {simulated_dir / 'positive_heatmap.csv'}")


if __name__ == "__main__":
    main()
