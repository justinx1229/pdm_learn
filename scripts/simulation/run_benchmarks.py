from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pdm_learn.modeling import KFold_PR, KFold_ROC_AUC, LOOCV


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark simulated feature representations.")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--artifacts-dir", type=Path, default=PROJECT_ROOT / "artifacts")
    parser.add_argument("--model", default="XGB")
    parser.add_argument("--trials", type=int, default=25)
    parser.add_argument("--features-left", type=int, default=48)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sim_dir = args.data_dir / "simulated"
    results_dir = args.artifacts_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    datasets = {
        "PDM": (
            pd.read_csv(sim_dir / "positive_heatmap.csv"),
            pd.read_csv(sim_dir / "negative_heatmap.csv"),
            True,
        ),
        "Pearson": (
            pd.read_csv(sim_dir / "positive_pearson.csv"),
            pd.read_csv(sim_dir / "negative_pearson.csv"),
            False,
        ),
        "Spearman": (
            pd.read_csv(sim_dir / "positive_spearman.csv"),
            pd.read_csv(sim_dir / "negative_spearman.csv"),
            False,
        ),
        "MutualInfo": (
            pd.read_csv(sim_dir / "positive_mi.csv"),
            pd.read_csv(sim_dir / "negative_mi.csv"),
            False,
        ),
        "Bicor": (
            pd.read_csv(sim_dir / "positive_bicor.csv"),
            pd.read_csv(sim_dir / "negative_bicor.csv"),
            False,
        ),
    }

    rows = []
    for name, (positive, negative, use_ks) in datasets.items():
        pos_values = positive.to_numpy()
        neg_values = negative.to_numpy()
        feature_limit = args.features_left if use_ks else None
        pr_area, _, _ = KFold_PR(
            pos_values,
            neg_values,
            args.trials,
            model=args.model,
            ks_test=use_ks,
            features_left=feature_limit,
        )
        rows.append(
            {
                "method": name,
                "average_precision": pr_area,
                "roc_auc": KFold_ROC_AUC(
                    pos_values,
                    neg_values,
                    model=args.model,
                    ks_test=use_ks,
                    features_left=feature_limit,
                ),
                "loocv_area": LOOCV(
                    pos_values,
                    neg_values,
                    5,
                    model=args.model,
                    ks_test=use_ks,
                    features_left=feature_limit,
                ),
            }
        )

    output_path = results_dir / "simulation_benchmarks.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"simulation benchmarks: {output_path}")


if __name__ == "__main__":
    main()
