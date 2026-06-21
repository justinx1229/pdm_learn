from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pdm_learn.modeling import KFold_PR, LOOCV, area_table, ks_pvalue
from pdm_learn.oncogene import load_oncogene_feature_sets, rank_candidate_oncogenes


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run oncogene ranking and benchmark summaries.")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--artifacts-dir", type=Path, default=PROJECT_ROOT / "artifacts")
    parser.add_argument("--density-name", default="dataset_trimmed_v3.csv")
    parser.add_argument("--model", default="XGB")
    parser.add_argument("--ranking-trials", type=int, default=100)
    parser.add_argument("--curve-trials", type=int, default=25)
    parser.add_argument("--feature-sweep-trials", type=int, default=20)
    parser.add_argument("--features-left", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.artifacts_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    data_dict, _ = load_oncogene_feature_sets(args.data_dir, density_name=args.density_name)
    positive, negative = data_dict["PDM"]

    ranking = rank_candidate_oncogenes(
        positive,
        negative,
        trials=args.ranking_trials,
        model=args.model,
        ks_test=True,
        features_left=args.features_left,
    )
    ranking_path = results_dir / "oncogene_ranking.csv"
    ranking.to_csv(ranking_path, index=False)

    curve_rows = []
    for name, (pos, neg) in data_dict.items():
        pos_values = pos.iloc[:, 1:].to_numpy()
        neg_values = neg.iloc[:, 1:].to_numpy()
        use_ks = name == "PDM"
        pr_area, _, _ = KFold_PR(
            pos_values,
            neg_values,
            args.curve_trials,
            model=args.model,
            ks_test=use_ks,
            features_left=args.features_left,
        )
        loocv_area = LOOCV(
            pos_values,
            neg_values,
            args.curve_trials,
            model=args.model,
            ks_test=use_ks,
            features_left=args.features_left,
        )
        curve_rows.append({"method": name, "average_precision": pr_area, "loocv_area": loocv_area})

    benchmark_path = results_dir / "oncogene_method_benchmarks.csv"
    pd.DataFrame(curve_rows).to_csv(benchmark_path, index=False)

    feature_counts = [10, 20, 50, 100, 124, 150, 200, 250, 300, 349]
    sweep_areas = area_table(
        positive.iloc[:, 1:].to_numpy(),
        negative.iloc[:, 1:].to_numpy(),
        args.feature_sweep_trials,
        model=args.model,
        feat_arr=feature_counts,
    )
    sweep_path = results_dir / "oncogene_feature_sweep.csv"
    pd.DataFrame({"features_left": feature_counts, "loocv_area": sweep_areas}).to_csv(sweep_path, index=False)

    p_values = ks_pvalue(positive.iloc[:, 1:].to_numpy(), negative.iloc[:, 1:].to_numpy())
    pvalue_path = results_dir / "oncogene_ks_pvalues.csv"
    pd.DataFrame({"feature": positive.columns[1:], "ks_pvalue": p_values, "log10_ks_pvalue": np.log10(np.clip(p_values, 1e-300, None))}).to_csv(
        pvalue_path,
        index=False,
    )

    print(f"ranking: {ranking_path}")
    print(f"benchmarks: {benchmark_path}")
    print(f"feature sweep: {sweep_path}")
    print(f"KS p-values: {pvalue_path}")


if __name__ == "__main__":
    main()
