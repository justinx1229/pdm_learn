from __future__ import annotations

import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run gene-label ranking and benchmark summaries.")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--artifacts-dir", type=Path, default=PROJECT_ROOT / "artifacts")
    parser.add_argument("--feature-dir", type=Path, default=None, help="Directory containing generated feature tables.")
    parser.add_argument("--density-name", default="dataset_trimmed_v3.csv")
    parser.add_argument("--model", default="XGB")
    parser.add_argument("--ranking-trials", type=int, default=100)
    parser.add_argument("--curve-trials", type=int, default=25)
    parser.add_argument("--feature-sweep-trials", type=int, default=20)
    parser.add_argument("--features-left", type=int, default=None)
    parser.add_argument("--oncogene", type=Path, default=None, help="One-column known-oncogene list.")
    parser.add_argument("--label-file", type=Path, default=None, help="One-column positive-label gene list.")
    parser.add_argument("--label-name", default="oncogene", help="Name used in output filenames.")
    parser.add_argument(
        "--positive-extra",
        action="append",
        default=[],
        help="Additional positive label to include. Repeat for multiple genes.",
    )
    parser.add_argument(
        "--skip-feature-sweep",
        action="store_true",
        help="Skip the feature-count LOOCV sweep.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import numpy as np
    import pandas as pd

    from pdm_learn.modeling import KFold_PR, LOOCV, area_table, ks_pvalue
    from pdm_learn.oncogene import load_oncogene_feature_sets, rank_candidate_genes

    results_dir = args.artifacts_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("Loading feature tables and positive-label list...")
    positive_extras = list(args.positive_extra)
    data_dict, _ = load_oncogene_feature_sets(
        args.data_dir,
        feature_dir=args.feature_dir,
        density_name=args.density_name,
        oncogene_path=args.oncogene,
        label_path=args.label_file,
        positive_label_extras=positive_extras,
    )
    if "PDM" not in data_dict:
        raise FileNotFoundError("The PDM feature table was not found.")
    positive, negative = data_dict["PDM"]

    print(f"Ranking candidate genes with {args.ranking_trials} trials...")
    ranking = rank_candidate_genes(
        positive,
        negative,
        trials=args.ranking_trials,
        model=args.model,
        ks_test=True,
        features_left=args.features_left,
        progress=True,
    )
    ranking_path = results_dir / f"{args.label_name}_ranking.csv"
    ranking.to_csv(ranking_path, index=False)

    print("Benchmarking feature sets...")
    curve_rows = []
    for name, (pos, neg) in data_dict.items():
        print(f"  {name}")
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
            progress=True,
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

    benchmark_path = results_dir / f"{args.label_name}_method_benchmarks.csv"
    pd.DataFrame(curve_rows).to_csv(benchmark_path, index=False)

    sweep_path = None
    if not args.skip_feature_sweep:
        print("Running feature-count sweep...")
        feature_counts = [10, 20, 50, 100, 124, 150, 200, 250, 300, 349]
        feature_counts = [value for value in feature_counts if value <= positive.shape[1] - 1]
        sweep_areas = area_table(
            positive.iloc[:, 1:].to_numpy(),
            negative.iloc[:, 1:].to_numpy(),
            args.feature_sweep_trials,
            model=args.model,
            feat_arr=feature_counts,
        )
        sweep_path = results_dir / f"{args.label_name}_feature_sweep.csv"
        pd.DataFrame({"features_left": feature_counts, "loocv_area": sweep_areas}).to_csv(sweep_path, index=False)

    print("Computing KS p-values...")
    p_values = ks_pvalue(positive.iloc[:, 1:].to_numpy(), negative.iloc[:, 1:].to_numpy())
    pvalue_path = results_dir / f"{args.label_name}_ks_pvalues.csv"
    pd.DataFrame({"feature": positive.columns[1:], "ks_pvalue": p_values, "log10_ks_pvalue": np.log10(np.clip(p_values, 1e-300, None))}).to_csv(
        pvalue_path,
        index=False,
    )

    print(f"ranking: {ranking_path}")
    print(f"benchmarks: {benchmark_path}")
    if sweep_path is not None:
        print(f"feature sweep: {sweep_path}")
    print(f"KS p-values: {pvalue_path}")


if __name__ == "__main__":
    main()
