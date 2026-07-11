from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run generic gene-label ranking and benchmark summaries.")
    parser.add_argument("--label-file", type=Path, required=True, help="One-column positive-label gene list.")
    parser.add_argument("--feature-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "generic" / "features")
    parser.add_argument("--manifest", type=Path, default=None, help="Feature manifest from scripts/generic/make_features.py.")
    parser.add_argument(
        "--feature",
        action="append",
        default=[],
        metavar="NAME=CSV",
        help="Feature table to evaluate. Repeat to bypass or supplement the manifest.",
    )
    parser.add_argument("--ranking-feature", default="PDM", help="Feature table name used for candidate ranking.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "generic" / "results")
    parser.add_argument("--label-name", default="positive_label")
    parser.add_argument("--id-column-name", default=None, help="Identifier column name for ranking output. Defaults to the feature table's first column.")
    parser.add_argument("--positive-extra", action="append", default=[], help="Additional positive label. Repeat as needed.")
    parser.add_argument("--model", default="XGB")
    parser.add_argument("--ranking-trials", type=int, default=100)
    parser.add_argument("--curve-trials", type=int, default=25)
    parser.add_argument("--feature-sweep-trials", type=int, default=20)
    parser.add_argument("--features-left", type=int, default=None)
    parser.add_argument("--feature-sweep-values", default="10,20,50,100,124,150,200,250,300,349")
    parser.add_argument("--skip-ranking", action="store_true")
    parser.add_argument("--skip-benchmarks", action="store_true")
    parser.add_argument("--skip-feature-sweep", action="store_true")
    return parser.parse_args()


def _parse_name_value(value: str, separator: str = "=") -> tuple[str, str]:
    if separator not in value:
        raise ValueError(f"Expected NAME{separator}VALUE, got {value!r}")
    name, text = value.split(separator, maxsplit=1)
    name = name.strip()
    text = text.strip()
    if not name or not text:
        raise ValueError(f"Expected NAME{separator}VALUE, got {value!r}")
    return name, text


def _resolve_path(path_text: str | Path, *, manifest_dir: Path | None = None) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    if (PROJECT_ROOT / path).exists():
        return PROJECT_ROOT / path
    if manifest_dir is not None:
        return manifest_dir / path
    return path


def feature_paths(args: argparse.Namespace) -> dict[str, Path]:
    manifest_path = args.manifest or args.feature_dir / "feature_manifest.json"
    paths = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        manifest_dir = manifest_path.parent
        paths.update(
            {
                name: _resolve_path(path, manifest_dir=manifest_dir)
                for name, path in manifest.get("feature_tables", {}).items()
            }
        )

    for value in args.feature:
        name, path_text = _parse_name_value(value)
        paths[name] = Path(path_text)

    if not paths:
        raise FileNotFoundError(
            "No feature tables were found. Provide --manifest, --feature-dir with a feature_manifest.json, "
            "or one or more --feature NAME=CSV values."
        )
    return paths


def positive_labels(label_file: Path, extras: list[str]) -> set[str]:
    import pandas as pd

    labels = set(pd.read_csv(label_file).iloc[:, 0].astype(str).str.strip())
    labels.update(str(value).strip() for value in extras if str(value).strip())
    return labels


def load_feature_sets(paths: dict[str, Path], labels: set[str]):
    import pandas as pd

    data = {}
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing feature table for {name}: {path}")
        table = pd.read_csv(path).dropna().reset_index(drop=True)
        label_mask = table.iloc[:, 0].astype(str).str.strip().isin(labels)
        data[name] = (
            table.loc[label_mask].reset_index(drop=True),
            table.loc[~label_mask].reset_index(drop=True),
        )
    return data


def parse_sweep_values(text: str, feature_count: int) -> list[int]:
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    return [value for value in values if 0 < value <= feature_count]


def main() -> None:
    args = parse_args()

    import numpy as np
    import pandas as pd

    from pdm_learn import rank_candidate_genes
    from pdm_learn.modeling import KFold_PR, LOOCV, area_table, ks_pvalue

    paths = feature_paths(args)
    labels = positive_labels(args.label_file, args.positive_extra)
    data = load_feature_sets(paths, labels)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.ranking_feature not in data:
        raise ValueError(
            f"Ranking feature {args.ranking_feature!r} was not found. Available: {', '.join(sorted(data))}"
        )
    positive, negative = data[args.ranking_feature]

    if not args.skip_ranking:
        print(f"Ranking candidates with {args.ranking_feature} features...")
        ranking = rank_candidate_genes(
            positive,
            negative,
            trials=args.ranking_trials,
            model=args.model,
            ks_test=True,
            features_left=args.features_left,
            id_column_name=args.id_column_name,
            progress=True,
        )
        ranking_path = args.output_dir / f"{args.label_name}_ranking.csv"
        ranking.to_csv(ranking_path, index=False)
        print(f"ranking: {ranking_path}")

    if not args.skip_benchmarks:
        print("Benchmarking feature tables...")
        rows = []
        for name, (pos, neg) in data.items():
            print(f"  {name}")
            pos_values = pos.iloc[:, 1:].to_numpy()
            neg_values = neg.iloc[:, 1:].to_numpy()
            use_ks = name == args.ranking_feature
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
            rows.append({"feature_table": name, "average_precision": pr_area, "loocv_area": loocv_area})

        benchmark_path = args.output_dir / f"{args.label_name}_feature_benchmarks.csv"
        pd.DataFrame(rows).to_csv(benchmark_path, index=False)
        print(f"benchmarks: {benchmark_path}")

    if not args.skip_feature_sweep:
        feature_counts = parse_sweep_values(args.feature_sweep_values, positive.shape[1] - 1)
        if feature_counts:
            print("Running ranking-feature feature-count sweep...")
            sweep_areas = area_table(
                positive.iloc[:, 1:].to_numpy(),
                negative.iloc[:, 1:].to_numpy(),
                args.feature_sweep_trials,
                model=args.model,
                feat_arr=feature_counts,
            )
            sweep_path = args.output_dir / f"{args.label_name}_feature_sweep.csv"
            pd.DataFrame({"features_left": feature_counts, "loocv_area": sweep_areas}).to_csv(sweep_path, index=False)
            print(f"feature sweep: {sweep_path}")

    print("Computing ranking-feature KS p-values...")
    p_values = ks_pvalue(positive.iloc[:, 1:].to_numpy(), negative.iloc[:, 1:].to_numpy())
    pvalue_path = args.output_dir / f"{args.label_name}_ks_pvalues.csv"
    pd.DataFrame(
        {
            "feature": positive.columns[1:],
            "ks_pvalue": p_values,
            "log10_ks_pvalue": np.log10(np.clip(p_values, 1e-300, None)),
        }
    ).to_csv(pvalue_path, index=False)
    print(f"KS p-values: {pvalue_path}")


if __name__ == "__main__":
    main()
