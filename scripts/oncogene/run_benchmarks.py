"""Run oncogene benchmarks and save CSV results; plotting is a separate step."""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from pdm_learn.labels import load_label_set
from pdm_learn.modeling import (
    MODEL_NAMES,
    evaluate_labeled_features,
    feature_ks_pvalues,
    loocv_rank_recovery,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSAYS = ("gene_exp", "copy_num", "shRNA", "gene_mut", "CRISPR")


def _common_tables(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Use identical complete genes and row order for every comparison."""
    clean = {}
    for name, table in tables.items():
        identifier = table.columns[0]
        ids = table[identifier].astype("string").str.strip()
        values = table.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
        valid = ids.notna() & ids.ne("") & np.isfinite(values).all(axis=1)
        values = values.loc[valid].copy()
        values.index = pd.Index(ids.loc[valid], name=identifier)
        if values.index.has_duplicates:
            raise ValueError(f"{name} has duplicate gene identifiers.")
        clean[name] = values
    common = sorted(set.intersection(*(set(table.index) for table in clean.values())))
    if not common:
        raise ValueError("No complete genes are shared by the selected feature tables.")
    return {name: table.loc[common].reset_index() for name, table in clean.items()}


def _load_feature_tables(feature_dir: Path, density_name: str) -> dict[str, pd.DataFrame]:
    paths = {
        "PDM": feature_dir / density_name,
        "Pearson": feature_dir / "pearson.csv",
        "Spearman": feature_dir / "spearman.csv",
        "MutualInfo": feature_dir / "mi.csv",
        "Bicor": feature_dir / "bicor.csv",
    }
    tables = {name: pd.read_csv(path) for name, path in paths.items() if path.exists()}
    if not tables:
        raise FileNotFoundError(f"No feature tables found in {feature_dir}.")
    return tables


def _method_names(value: str, tables: dict[str, pd.DataFrame]) -> list[str]:
    requested = [name.strip() for name in value.split(",") if name.strip()]
    if not requested:
        raise ValueError("Provide at least one feature method or 'all'.")
    if len(requested) == 1 and requested[0].lower() == "all":
        return list(tables)
    lookup = {name.lower(): name for name in tables}
    unknown = [name for name in requested if name.lower() not in lookup]
    if unknown:
        available = ", ".join(tables)
        raise ValueError(f"Unknown feature methods: {', '.join(unknown)}. Available: {available}.")
    return [lookup[name.lower()] for name in requested]


def _ks_table(
    feature_table: pd.DataFrame,
    positive_labels: set[str],
    negative_labels: set[str] | None = None,
) -> pd.DataFrame:
    identifier = feature_table.columns[0]
    values = feature_table.drop(columns=identifier).apply(pd.to_numeric, errors="coerce")
    complete = values.notna().all(axis=1)
    labels = feature_table.loc[complete, identifier].astype("string").str.strip()
    positive = labels.isin(positive_labels).to_numpy()
    comparison = (
        labels.isin(negative_labels).to_numpy() if negative_labels is not None else ~positive
    )
    if not positive.any() or not comparison.any():
        raise ValueError(
            "PDM feature table needs positive labels and a comparison set of complete rows."
        )
    complete_values = values.loc[complete].to_numpy()
    pvalues = feature_ks_pvalues(complete_values[positive], complete_values[comparison])
    return pd.DataFrame(
        {
            "feature": values.columns,
            "comparison": (
                "explicit_negative_controls"
                if negative_labels is not None
                else "unlabeled_candidates"
            ),
            "ks_pvalue": pvalues,
            "log10_ks_pvalue": np.log10(np.clip(pvalues, 1e-300, None)),
        }
    )


def run_ks(
    tables: dict[str, pd.DataFrame], labels: set[str], args: argparse.Namespace
) -> dict[str, pd.DataFrame]:
    """Write the standalone PDM feature-distribution comparison."""
    if "PDM" not in tables:
        raise ValueError("The KS diagnostic requires a PDM feature table.")
    negatives = (
        load_label_set(args.negative_label_file, header=None)
        if args.negative_label_file is not None
        else None
    )
    return {"ks_pvalues": _ks_table(tables["PDM"], labels, negatives)}


def run_loocv(
    tables: dict[str, pd.DataFrame], labels: set[str], args: argparse.Namespace
) -> dict[str, pd.DataFrame]:
    """Return recovery curves and held-out ranks for each selected feature method."""
    curves, ranks = [], []
    for name, table in tables.items():
        curve, held_out = loocv_rank_recovery(
            table,
            labels,
            model=args.model,
            trials=args.loocv_trials,
            feature_selection=name == "PDM",
            n_features=args.features_left,
            random_state=args.random_state,
            progress=True,
        )
        metadata = {"method": name, "model": args.model, "comparison": "unlabeled_candidates"}
        curves.append(curve.assign(**metadata))
        ranks.append(held_out.assign(**metadata))
    return {
        "loocv_curve": pd.concat(curves, ignore_index=True),
        "loocv_held_out_ranks": pd.concat(ranks, ignore_index=True),
    }


def run_labeled(
    tables: dict[str, pd.DataFrame], labels: set[str], args: argparse.Namespace
) -> dict[str, pd.DataFrame]:
    """Evaluate ROC AUC and average precision against explicit negative controls."""
    return run_pr(tables, labels, args)


def run_pr(
    tables: dict[str, pd.DataFrame], labels: set[str], args: argparse.Namespace
) -> dict[str, pd.DataFrame]:
    """Save PR/ROC curves from pooled out-of-fold predictions, not LOOCV ranks."""
    negatives = (
        load_label_set(args.negative_label_file, header=None)
        if args.negative_label_file is not None else None
    )
    comparison = "explicit_negative_controls" if negatives is not None else "unlabeled_candidates"
    results, predictions, pr_curves, roc_curves = [], [], [], []
    for name, table in tables.items():
        result, predicted = evaluate_labeled_features(
            table,
            labels,
            negative_labels=negatives,
            model=args.model,
            n_splits=args.n_splits,
            feature_selection=name == "PDM",
            n_features=args.features_left,
            random_state=args.random_state,
            return_predictions=True,
        )
        metadata = {"method": name, "model": args.model, "comparison": comparison}
        results.append(result.assign(**metadata))
        predictions.append(predicted.assign(**metadata))
        precision, recall, thresholds = precision_recall_curve(predicted.target, predicted.score)
        pr_curves.append(pd.DataFrame({
            "recall": recall, "precision": precision, "threshold": np.r_[thresholds, np.nan],
            "average_precision": result.average_precision.iloc[0],
            "prevalence": predicted.target.mean(),
        }).assign(**metadata))
        fpr, tpr, thresholds = roc_curve(predicted.target, predicted.score)
        roc_curves.append(pd.DataFrame({
            "false_positive_rate": fpr, "true_positive_rate": tpr, "threshold": thresholds,
            "roc_auc": result.roc_auc.iloc[0],
        }).assign(**metadata))
    return {
        "method_benchmarks": pd.concat(results, ignore_index=True),
        "cv_predictions": pd.concat(predictions, ignore_index=True),
        "pr_curve": pd.concat(pr_curves, ignore_index=True),
        "roc_curve": pd.concat(roc_curves, ignore_index=True),
    }


def _pdm_recovery(table, labels, args, model):
    return loocv_rank_recovery(
        table, labels, model=model, trials=args.loocv_trials, feature_selection=True,
        n_features=args.features_left, random_state=args.random_state, progress=True,
    )


def run_models(tables, labels, args):
    """Compare supported classifiers on the same PDM features and genes."""
    curves, ranks, summaries = [], [], []
    for model in dict.fromkeys(args.models):
        started = perf_counter()
        curve, held_out = _pdm_recovery(tables["PDM"], labels, args, model)
        metadata = {"model": model, "method": "PDM", "comparison": "unlabeled_candidates"}
        curves.append(curve.assign(**metadata))
        ranks.append(held_out.assign(**metadata))
        summaries.append({
            **metadata, "loocv_area": curve.loocv_area.iloc[0],
            "elapsed_seconds": perf_counter() - started,
            "positive_rows": len(held_out),
            "unlabeled_candidate_rows": curve.unlabeled_candidate_rows.iloc[0],
        })
    return {
        "model_loocv_curve": pd.concat(curves, ignore_index=True),
        "model_loocv_held_out_ranks": pd.concat(ranks, ignore_index=True),
        "model_benchmarks": pd.DataFrame(summaries),
    }


def run_ablation(tables, labels, args):
    """Remove zero, one, or two assays, retaining only pairs among used assays."""
    table = tables["PDM"]
    summaries, ranks = [], []
    for count in range(3):
        for omitted in combinations(ASSAYS, count):
            used = set(ASSAYS) - set(omitted)
            columns = [column for column in table.columns[1:]
                       if set(column.split(".")[:2]) <= used]
            if not columns:
                raise ValueError(f"No PDM features remain after omitting {omitted}.")
            name = "+".join(omitted) or "none"
            print(f"ablation: omitted {name}", flush=True)
            curve, held_out = _pdm_recovery(table[[table.columns[0], *columns]], labels,
                                          args, args.model)
            metadata = {"omitted": name, "model": args.model, "comparison": "unlabeled_candidates"}
            summaries.append({
                **metadata, **{assay: int(assay in used) for assay in ASSAYS},
                "feature_columns": len(columns), "loocv_area": curve.loocv_area.iloc[0],
                "positive_rows": len(held_out),
                "unlabeled_candidate_rows": curve.unlabeled_candidate_rows.iloc[0],
            })
            ranks.append(held_out.assign(**metadata))
    return {"ablation": pd.DataFrame(summaries),
            "ablation_held_out_ranks": pd.concat(ranks, ignore_index=True)}


# Add a benchmark function above and register it here. Each function returns
# {filename_suffix: dataframe}; main handles the common CSV writing.
BENCHMARKS = {
    "ks": run_ks, "loocv": run_loocv, "models": run_models,
    "ablation": run_ablation, "pr": run_pr, "labeled": run_labeled,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument(
        "--feature-dir", type=Path, default=PROJECT_ROOT / "runs" / "oncogene" / "features"
    )
    parser.add_argument(
        "--results-dir", type=Path, default=PROJECT_ROOT / "runs" / "oncogene" / "results"
    )
    parser.add_argument("--density-name", default="dataset_trimmed_package.csv")
    parser.add_argument("--label-file", type=Path, help="Defaults to DATA_DIR/oncogene.txt.")
    parser.add_argument("--label-name", default="oncogene", help="Prefix for output filenames.")
    parser.add_argument("--positive-extra", action="append", default=[])
    parser.add_argument(
        "--negative-label-file",
        type=Path,
        help="Required for labeled evaluation; optional comparison labels for KS.",
    )
    parser.add_argument(
        "--benchmarks", nargs="+", choices=tuple(BENCHMARKS), default=["ks", "loocv"]
    )
    parser.add_argument(
        "--methods",
        default="all",
        help="Comma-separated methods, or all; also define the shared gene set. KS uses full PDM.",
    )
    parser.add_argument("--model", choices=MODEL_NAMES, default="gradient_boosting")
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES),
                        help="Classifiers for the models benchmark (PDM features only).")
    parser.add_argument("--loocv-trials", type=int, default=25)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--features-left", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    if "labeled" in args.benchmarks and args.negative_label_file is None:
        parser.error("The labeled benchmark requires --negative-label-file.")
    if "labeled" in args.benchmarks and "pr" in args.benchmarks:
        parser.error("Select pr or labeled; both write the same classification outputs.")
    return args


def main() -> None:
    args = parse_args()
    labels = load_label_set(
        args.label_file or args.data_dir / "oncogene.txt", header=None, labels=args.positive_extra
    )
    tables = _load_feature_tables(args.feature_dir, args.density_name)
    selected_methods = _method_names(args.methods, tables)
    selected_tables = {name: tables[name] for name in selected_methods}
    if any(name in args.benchmarks for name in ("models", "ablation")):
        if "PDM" not in tables:
            raise ValueError("Model comparison and assay ablation require PDM features.")
        selected_tables["PDM"] = tables["PDM"]
    common = _common_tables(selected_tables) if args.benchmarks != ["ks"] else selected_tables
    if args.benchmarks != ["ks"]:
        count = len(next(iter(common.values())))
        print(f"Comparison tables: {', '.join(common)}; shared genes: {count}", flush=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    for name in dict.fromkeys(args.benchmarks):
        inputs = tables if name == "ks" else common
        if name in ("loocv", "pr", "labeled"):
            inputs = {method: common[method] for method in selected_methods}
        for suffix, result in BENCHMARKS[name](inputs, labels, args).items():
            if name != "ks":
                result = result.assign(
                    random_state=args.random_state, n_features=args.features_left
                )
                setting = {"n_splits": args.n_splits} if name in ("pr", "labeled") else {
                    "trials": args.loocv_trials
                }
                result = result.assign(**setting)
            path = args.results_dir / f"{args.label_name}_{suffix}.csv"
            result.to_csv(path, index=False)
            print(f"{name}: {path}")


if __name__ == "__main__":
    main()
