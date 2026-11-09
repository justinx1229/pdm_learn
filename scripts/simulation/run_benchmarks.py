"""Save cross-validated simulation benchmarks and curves; plotting is separate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from pdm_learn.modeling import MODEL_NAMES, evaluate_labeled_features

PROJECT_ROOT = Path(__file__).resolve().parents[2]
METHODS = ("pdm", "pearson", "spearman", "mi", "bicor")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark simulated feature representations.")
    parser.add_argument(
        "--feature-dir", type=Path, default=PROJECT_ROOT / "runs" / "simulation" / "features"
    )
    parser.add_argument(
        "--results-dir", type=Path, default=PROJECT_ROOT / "runs" / "simulation" / "results"
    )
    parser.add_argument("--model", choices=MODEL_NAMES, default="gradient_boosting")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES,
                        help="Also compare these classifiers using PDM features.")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--features-left", type=int, default=48)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def _labeled_table(positive_path: Path, negative_path: Path) -> tuple[pd.DataFrame, set[str]]:
    positive = pd.read_csv(positive_path)
    negative = pd.read_csv(negative_path)
    if not positive.columns.equals(negative.columns):
        raise ValueError(f"Positive and negative feature columns differ: {positive_path.name}.")
    positive.insert(0, "simulation_id", [f"positive_{index}" for index in range(len(positive))])
    negative.insert(0, "simulation_id", [f"negative_{index}" for index in range(len(negative))])
    return pd.concat([positive, negative], ignore_index=True), set(positive["simulation_id"])


def load_tables(feature_dir, methods):
    """Match generated rows across methods before filtering incomplete features."""
    tables, shared, labels = {}, None, None
    for method in methods:
        table, positive_ids = _labeled_table(
            feature_dir / "positive" / f"{method}.csv",
            feature_dir / "negative" / f"{method}.csv",
        )
        if labels is not None and (labels != positive_ids or len(table) != len(shared)):
            raise ValueError("Simulation methods must contain the same positive and negative rows.")
        labels = positive_ids
        values = table.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
        valid = np.isfinite(values).all(axis=1).to_numpy()
        shared = valid if shared is None else shared & valid
        values.insert(0, "simulation_id", table.simulation_id)
        tables[method] = values
    if not shared.any():
        raise ValueError("No complete simulation rows are shared by the selected methods.")
    tables = {method: table.loc[shared].reset_index(drop=True) for method, table in tables.items()}
    return tables, labels


def write_results(results, output_dir, prefix):
    """Derive curves from the same held-out scores used for summary metrics."""
    summaries, predictions, pr_curves, roc_curves = [], [], [], []
    for summary, predicted in results:
        columns = [column for column in ("method", "model", "epsilon_std") if column in summary]
        metadata = summary.iloc[0][columns].to_dict()
        precision, recall, thresholds = precision_recall_curve(predicted.target, predicted.score)
        pr_curves.append(pd.DataFrame({
            "precision": precision, "recall": recall, "threshold": np.r_[thresholds, np.nan],
            "average_precision": summary.average_precision.iloc[0],
            "prevalence": predicted.target.mean(),
        }).assign(**metadata))
        fpr, tpr, thresholds = roc_curve(predicted.target, predicted.score)
        roc_curves.append(pd.DataFrame({
            "false_positive_rate": fpr, "true_positive_rate": tpr, "threshold": thresholds,
            "roc_auc": summary.roc_auc.iloc[0],
        }).assign(**metadata))
        summaries.append(summary)
        predictions.append(predicted)
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix, tables in {"benchmarks": summaries, "cv_predictions": predictions,
                           "pr_curve": pr_curves, "roc_curve": roc_curves}.items():
        path = output_dir / f"simulation_{prefix}{suffix}.csv"
        pd.concat(tables, ignore_index=True).to_csv(path, index=False)
        print(f"simulation: {path}", flush=True)


def main() -> None:
    args = parse_args()
    methods = list(dict.fromkeys(args.methods))
    needed = list(dict.fromkeys([*methods, *(["pdm"] if args.models else [])]))
    manifest_path = args.feature_dir / "simulation_manifest.json"
    epsilons = (
        json.loads(manifest_path.read_text())["epsilon_stds"] if manifest_path.exists() else [None]
    )
    datasets = []
    for epsilon in epsilons:
        directory = args.feature_dir
        if len(epsilons) > 1:
            directory = directory / f"epsilon_{epsilon}"
        tables, labels = load_tables(directory, needed)
        datasets.append((epsilon, tables, labels))
    # Retain identical rows and fold assignments across methods AND noise levels.
    shared_ids = set.intersection(*(set(tables[needed[0]].simulation_id)
                                    for _, tables, _ in datasets))
    ids = [value for value in datasets[0][1][needed[0]].simulation_id if value in shared_ids]
    if not ids:
        raise ValueError("No complete simulation rows are shared across noise levels.")
    print(f"Shared simulation rows per noise level: {len(ids)}", flush=True)
    comparisons = {"": [(method, args.model) for method in methods]}
    if args.models:
        comparisons["model_"] = [("pdm", model) for model in dict.fromkeys(args.models)]
    results = {prefix: [] for prefix in comparisons}
    for epsilon, tables, labels in datasets:
        tables = {name: table.set_index("simulation_id").loc[ids].reset_index()
                  for name, table in tables.items()}
        cache = {}
        for prefix, settings in comparisons.items():
            for method, model in settings:
                if (method, model) not in cache:
                    print(f"Evaluating epsilon={epsilon}: {method} / {model}", flush=True)
                    summary, predicted = evaluate_labeled_features(
                        tables[method], labels, model=model, n_splits=args.n_splits,
                        negative_labels=set(ids) - labels,
                        feature_selection=method == "pdm",
                        n_features=args.features_left if method == "pdm" else None,
                        random_state=args.random_state, return_predictions=True,
                    )
                    metadata = {"method": method.upper(), "model": model,
                                "random_state": args.random_state,
                                "n_splits": predicted.fold.nunique(),
                                "n_features": args.features_left if method == "pdm" else None}
                    if epsilon is not None:
                        metadata["epsilon_std"] = epsilon
                    cache[method, model] = (
                        summary.assign(**metadata), predicted.assign(**metadata)
                    )
                results[prefix].append(cache[method, model])
    for prefix, values in results.items():
        write_results(values, args.results_dir, prefix)


if __name__ == "__main__":
    main()
