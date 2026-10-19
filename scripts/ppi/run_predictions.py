"""Rank BioGRID pairs using curated cancer-complex pairs as positive labels."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from pdm_learn.modeling import MODEL_NAMES, evaluate_labeled_features, rank_candidates
from pdm_learn.workflows.ppi import (
    COMPLEX_NAME_COLUMN,
    cancer_complex_pair_groups,
    canonical_pair,
    load_cancer_complexes,
    pdm_feature_columns,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark and rank BioGRID PPI feature rows.")
    parser.add_argument(
        "--features",
        type=Path,
        default=PROJECT_ROOT / "runs" / "ppi" / "features" / "biogrid_pdm.csv",
    )
    parser.add_argument(
        "--complexes", type=Path, default=PROJECT_ROOT / "data" / "cancer_complexes_clean.xlsx"
    )
    parser.add_argument(
        "--results-dir", type=Path, default=PROJECT_ROOT / "runs" / "ppi" / "results"
    )
    parser.add_argument("--model", choices=MODEL_NAMES, default="gradient_boosting")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument(
        "--max-pairs-per-complex", type=int, default=10,
        help="Maximum positive pairs per complex in each ranking trial (default: 10).",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--features-left", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--positive-controls",
        type=Path,
        default=None,
        help="Pickle containing one DataFrame, or a sequence of DataFrames, of positive controls.",
    )
    parser.add_argument(
        "--negative-controls",
        type=Path,
        default=None,
        help="Pickle containing one DataFrame, or a sequence of DataFrames, of negative controls.",
    )
    return parser.parse_args()


def _control_tables(value: object, *, name: str) -> tuple[pd.DataFrame, ...]:
    """Normalize one or many explicitly labelled control feature tables."""
    if isinstance(value, pd.DataFrame):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        tables = tuple(value)
        if tables and all(isinstance(table, pd.DataFrame) for table in tables):
            return tables
    raise TypeError(f"{name} controls must be a DataFrame or a non-empty sequence of DataFrames.")


def _shared_numeric_features(positive: pd.DataFrame, negative: pd.DataFrame) -> list[object]:
    """Return common generated PDM columns, never pair-source metadata."""
    candidates = [column for column in pdm_feature_columns(positive) if column in negative.columns]
    return [
        column
        for column in candidates
        if pd.to_numeric(pd.concat([positive[column], negative[column]]), errors="coerce")
        .notna()
        .any()
    ]


def _control_pair_ids(table: pd.DataFrame) -> pd.Series:
    if {"Gene1", "Gene2"}.issubset(table.columns):
        if table[["Gene1", "Gene2"]].isna().any().any():
            raise ValueError("Control genes must not be missing.")
        ids = pd.Series([
            canonical_pair(left, right)
            for left, right in zip(table["Gene1"], table["Gene2"], strict=True)
        ])
    elif "canonical_pair" in table.columns:
        ids = table["canonical_pair"].astype("string").str.strip().reset_index(drop=True)
    else:
        raise ValueError("Controls must contain canonical_pair or Gene1/Gene2 identifiers.")
    if ids.isna().any() or ids.eq("").any() or ids.duplicated().any():
        raise ValueError("Control pair identifiers must be nonblank and unique within each class.")
    return ids


def _complex_holdout_groups(complex_pairs: Mapping[str, set[str]]) -> dict[str, str]:
    """Keep complexes connected by shared positive pairs in the same fold."""
    parents = {name: name for name in complex_pairs}

    def root(name: str) -> str:
        while parents[name] != name:
            parents[name] = parents[parents[name]]
            name = parents[name]
        return name

    owners: dict[str, str] = {}
    for name, pairs in complex_pairs.items():
        for pair in pairs:
            owner = owners.setdefault(pair, name)
            left, right = sorted((root(name), root(owner)))
            parents[right] = left
    return {name: root(name) for name in complex_pairs}


def evaluate_control_sets(
    positive_controls: object,
    negative_controls: object,
    *,
    model: str,
    n_splits: int,
    n_features: int | None,
    random_state: int,
    complex_pairs: Mapping[str, set[str]] | None = None,
) -> pd.DataFrame:
    """Evaluate only the supplied positive and negative control sets.

    Candidate BioGRID rows are intentionally excluded: an absence from a
    curated complex list is an unknown label, not a labelled control.
    """
    positives = _control_tables(positive_controls, name="positive")
    negatives = _control_tables(negative_controls, name="negative")
    if len(positives) != len(negatives):
        raise ValueError("Positive and negative control collections must have the same length.")

    rows = []
    for control, (positive, negative) in enumerate(zip(positives, negatives, strict=True), start=1):
        feature_columns = _shared_numeric_features(positive, negative)
        if not feature_columns:
            raise ValueError(f"Control set {control} has no shared numeric feature columns.")
        positive_ids = [f"positive:{control}:{index}" for index in range(len(positive))]
        negative_ids = [f"negative:{control}:{index}" for index in range(len(negative))]
        if COMPLEX_NAME_COLUMN not in positive.columns:
            raise ValueError(
                f"Positive control set {control} must contain {COMPLEX_NAME_COLUMN!r} "
                "to hold out whole complexes."
            )
        complex_names = positive[COMPLEX_NAME_COLUMN].astype("string").str.strip()
        if complex_names.isna().any() or complex_names.eq("").any():
            raise ValueError(
                f"Positive control set {control} contains missing or blank complex names."
            )
        positive_pairs, negative_pairs = _control_pair_ids(positive), _control_pair_ids(negative)
        catalogue = complex_pairs
        if catalogue is None:
            catalogue = {}
            for name, pair in zip(complex_names, positive_pairs, strict=True):
                catalogue.setdefault(name, set()).add(pair)
        if any(
            pair not in catalogue.get(name, set())
            for name, pair in zip(complex_names, positive_pairs, strict=True)
        ):
            raise ValueError("Positive controls must belong to their named curated complex.")
        if set(negative_pairs) & set().union(*catalogue.values()):
            raise ValueError("Negative controls must not contain known positive pairs.")
        holdout_groups = _complex_holdout_groups(catalogue)
        positive_groups = [
            f"positive:{control}:{holdout_groups[value]}"
            for value in complex_names
        ]
        groups = [*positive_groups, *negative_ids]
        table = pd.concat(
            [
                positive.loc[:, feature_columns].assign(control_row=positive_ids),
                negative.loc[:, feature_columns].assign(control_row=negative_ids),
            ],
            ignore_index=True,
        )
        table = table.loc[:, ["control_row", *feature_columns]]
        result = evaluate_labeled_features(
            table,
            positive_ids,
            id_column="control_row",
            feature_columns=feature_columns,
            groups=groups,
            model=model,
            n_splits=n_splits,
            feature_selection=True,
            n_features=n_features,
            random_state=random_state + control,
        )
        result.insert(0, "control", control)
        result.loc[:, "method"] = "PDM"
        rows.append(result)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    args = parse_args()
    if (args.positive_controls is None) != (args.negative_controls is None):
        raise ValueError(
            "Pass both --positive-controls and --negative-controls to run a benchmark."
        )
    args.results_dir.mkdir(parents=True, exist_ok=True)
    features = pd.read_csv(args.features, low_memory=False)
    if "canonical_pair" not in features.columns:
        raise ValueError(
            "PPI features must contain the canonical_pair column from make_features.py."
        )
    positive_groups = cancer_complex_pair_groups(load_cancer_complexes(args.complexes))
    labels = set().union(*positive_groups.values())
    feature_columns = pdm_feature_columns(features)
    if not feature_columns:
        raise ValueError("PPI feature table does not contain generated PDM feature columns.")

    ranking = rank_candidates(
        features,
        labels,
        id_column="canonical_pair",
        feature_columns=feature_columns,
        trials=args.trials,
        model=args.model,
        n_features=args.features_left,
        positive_groups=positive_groups,
        max_positive_per_group=args.max_pairs_per_complex,
        random_state=args.random_state,
        progress=True,
    )
    ranking_path = args.results_dir / "ranked_PPI_pairs_cancer.csv"
    ranking.to_csv(ranking_path, index=False)

    print(f"PPI ranking: {ranking_path}")
    if args.positive_controls is None:
        print("Skipping PPI benchmark: positive and negative control sets were not supplied.")
        return

    benchmark = evaluate_control_sets(
        pd.read_pickle(args.positive_controls),
        pd.read_pickle(args.negative_controls),
        model=args.model,
        n_splits=args.n_splits,
        n_features=args.features_left,
        random_state=args.random_state,
        complex_pairs=positive_groups,
    )
    benchmark_path = args.results_dir / "ppi_control_benchmarks.csv"
    benchmark.to_csv(benchmark_path, index=False)
    print(f"PPI control benchmark: {benchmark_path}")


if __name__ == "__main__":
    main()
