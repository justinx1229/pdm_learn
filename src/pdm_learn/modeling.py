"""Ranking, cross-validation, and feature selection for feature tables."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm.auto import tqdm

MODEL_NAMES = ("logistic_regression", "random_forest", "gradient_boosting", "svm")
"""Classifier names accepted by the public ranking and evaluation functions."""


def _resolve_column(table: pd.DataFrame, requested: str | None) -> object:
    if table.shape[1] < 2:
        raise ValueError("A feature table needs an identifier and at least one feature column.")
    if requested is None:
        return table.columns[0]
    if requested in table.columns:
        return requested
    matches = [column for column in table.columns if str(column).strip() == requested.strip()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Identifier column {requested!r} is ambiguous after trimming whitespace.")
    raise ValueError(f"Identifier column {requested!r} was not found.")


def _normalize_labels(labels: Iterable[object]) -> set[str]:
    normalized = {
        str(value).strip()
        for value in labels
        if value is not None and not pd.isna(value) and str(value).strip()
    }
    if not normalized:
        raise ValueError("Provide at least one positive label.")
    return normalized


def _numeric_features(
    table: pd.DataFrame,
    identifier: object,
    feature_columns: Sequence[object] | None,
) -> tuple[list[object], np.ndarray]:
    if feature_columns is None:
        candidates = [column for column in table.columns if column != identifier]
    else:
        candidates = list(feature_columns)
        unknown = [column for column in candidates if column not in table.columns]
        if unknown:
            raise ValueError(f"Feature columns were not found: {unknown[:5]!r}.")
        if identifier in candidates:
            raise ValueError("The identifier column cannot also be a feature column.")
    if not candidates:
        raise ValueError("Provide at least one feature column.")

    converted = table.loc[:, candidates].apply(pd.to_numeric, errors="coerce")
    if feature_columns is None:
        numeric_columns = [column for column in candidates if converted[column].notna().any()]
        if not numeric_columns:
            raise ValueError("The feature table does not contain numeric feature columns.")
        converted = converted.loc[:, numeric_columns]
        candidates = numeric_columns
    values = converted.to_numpy(dtype=float)
    return candidates, values


def _labeled_matrix(
    feature_table: pd.DataFrame,
    positive_labels: Iterable[object],
    *,
    id_column: str | None,
    feature_columns: Sequence[object] | None,
) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, list[object]]:
    if not isinstance(feature_table, pd.DataFrame):
        raise TypeError("feature_table must be a pandas DataFrame.")
    identifier = _resolve_column(feature_table, id_column)
    selected_columns, values = _numeric_features(feature_table, identifier, feature_columns)
    identifiers = feature_table[identifier].astype("string").str.strip()
    valid = identifiers.notna() & identifiers.ne("") & np.isfinite(values).all(axis=1)
    if not valid.any():
        raise ValueError("No complete rows remain after selecting numeric features.")
    labels = _normalize_labels(positive_labels)
    values = values[valid.to_numpy()]
    clean_ids = identifiers.loc[valid].astype(str).to_numpy()
    positive = np.isin(clean_ids, tuple(labels))
    return str(identifier), clean_ids, values, positive, selected_columns


def feature_ks_pvalues(positive: np.ndarray, negative: np.ndarray) -> np.ndarray:
    """Return two-sample KS p-values for each feature column."""
    positive_values = np.asarray(positive, dtype=float)
    negative_values = np.asarray(negative, dtype=float)
    if positive_values.ndim != 2 or negative_values.ndim != 2:
        raise ValueError("positive and negative feature values must be two-dimensional.")
    if not len(positive_values) or not len(negative_values):
        raise ValueError("Feature selection requires both positive and negative rows.")
    if positive_values.shape[1] != negative_values.shape[1]:
        raise ValueError("Positive and negative feature matrices must have the same width.")
    return np.asarray(
        [
            ks_2samp(positive_values[:, index], negative_values[:, index]).pvalue
            for index in range(positive_values.shape[1])
        ],
        dtype=float,
    )


def select_features(
    positive: np.ndarray,
    negative: np.ndarray,
    *,
    n_features: int | None = None,
) -> np.ndarray:
    """Select the most class-separating columns by KS-test p-value."""
    pvalues = feature_ks_pvalues(positive, negative)
    if n_features is None:
        count = len(pvalues)
    elif isinstance(n_features, int) and not isinstance(n_features, bool) and n_features > 0:
        count = min(n_features, len(pvalues))
    else:
        raise ValueError("n_features must be a positive integer or None.")
    return np.sort(np.argsort(pvalues)[:count])


def _build_classifier(model: str, random_state: int):
    if model == "logistic_regression":
        return make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1_000, random_state=random_state)
        )
    if model == "random_forest":
        return RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=1)
    if model == "gradient_boosting":
        return GradientBoostingClassifier(random_state=random_state)
    if model == "svm":
        return make_pipeline(StandardScaler(), SVC(probability=True, random_state=random_state))
    allowed = ", ".join(MODEL_NAMES)
    raise ValueError(f"Unsupported model {model!r}; choose one of: {allowed}.")


def _positive_scores(classifier, values: np.ndarray) -> np.ndarray:
    probabilities = classifier.predict_proba(values)
    classes = np.asarray(classifier.classes_)
    return probabilities[:, np.flatnonzero(classes == 1)[0]]


def rank_candidates(
    feature_table: pd.DataFrame,
    positive_labels: Iterable[object],
    *,
    id_column: str | None = None,
    feature_columns: Sequence[object] | None = None,
    model: str = "logistic_regression",
    trials: int = 100,
    feature_selection: bool = True,
    n_features: int | None = None,
    random_state: int = 42,
    progress: bool = False,
) -> pd.DataFrame:
    """Rank non-positive rows using repeated balanced positive/unlabeled fits.

    The known-positive rows train the positive class. Each trial samples an
    equally sized subset of the remaining rows as provisional negatives, then
    averages predicted positive probabilities across all candidate rows.
    """
    if isinstance(trials, bool) or not isinstance(trials, int) or trials < 1:
        raise ValueError("trials must be a positive integer.")
    identifier, identifiers, values, positive_mask, _ = _labeled_matrix(
        feature_table,
        positive_labels,
        id_column=id_column,
        feature_columns=feature_columns,
    )
    positive_values, candidate_values = values[positive_mask], values[~positive_mask]
    candidate_ids = identifiers[~positive_mask]
    if not len(positive_values) or not len(candidate_values):
        raise ValueError(
            "Ranking requires at least one positive row and one unlabeled candidate row."
        )

    rng = np.random.default_rng(random_state)
    scores = np.zeros(len(candidate_values), dtype=float)
    sample_size = min(len(positive_values), len(candidate_values))
    for trial in tqdm(range(trials), desc="Candidate ranking", disable=not progress):
        positive_index = rng.choice(len(positive_values), size=sample_size, replace=False)
        negative_index = rng.choice(len(candidate_values), size=sample_size, replace=False)
        train_positive = positive_values[positive_index]
        train_negative = candidate_values[negative_index]
        selected = (
            select_features(train_positive, train_negative, n_features=n_features)
            if feature_selection
            else np.arange(values.shape[1])
        )
        train_values = np.vstack([train_positive[:, selected], train_negative[:, selected]])
        train_labels = np.concatenate(
            [np.ones(sample_size, dtype=int), np.zeros(sample_size, dtype=int)]
        )
        classifier = _build_classifier(model, random_state + trial).fit(train_values, train_labels)
        scores += _positive_scores(classifier, candidate_values[:, selected])
    scores /= trials
    return (
        pd.DataFrame({identifier: candidate_ids, "score": scores})
        .sort_values("score", ascending=False, kind="stable")
        .reset_index(drop=True)
    )


def rank_unlabeled(
    feature_table: pd.DataFrame,
    positive_labels: Iterable[object],
    **kwargs: object,
) -> pd.DataFrame:
    """Alias for :func:`rank_candidates` that emphasizes unlabeled rows."""
    return rank_candidates(feature_table, positive_labels, **kwargs)


def evaluate_labeled_features(
    feature_tables: Mapping[str, pd.DataFrame] | pd.DataFrame,
    positive_labels: Iterable[object],
    *,
    id_column: str | None = None,
    feature_columns: Sequence[object] | Mapping[str, Sequence[object]] | None = None,
    model: str = "logistic_regression",
    n_splits: int = 5,
    feature_selection: bool = False,
    n_features: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Cross-validate one or more labeled feature tables.

    Returns ROC AUC and average precision for each table. Non-numeric metadata
    is ignored by default; pass ``feature_columns`` to select features
    explicitly, especially for pair-table data.
    """
    positive_label_set = _normalize_labels(positive_labels)
    tables = (
        {"features": feature_tables}
        if isinstance(feature_tables, pd.DataFrame)
        else dict(feature_tables)
    )
    rows: list[dict[str, object]] = []
    for name, table in tables.items():
        selected_columns = (
            feature_columns.get(name) if isinstance(feature_columns, Mapping) else feature_columns
        )
        _, _, values, positive_mask, used_columns = _labeled_matrix(
            table,
            positive_label_set,
            id_column=id_column,
            feature_columns=selected_columns,
        )
        targets = positive_mask.astype(int)
        positives, negatives = int(targets.sum()), int((targets == 0).sum())
        row: dict[str, object] = {
            "method": str(name),
            "positive_rows": positives,
            "negative_rows": negatives,
            "feature_columns": len(used_columns),
            "roc_auc": np.nan,
            "average_precision": np.nan,
        }
        folds = min(n_splits, positives, negatives)
        if folds >= 2:
            scores = np.empty(len(values), dtype=float)
            splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
            for fold, (train_index, test_index) in enumerate(splitter.split(values, targets)):
                train_values, test_values = values[train_index], values[test_index]
                train_labels = targets[train_index]
                if feature_selection:
                    selected = select_features(
                        train_values[train_labels == 1],
                        train_values[train_labels == 0],
                        n_features=n_features,
                    )
                else:
                    selected = np.arange(values.shape[1])
                classifier = _build_classifier(model, random_state + fold).fit(
                    train_values[:, selected], train_labels
                )
                scores[test_index] = _positive_scores(classifier, test_values[:, selected])
            row["roc_auc"] = float(roc_auc_score(targets, scores))
            row["average_precision"] = float(average_precision_score(targets, scores))
        rows.append(row)
    return pd.DataFrame(rows)


__all__ = [
    "MODEL_NAMES",
    "evaluate_labeled_features",
    "feature_ks_pvalues",
    "rank_candidates",
    "rank_unlabeled",
    "select_features",
]
