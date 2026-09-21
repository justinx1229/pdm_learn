"""Ranking, cross-validation, and feature selection for feature tables."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
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
) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, list[object], np.ndarray]:
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
    return str(identifier), clean_ids, values, positive, selected_columns, valid.to_numpy()


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

    Each trial samples provisional negatives from the unlabeled candidates,
    fits one balanced classifier, and scores only the remaining candidates.
    Feature selection also uses only the sampled training rows. All positives
    are used unless balancing requires subsampling them to leave at least one
    candidate out of training.

    ``score`` averages only the out-of-bag predictions for that candidate;
    ``score_trials`` counts those predictions and may differ between rows.
    Unscored candidates have a NaN score, a zero count, and sort last. Increase
    trials for better coverage. Scores are ranking signals, not calibrated
    probabilities of true positive status under positive/unlabeled sampling.
    """
    if isinstance(trials, bool) or not isinstance(trials, int) or trials < 1:
        raise ValueError("trials must be a positive integer.")
    identifier, identifiers, values, positive_mask, _, _ = _labeled_matrix(
        feature_table,
        positive_labels,
        id_column=id_column,
        feature_columns=feature_columns,
    )
    positive_values, candidate_values = values[positive_mask], values[~positive_mask]
    candidate_ids = identifiers[~positive_mask]
    if not len(positive_values) or len(candidate_values) < 2:
        raise ValueError(
            "Ranking requires at least one positive row and two unlabeled candidate rows."
        )

    rng = np.random.default_rng(random_state)
    scores = np.zeros(len(candidate_values), dtype=float)
    score_counts = np.zeros(len(candidate_values), dtype=int)
    # Keep training balanced while leaving at least one candidate to score.
    sample_size = min(len(positive_values), len(candidate_values) - 1)
    candidate_index = np.arange(len(candidate_values))
    for trial in tqdm(range(trials), desc="Candidate ranking", disable=not progress):
        positive_index = rng.choice(len(positive_values), size=sample_size, replace=False)
        negative_index = rng.choice(len(candidate_values), size=sample_size, replace=False)
        held_out_index = np.setdiff1d(candidate_index, negative_index, assume_unique=True)
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
        scores[held_out_index] += _positive_scores(
            classifier, candidate_values[held_out_index][:, selected]
        )
        score_counts[held_out_index] += 1
    scores = np.divide(
        scores,
        score_counts,
        out=np.full(len(candidate_values), np.nan, dtype=float),
        where=score_counts > 0,
    )
    return (
        pd.DataFrame(
            {identifier: candidate_ids, "score": scores, "score_trials": score_counts}
        )
        .sort_values("score", ascending=False, kind="stable", na_position="last")
        .reset_index(drop=True)
    )


def rank_unlabeled(
    feature_table: pd.DataFrame,
    positive_labels: Iterable[object],
    **kwargs: object,
) -> pd.DataFrame:
    """Alias for :func:`rank_candidates` that emphasizes unlabeled rows."""
    return rank_candidates(feature_table, positive_labels, **kwargs)


def loocv_rank_recovery(
    feature_table: pd.DataFrame,
    positive_labels: Iterable[object],
    *,
    id_column: str | None = None,
    feature_columns: Sequence[object] | None = None,
    model: str = "logistic_regression",
    trials: int = 25,
    feature_selection: bool = True,
    n_features: int | None = None,
    random_state: int = 42,
    progress: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a leave-one-positive-out rank-recovery curve and held-out ranks.

    Each known positive is withheld from the positive-label set, ranked with
    :func:`rank_candidates`, and compared with the unlabeled candidates.  The
    returned curve uses the main-branch rank-recovery convention: a threshold
    of zero means the held-out positive was ranked first.  Its ``area`` is the
    normalized area under the cumulative recovery curve.

    This is a positive-unlabeled ranking diagnostic.  It is not a substitute
    for ROC AUC or average precision against confirmed negative controls.
    Every candidate must receive a score; otherwise increase ``trials``.
    """
    identifier, identifiers, values, positive_mask, selected_columns, _ = _labeled_matrix(
        feature_table,
        positive_labels,
        id_column=id_column,
        feature_columns=feature_columns,
    )
    positive_ids = identifiers[positive_mask]
    candidate_count = int((~positive_mask).sum())
    if len(positive_ids) < 2:
        raise ValueError("LOOCV rank recovery requires at least two positive rows.")
    if candidate_count < 1:
        raise ValueError("LOOCV rank recovery requires at least one unlabeled candidate row.")

    clean_table = pd.DataFrame(values, columns=selected_columns)
    clean_table.insert(0, identifier, identifiers)
    positive_set = set(positive_ids.tolist())
    held_out_rows: list[dict[str, object]] = []
    for held_out_index, held_out_id in enumerate(
        tqdm(positive_ids, desc="LOOCV positives", disable=not progress)
    ):
        training_labels = positive_set - {held_out_id}
        ranking = rank_candidates(
            clean_table,
            training_labels,
            id_column=identifier,
            feature_columns=selected_columns,
            model=model,
            trials=trials,
            feature_selection=feature_selection,
            n_features=n_features,
            random_state=random_state + held_out_index * trials,
            progress=False,
        )
        if (ranking["score_trials"] == 0).any():
            raise ValueError(
                "LOOCV rank recovery requires a score for every candidate. Increase trials."
            )
        rank_matches = np.flatnonzero(ranking[identifier].to_numpy() == held_out_id)
        if len(rank_matches) != 1:
            raise RuntimeError(f"Held-out positive {held_out_id!r} was not ranked exactly once.")
        rank = int(rank_matches[0])
        held_out_rows.append(
            {
                identifier: held_out_id,
                "rank": rank,
                "rank_fraction": rank / candidate_count,
                "candidate_count": candidate_count,
                "score": float(ranking.iloc[rank]["score"]),
            }
        )

    held_out = pd.DataFrame(held_out_rows)
    recovered = np.cumsum(
        np.bincount(held_out["rank"].to_numpy(dtype=int), minlength=candidate_count + 1)
    )
    area = float(recovered.sum() / (len(held_out) * (candidate_count + 1)))
    curve = pd.DataFrame(
        {
            "rank_threshold": np.arange(candidate_count + 1),
            "recovered_positives": recovered,
            "recovered_fraction": recovered / len(held_out),
            "loocv_area": area,
            "positive_rows": len(held_out),
            "unlabeled_candidate_rows": candidate_count,
        }
    )
    return curve, held_out


def evaluate_labeled_features(
    feature_tables: Mapping[str, pd.DataFrame] | pd.DataFrame,
    positive_labels: Iterable[object],
    *,
    negative_labels: Iterable[object] | None = None,
    id_column: str | None = None,
    feature_columns: Sequence[object] | Mapping[str, Sequence[object]] | None = None,
    groups: Sequence[object] | Mapping[str, Sequence[object]] | None = None,
    model: str = "logistic_regression",
    n_splits: int = 5,
    feature_selection: bool = False,
    n_features: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Cross-validate one or more labeled feature tables.

    Returns ROC AUC and average precision for each table. When
    ``negative_labels`` is supplied, evaluation uses only rows whose IDs are
    explicitly positive or negative; otherwise every non-positive row is
    treated as negative. ``groups`` enables stratified group cross-validation
    and must provide one group value per input row. Non-numeric metadata is
    ignored by default; pass ``feature_columns`` to select features
    explicitly, especially for pair-table data.
    """
    positive_label_set = _normalize_labels(positive_labels)
    negative_label_set = (
        None if negative_labels is None else _normalize_labels(negative_labels)
    )
    if negative_label_set is not None and positive_label_set & negative_label_set:
        overlap = sorted(positive_label_set & negative_label_set)
        raise ValueError(f"Positive and negative labels overlap: {overlap[:5]!r}.")
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
        _, identifiers, values, positive_mask, used_columns, complete_rows = _labeled_matrix(
            table,
            positive_label_set,
            id_column=id_column,
            feature_columns=selected_columns,
        )
        selected_groups = groups.get(name) if isinstance(groups, Mapping) else groups
        group_values: np.ndarray | None = None
        if selected_groups is not None:
            if isinstance(selected_groups, (str, bytes)):
                raise TypeError("groups must provide one value per feature-table row.")
            group_values = np.asarray(tuple(selected_groups), dtype=object)
            if group_values.ndim != 1 or len(group_values) != len(table):
                raise ValueError(
                    f"groups for {name!r} must contain exactly one value per feature-table row."
                )
            group_values = group_values[complete_rows]

        if negative_label_set is not None:
            negative_mask = np.isin(identifiers, tuple(negative_label_set))
            labeled_mask = positive_mask | negative_mask
            values = values[labeled_mask]
            positive_mask = positive_mask[labeled_mask]
            if group_values is not None:
                group_values = group_values[labeled_mask]
        if group_values is not None and pd.isna(group_values).any():
            raise ValueError(f"groups for {name!r} must not contain missing values.")
        targets = positive_mask.astype(int)
        positives, negatives = int(targets.sum()), int((targets == 0).sum())
        row: dict[str, object] = {
            "method": str(name),
            "positive_rows": positives,
            "negative_rows": negatives,
            "feature_columns": len(used_columns),
            "cv_strategy": (
                "stratified_group_kfold" if group_values is not None else "stratified_kfold"
            ),
            "roc_auc": np.nan,
            "average_precision": np.nan,
        }
        folds = min(n_splits, positives, negatives)
        if group_values is not None:
            positive_groups = np.unique(group_values[targets == 1])
            negative_groups = np.unique(group_values[targets == 0])
            folds = min(folds, len(positive_groups), len(negative_groups))
        if folds >= 2:
            scores = np.empty(len(values), dtype=float)
            splitter = (
                StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=random_state)
                if group_values is not None
                else StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
            )
            split_iterator = (
                splitter.split(values, targets, group_values)
                if group_values is not None
                else splitter.split(values, targets)
            )
            for fold, (train_index, test_index) in enumerate(split_iterator):
                train_values, test_values = values[train_index], values[test_index]
                train_labels = targets[train_index]
                if train_labels.min() == train_labels.max():
                    raise ValueError(
                        "Each cross-validation training fold must contain both classes."
                    )
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
    "loocv_rank_recovery",
    "rank_candidates",
    "rank_unlabeled",
    "select_features",
]
