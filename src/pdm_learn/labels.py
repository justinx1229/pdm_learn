"""Generic helpers for loading and splitting positive-label sets.

These utilities work with any row identifier, rather than assuming a gene
column or a particular label-file layout.  A *negative* result from
``split_labeled_rows`` means "not in the positive set"; it is often an
unlabeled candidate set rather than a collection of confirmed negatives.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from os import PathLike
from pathlib import Path
from typing import TypeAlias

import pandas as pd

from .datasets import DatasetSpec, matrix_id_column


LabelPaths: TypeAlias = str | Path | PathLike[str] | Iterable[str | Path | PathLike[str]] | None
"""Accepted one-file or many-file inputs for :func:`load_label_set`."""


def _as_sequence(value: object | None) -> tuple[object, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, Path, PathLike)):
        return (value,)
    if isinstance(value, Iterable):
        return tuple(value)
    return (value,)


def _clean_label(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    label = str(value).strip()
    return label or None


def _normalize_labels(values: object | None) -> set[str]:
    return {label for value in _as_sequence(values) if (label := _clean_label(value)) is not None}


def _label_column(dataframe: pd.DataFrame, column: int | str) -> pd.Series:
    if dataframe.shape[1] == 0:
        raise ValueError("A label file must contain at least one column.")
    if isinstance(column, int):
        try:
            return dataframe.iloc[:, column]
        except IndexError as error:
            raise ValueError(f"Label column index {column} is outside the file's columns.") from error
    if column in dataframe.columns:
        return dataframe[column]

    matches = [candidate for candidate in dataframe.columns if str(candidate).strip() == column.strip()]
    if len(matches) == 1:
        return dataframe[matches[0]]
    if len(matches) > 1:
        raise ValueError(f"Label column {column!r} is ambiguous after trimming whitespace.")
    available = ", ".join(repr(candidate) for candidate in dataframe.columns)
    raise ValueError(f"Label column {column!r} was not found. Available columns: {available}.")


def load_label_set(
    paths: LabelPaths = None,
    *,
    labels: Iterable[object] | object | None = None,
    column: int | str = 0,
    header: int | None = None,
    read_csv_kwargs: Mapping[str, object] | None = None,
) -> set[str]:
    """Load a normalized positive-label set from files and/or in-memory labels.

    Parameters
    ----------
    paths:
        One CSV path or an iterable of CSV paths.  Files are combined by set
        union.  The default ``header=None`` is suitable for simple one-column
        label lists; pass ``header=0`` and a named ``column`` for headered
        files.
    labels:
        Optional individual labels to add without creating a temporary file.
        A single string is treated as one label, not an iterable of letters.
    column:
        Integer position or header name of the column containing labels.
    header:
        Header setting passed to :func:`pandas.read_csv`.
    read_csv_kwargs:
        Additional arguments passed to :func:`pandas.read_csv` for every file.

    Empty values and surrounding whitespace are discarded.  Supplying neither
    files nor labels returns an empty set.
    """
    kwargs = dict(read_csv_kwargs or {})
    if "header" in kwargs:
        raise ValueError("Pass header directly to load_label_set, not in read_csv_kwargs.")

    result = _normalize_labels(labels)
    for path in _as_sequence(paths):
        dataframe = pd.read_csv(path, header=header, **kwargs)
        result.update(_normalize_labels(_label_column(dataframe, column).tolist()))
    return result


def split_labeled_rows(
    table: pd.DataFrame,
    positive_labels: Iterable[object] | object,
    *,
    id_column: str | None = None,
    drop_missing: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a feature table into positive and non-positive rows.

    ``id_column`` selects the row identifier; when omitted, the table's first
    column is used.  The second output contains rows absent from the supplied
    positive set, so it can represent unknown candidates as well as explicit
    negatives.  When ``drop_missing`` is true (the default), rows with a
    missing value in any column are removed before splitting.
    """
    if not isinstance(table, pd.DataFrame):
        raise TypeError("table must be a pandas DataFrame.")
    positive_set = _normalize_labels(positive_labels)
    if not positive_set:
        raise ValueError("Provide at least one positive label.")

    selector = DatasetSpec("labels", normalize=None, id_column=id_column)
    resolved_id_column = matrix_id_column(table, selector)
    clean = table.dropna(axis=0, how="any").copy() if drop_missing else table.copy()
    identifiers = clean[resolved_id_column].astype("string").str.strip()
    if identifiers.isna().any() or identifiers.eq("").any():
        raise ValueError("Feature-table identifiers must not be missing or blank.")

    positive_mask = identifiers.isin(positive_set)
    positive = clean.loc[positive_mask].reset_index(drop=True)
    negative = clean.loc[~positive_mask].reset_index(drop=True)
    return positive, negative


def balance_labeled_rows(
    positive: pd.DataFrame,
    negative: pd.DataFrame,
    *,
    n_per_class: int | None = None,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sample equal-sized positive and negative tables without replacement.

    By default, the smaller input determines the size of both outputs.  Set
    ``n_per_class`` to request a smaller balanced sample.  Inputs must have the
    same columns, and both must contain at least one row.
    """
    if not isinstance(positive, pd.DataFrame) or not isinstance(negative, pd.DataFrame):
        raise TypeError("positive and negative must both be pandas DataFrames.")
    if list(positive.columns) != list(negative.columns):
        raise ValueError("Positive and negative tables must have identical columns.")
    if positive.empty or negative.empty:
        raise ValueError("Balancing requires at least one positive and one negative row.")

    available = min(len(positive), len(negative))
    if n_per_class is None:
        count = available
    elif isinstance(n_per_class, int) and not isinstance(n_per_class, bool) and n_per_class > 0:
        count = n_per_class
    else:
        raise ValueError("n_per_class must be a positive integer or None.")
    if count > available:
        raise ValueError(
            f"Cannot sample {count} rows per class; the smaller class contains only {available} rows."
        )

    def sample(frame: pd.DataFrame) -> pd.DataFrame:
        if len(frame) == count:
            return frame.copy().reset_index(drop=True)
        return frame.sample(n=count, replace=False, random_state=random_state).reset_index(drop=True)

    return sample(positive), sample(negative)


__all__ = [
    "LabelPaths",
    "balance_labeled_rows",
    "load_label_set",
    "split_labeled_rows",
]
