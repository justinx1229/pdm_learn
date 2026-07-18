"""Generic loading, preparation, and alignment for entity-by-sample matrices.

The feature builders operate on tables with one entity identifier and one or
more sample columns.  This module deliberately does not assign a biological
meaning to an entity: identifiers can be genes, proteins, compounds, or any
other row key.

``DatasetSpec`` describes how one matrix should be prepared.  In particular,
the identifier does not need to be the first input column; set ``id_column``
when it is elsewhere.  Prepared matrices retain that identifier's original
header while placing it first, which lets downstream code use positional
sample columns without imposing a global name such as ``"gene name"``.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass
from math import isfinite
from os import PathLike
from pathlib import Path
from typing import Literal, TypeAlias

import pandas as pd


DatasetKind: TypeAlias = Literal["continuous", "discrete", "binary"]
"""Supported value types for a matrix."""

Normalization: TypeAlias = Literal["row_center"] | None
"""Supported matrix normalizations."""

DatasetSpecs: TypeAlias = Mapping[str, "DatasetSpec"] | Iterable["DatasetSpec"] | None
"""Accepted forms for a collection of matrix specifications."""

_VALID_KINDS = frozenset({"continuous", "discrete", "binary"})


@dataclass(frozen=True)
class DatasetSpec:
    """Metadata and preparation rules for one entity-by-sample matrix.

    Parameters
    ----------
    name:
        Name used as the matrix's key in a dataset mapping.
    kind:
        ``"continuous"``, ``"discrete"``, or ``"binary"``.  The value is
        metadata for feature construction; preparation leaves valid numeric
        values unchanged.
    levels:
        Optional numeric levels for a discrete or binary matrix.
    normalize:
        ``"row_center"`` (the default) subtracts each row's mean from its
        numeric sample values.  Pass ``None`` to preserve those values.
    id_column:
        Header of the entity identifier column.  If omitted, the first input
        column is used.  The header is preserved in prepared output.

    ``True`` and ``False`` are accepted as compatibility spellings for
    ``"row_center"`` and ``None`` respectively; new code should use the
    explicit values above.
    """

    name: str
    kind: DatasetKind = "continuous"
    levels: tuple[float, ...] | None = None
    normalize: Normalization | bool = "row_center"
    id_column: str | None = None

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("Dataset names must not be empty.")
        object.__setattr__(self, "name", name)

        kind = str(self.kind).strip().lower()
        if kind not in _VALID_KINDS:
            allowed = ", ".join(sorted(_VALID_KINDS))
            raise ValueError(f"Unsupported dataset kind {self.kind!r}; expected one of: {allowed}.")
        object.__setattr__(self, "kind", kind)

        if self.levels is not None:
            try:
                levels = tuple(float(value) for value in self.levels)
            except (TypeError, ValueError) as error:
                raise ValueError("Dataset levels must be numeric.") from error
            if not levels:
                raise ValueError("Dataset levels must be non-empty when provided.")
            if not all(isfinite(value) for value in levels):
                raise ValueError("Dataset levels must be finite.")
            if len(set(levels)) != len(levels):
                raise ValueError("Dataset levels must not contain duplicates.")
            object.__setattr__(self, "levels", levels)

        normalization: Normalization
        if self.normalize is True:
            normalization = "row_center"
        elif self.normalize is False or self.normalize is None:
            normalization = None
        elif self.normalize == "row_center":
            normalization = "row_center"
        else:
            raise ValueError("normalize must be 'row_center' or None.")
        object.__setattr__(self, "normalize", normalization)

        if self.id_column is not None:
            id_column = str(self.id_column).strip()
            if not id_column:
                raise ValueError("id_column must not be empty when provided.")
            object.__setattr__(self, "id_column", id_column)

    @property
    def is_continuous(self) -> bool:
        """Whether this specification describes continuous values."""
        return self.kind == "continuous"

    @property
    def is_discrete(self) -> bool:
        """Whether this specification describes discrete or binary values."""
        return not self.is_continuous


def _normalized_name(value: Hashable) -> str:
    """Return the comparison form used for headers and identifiers."""
    return str(value).strip()


def _column_lookup(dataframe: pd.DataFrame) -> dict[str, Hashable]:
    """Map whitespace-normalized headers to their original labels."""
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")
    if dataframe.shape[1] == 0:
        raise ValueError("A matrix must contain at least an identifier column.")

    lookup: dict[str, Hashable] = {}
    for column in dataframe.columns:
        normalized = _normalized_name(column)
        if not normalized:
            raise ValueError("Matrix column headers must not be empty.")
        if normalized in lookup:
            raise ValueError(
                "Matrix column headers must be unique after trimming whitespace: "
                f"{normalized!r} is duplicated."
            )
        lookup[normalized] = column
    return lookup


def matrix_id_column(dataframe: pd.DataFrame, spec: DatasetSpec | None = None) -> Hashable:
    """Return the actual identifier header selected for ``dataframe``.

    If ``spec.id_column`` is supplied, it is matched exactly or after trimming
    surrounding whitespace.  Otherwise the first input column is selected.
    The returned object is the table's original header, rather than a fixed
    canonical name.
    """
    lookup = _column_lookup(dataframe)
    if spec is None or spec.id_column is None:
        return dataframe.columns[0]

    if spec.id_column in dataframe.columns:
        return spec.id_column
    try:
        return lookup[spec.id_column]
    except KeyError as error:
        available = ", ".join(repr(column) for column in dataframe.columns)
        raise ValueError(
            f"Identifier column {spec.id_column!r} was not found. Available columns: {available}."
        ) from error


def matrix_sample_columns(dataframe: pd.DataFrame, spec: DatasetSpec | None = None) -> tuple[Hashable, ...]:
    """Return all non-identifier column headers in their input order."""
    id_column = matrix_id_column(dataframe, spec)
    return tuple(column for column in dataframe.columns if column != id_column)


def prepare_matrix(dataframe: pd.DataFrame, spec: DatasetSpec) -> pd.DataFrame:
    """Standardize one matrix for feature construction.

    The identifier column is selected by ``spec.id_column`` (or the first
    input column), stripped of surrounding whitespace, and placed first.  Its
    header is retained.  Every sample column is converted with
    :func:`pandas.to_numeric`, turning non-numeric values into ``NaN``.  Row
    centering is then applied when requested by ``spec.normalize``.

    Blank or duplicate entity identifiers are rejected because they make row
    alignment ambiguous.  Missing numeric values are retained for the feature
    configuration to handle.
    """
    if not isinstance(spec, DatasetSpec):
        raise TypeError("spec must be a DatasetSpec.")

    id_column = matrix_id_column(dataframe, spec)
    sample_columns = matrix_sample_columns(dataframe, spec)
    if not sample_columns:
        raise ValueError(f"Dataset {spec.name!r} must contain at least one sample column.")

    output = dataframe.loc[:, [id_column, *sample_columns]].copy()
    identifiers = output[id_column].astype("string").str.strip()
    if identifiers.isna().any() or identifiers.eq("").any():
        raise ValueError(f"Dataset {spec.name!r} contains a missing or blank entity identifier.")
    if identifiers.duplicated().any():
        examples = identifiers.loc[identifiers.duplicated(keep=False)].unique()[:5].tolist()
        raise ValueError(
            f"Dataset {spec.name!r} contains duplicate entity identifiers after trimming: {examples!r}."
        )

    numeric = output.loc[:, sample_columns].apply(pd.to_numeric, errors="coerce")
    if spec.normalize == "row_center":
        numeric = numeric.sub(numeric.mean(axis=1), axis=0)

    prepared = pd.concat([identifiers.rename(id_column), numeric], axis=1)
    prepared.index = dataframe.index.copy()
    return prepared


def _clean_dataset_name(value: object) -> str:
    name = str(value).strip()
    if not name:
        raise ValueError("Dataset names must not be empty.")
    return name


def resolve_dataset_specs(names: Iterable[object], specs: DatasetSpecs = None) -> dict[str, DatasetSpec]:
    """Return one :class:`DatasetSpec` for each named input matrix.

    ``specs`` may be a mapping keyed by dataset name or an iterable of specs.
    Omitted names receive the default continuous, row-centered specification.
    Supplying a spec for an unknown matrix is an error, which catches spelling
    mistakes before feature generation starts.
    """
    clean_names = tuple(_clean_dataset_name(name) for name in names)
    if len(set(clean_names)) != len(clean_names):
        raise ValueError("Dataset names must be unique after trimming whitespace.")

    if specs is None:
        supplied: dict[str, DatasetSpec] = {}
    elif isinstance(specs, Mapping):
        supplied = {}
        for mapping_name, spec in specs.items():
            name = _clean_dataset_name(mapping_name)
            if not isinstance(spec, DatasetSpec):
                raise TypeError(f"Specification for {name!r} must be a DatasetSpec.")
            if spec.name != name:
                raise ValueError(
                    f"Specification key {name!r} does not match DatasetSpec.name {spec.name!r}."
                )
            if name in supplied:
                raise ValueError(f"Duplicate specification for dataset {name!r}.")
            supplied[name] = spec
    else:
        supplied = {}
        for spec in specs:
            if not isinstance(spec, DatasetSpec):
                raise TypeError("Each dataset specification must be a DatasetSpec.")
            if spec.name in supplied:
                raise ValueError(f"Duplicate specification for dataset {spec.name!r}.")
            supplied[spec.name] = spec

    unknown = sorted(set(supplied) - set(clean_names))
    if unknown:
        raise ValueError(f"Specifications were provided for unknown datasets: {', '.join(unknown)}.")
    return {name: supplied.get(name, DatasetSpec(name)) for name in clean_names}


def load_matrices(
    paths: Mapping[str, str | Path | PathLike[str]],
    specs: DatasetSpecs = None,
    *,
    read_csv_kwargs: Mapping[str, object] | None = None,
) -> dict[str, pd.DataFrame]:
    """Load and prepare a mapping of named CSV matrices.

    Parameters
    ----------
    paths:
        Dataset names mapped to CSV paths.
    specs:
        Optional mapping or iterable of :class:`DatasetSpec` objects.  When
        omitted, matrices are cleaned (identifier first, numeric samples) but
        not normalized.  This makes the common two-step flow safe:
        load files first, then provide feature-specific specs to
        :func:`pdm_learn.features.build_features`.
    read_csv_kwargs:
        Keyword arguments forwarded unchanged to :func:`pandas.read_csv`.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Prepared matrices keyed by their trimmed dataset names.
    """
    if not isinstance(paths, Mapping) or not paths:
        raise ValueError("Provide at least one named input matrix path.")

    clean_paths: dict[str, str | Path | PathLike[str]] = {}
    for raw_name, path in paths.items():
        name = _clean_dataset_name(raw_name)
        if name in clean_paths:
            raise ValueError(f"Duplicate matrix path for dataset {name!r}.")
        clean_paths[name] = path

    # Loading should not silently apply the continuous default to an unknown
    # binary/discrete file.  If the caller supplies no metadata, perform only
    # lossless structural preparation; the feature builder later applies its
    # documented defaults or explicit DatasetSpecs.
    resolved_specs = (
        resolve_dataset_specs(clean_paths, specs)
        if specs is not None
        else {name: DatasetSpec(name, normalize=None) for name in clean_paths}
    )
    kwargs = dict(read_csv_kwargs or {})
    if "filepath_or_buffer" in kwargs:
        raise ValueError("read_csv_kwargs must not include filepath_or_buffer.")

    return {
        name: prepare_matrix(pd.read_csv(path, **kwargs), resolved_specs[name])
        for name, path in clean_paths.items()
    }


def align_matrices(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_spec: DatasetSpec | None = None,
    right_spec: DatasetSpec | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare and inner-align two matrices by entity and sample identifiers.

    Rows follow the left matrix's order; shared sample columns also follow the
    left matrix's order.  Entity headers remain distinct (for example,
    ``"compound_id"`` and ``"protein_id"``), while the right output's sample
    headers are renamed to the corresponding left headers so both outputs are
    positionally and label-wise aligned.

    Both inputs are passed through :func:`prepare_matrix`, so their supplied
    specifications control identifier selection and normalization.  A clear
    error is raised when the matrices have no shared entities or samples.
    """
    left_spec = left_spec or DatasetSpec("left")
    right_spec = right_spec or DatasetSpec("right")
    left_prepared = prepare_matrix(left, left_spec)
    right_prepared = prepare_matrix(right, right_spec)

    left_id = matrix_id_column(left_prepared, left_spec)
    right_id = matrix_id_column(right_prepared, right_spec)
    left_samples = matrix_sample_columns(left_prepared, left_spec)
    right_samples = matrix_sample_columns(right_prepared, right_spec)
    right_sample_lookup = {_normalized_name(column): column for column in right_samples}
    shared_sample_keys = [
        _normalized_name(column)
        for column in left_samples
        if _normalized_name(column) in right_sample_lookup
    ]
    if not shared_sample_keys:
        raise ValueError("Matrices do not share any sample columns.")

    left_selected_samples = [
        next(column for column in left_samples if _normalized_name(column) == key)
        for key in shared_sample_keys
    ]
    right_selected_samples = [right_sample_lookup[key] for key in shared_sample_keys]

    right_by_id = right_prepared.set_index(right_id, drop=False)
    left_rows = left_prepared.loc[
        left_prepared[left_id].isin(right_by_id.index),
        [left_id, *left_selected_samples],
    ].copy()
    if left_rows.empty:
        raise ValueError("Matrices do not share any entity identifiers.")

    entity_order = left_rows[left_id].tolist()
    right_rows = right_by_id.loc[entity_order, [right_id, *right_selected_samples]].copy()
    left_rows = left_rows.reset_index(drop=True)
    right_rows = right_rows.reset_index(drop=True)

    output_right_headers = [right_id, *left_selected_samples]
    normalized_output_headers = [_normalized_name(column) for column in output_right_headers]
    if len(set(normalized_output_headers)) != len(normalized_output_headers):
        raise ValueError(
            "Aligned right matrix would have duplicate headers after matching sample columns."
        )
    right_rows.columns = output_right_headers
    return left_rows, right_rows


__all__ = [
    "DatasetKind",
    "DatasetSpec",
    "DatasetSpecs",
    "Normalization",
    "align_matrices",
    "load_matrices",
    "matrix_id_column",
    "matrix_sample_columns",
    "prepare_matrix",
    "resolve_dataset_specs",
]
