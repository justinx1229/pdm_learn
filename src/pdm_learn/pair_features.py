"""Feature builders for tables of ordered entity pairs.

The row-level builders in :mod:`pdm_learn.features` compare the *same* entity
across two matrices.  This module handles the complementary case where each
observation is a pair of entities: for example, a compound and a protein, an
upstream and downstream node, or two proteins in an interaction table.

The module is deliberately free of assay, gene, and file-layout assumptions.
It receives already-loaded matrices and a pair table, preserves the pair
table's metadata, and appends feature columns for requested ordered matrix
pairs.  Each matrix may use its own entity-ID header through
:class:`pdm_learn.datasets.DatasetSpec`.

Example
-------
>>> from pdm_learn.datasets import DatasetSpec
>>> from pdm_learn.features import FeaturePair
>>> from pdm_learn.pair_features import PairFeatureConfig, build_pair_features
>>> tables = build_pair_features(
...     {"response": response, "state": state},
...     pairs,
...     left_id_column="compound",
...     right_id_column="protein",
...     dataset_specs=[
...         DatasetSpec("response", id_column="compound_id"),
...         DatasetSpec("state", kind="binary", normalize=None, id_column="protein_id"),
...     ],
...     pairs=[FeaturePair("response", "state")],
...     methods=("pdm", "pearson"),
...     config=PairFeatureConfig(density_bins=7),
... )

``tables["pdm"]`` and ``tables["pearson"]`` retain all columns from
``pairs``.  A row whose entity IDs cannot be resolved is either retained with
missing generated values or removed according to ``drop_missing``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats
from tqdm.auto import tqdm

from .datasets import DatasetSpec, prepare_matrix, resolve_dataset_specs
from .features import bicor, compute_mutual_information
from .preprocessing import densitymap


STATISTIC_METHODS = ("pearson", "spearman", "mi", "bicor")
"""Scalar association methods accepted by :func:`build_pair_features`."""

PAIR_FEATURE_METHODS = ("pdm", *STATISTIC_METHODS)
"""All feature methods accepted by :func:`build_pair_features`."""


@dataclass(frozen=True)
class PairFeatureConfig:
    """Configuration shared by pair-table feature builders.

    Parameters
    ----------
    pairs:
        Optional ordered matrix pairs.  Entries may be any objects exposing
        ``left`` and ``right`` attributes (such as ``FeaturePair``), or
        two-item sequences of dataset names.  When omitted, every ordered
        pair of supplied matrices is used.
    density_bins:
        Number of centers on a continuous PDM axis.
    log_offset:
        Positive smoothing offset applied before the natural log of a PDM.
        ``"auto"`` and ``None`` use ``1 / (shared_samples + 1)``, matching
        the existing density-map convention.
    drop_missing:
        If true, remove pair-table rows that are missing one or more generated
        features.  Existing metadata columns do not affect this decision.

    Unlike a row-level entity universe, a pair table already defines its
    output universe.  Missing entity IDs are therefore never inferred or
    imputed; they become missing generated features.
    """

    pairs: Sequence[object] | None = None
    density_bins: int = 7
    log_offset: float | Literal["auto"] | None = "auto"
    drop_missing: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.density_bins, bool) or not isinstance(self.density_bins, int) or self.density_bins < 1:
            raise ValueError("density_bins must be a positive integer.")
        if self.pairs is not None:
            if isinstance(self.pairs, (str, bytes)):
                raise TypeError("pairs must be a sequence of matrix-pair specifications.")
            object.__setattr__(self, "pairs", tuple(self.pairs))
        if self.log_offset is not None and self.log_offset != "auto":
            if isinstance(self.log_offset, bool) or not isinstance(self.log_offset, (int, float)):
                raise TypeError("log_offset must be a positive number, None, or 'auto'.")
            if not np.isfinite(self.log_offset) or self.log_offset <= 0:
                raise ValueError("log_offset must be a finite positive number.")
        if not isinstance(self.drop_missing, bool):
            raise TypeError("drop_missing must be a boolean.")


@dataclass(frozen=True)
class _MatrixPair:
    """Validated internal representation of one ordered matrix pair."""

    left: str
    right: str


@dataclass(frozen=True)
class _PairContext:
    """Precomputed data required to build one feature block."""

    pair: _MatrixPair
    left_spec: DatasetSpec
    right_spec: DatasetSpec
    left_rows: Mapping[str, np.ndarray]
    right_rows: Mapping[str, np.ndarray]
    left_scale: float
    right_scale: float
    sigma: float
    left_levels: np.ndarray | None
    right_levels: np.ndarray | None
    sample_count: int


def _clean_dataset_name(value: object) -> str:
    name = str(value).strip()
    if not name:
        raise ValueError("Dataset names must not be empty.")
    return name


def _coerce_matrix_pair(value: object) -> _MatrixPair:
    """Accept local, public, and lightweight pair representations."""
    left = getattr(value, "left", None)
    right = getattr(value, "right", None)
    if left is not None and right is not None:
        return _MatrixPair(_clean_dataset_name(left), _clean_dataset_name(right))

    if isinstance(value, str):
        separator = ":" if ":" in value else ","
        if separator not in value:
            raise ValueError("String matrix pairs must use 'left:right' or 'left,right'.")
        left, right = value.split(separator, maxsplit=1)
        return _MatrixPair(_clean_dataset_name(left), _clean_dataset_name(right))

    try:
        left, right = value  # type: ignore[misc]
    except (TypeError, ValueError) as error:
        raise TypeError("Each matrix pair must provide exactly two dataset names.") from error
    return _MatrixPair(_clean_dataset_name(left), _clean_dataset_name(right))


def _resolve_matrix_pairs(
    datasets: Mapping[str, pd.DataFrame],
    pairs: Sequence[object] | None,
) -> tuple[_MatrixPair, ...]:
    if pairs is None:
        resolved = tuple(
            _MatrixPair(left, right)
            for left in datasets
            for right in datasets
        )
    else:
        resolved = tuple(_coerce_matrix_pair(pair) for pair in pairs)

    if not resolved:
        raise ValueError("Provide at least one matrix pair.")
    missing = sorted(
        {
            name
            for pair in resolved
            for name in (pair.left, pair.right)
            if name not in datasets
        }
    )
    if missing:
        raise ValueError(f"Matrix pairs reference missing datasets: {', '.join(missing)}.")
    if len(set(resolved)) != len(resolved):
        raise ValueError("Matrix pairs must be unique; duplicate pairs would duplicate feature columns.")
    return resolved


def _normalize_datasets(datasets: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    if not isinstance(datasets, Mapping) or not datasets:
        raise ValueError("Provide at least one named matrix.")

    normalized: dict[str, pd.DataFrame] = {}
    for raw_name, dataframe in datasets.items():
        name = _clean_dataset_name(raw_name)
        if name in normalized:
            raise ValueError(f"Duplicate dataset name after trimming whitespace: {name!r}.")
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(f"Dataset {name!r} must be a pandas DataFrame.")
        normalized[name] = dataframe
    return normalized


def _resolve_pair_column(dataframe: pd.DataFrame, requested: object, *, role: str) -> object:
    exact_matches = [column for column in dataframe.columns if column == requested]
    if len(exact_matches) == 1:
        return exact_matches[0]
    if len(exact_matches) > 1:
        raise ValueError(f"Pair-table {role} ID column {requested!r} is duplicated.")

    wanted = str(requested).strip()
    matches = [column for column in dataframe.columns if str(column).strip() == wanted]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Pair-table {role} ID column {requested!r} is ambiguous after trimming whitespace.")
    available = ", ".join(repr(column) for column in dataframe.columns)
    raise ValueError(f"Pair-table {role} ID column {requested!r} was not found. Available columns: {available}.")


def _clean_entity_id(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    identifier = str(value).strip()
    return identifier or None


def _sample_lookup(dataframe: pd.DataFrame) -> dict[str, object]:
    lookup: dict[str, object] = {}
    for column in dataframe.columns[1:]:
        name = str(column).strip()
        if not name:
            raise ValueError("Sample column names must not be blank.")
        if name in lookup:
            raise ValueError(f"Sample columns are duplicated after trimming whitespace: {name!r}.")
        lookup[name] = column
    return lookup


def _row_lookup(dataframe: pd.DataFrame, sample_columns: Sequence[object]) -> dict[str, np.ndarray]:
    identifiers = dataframe.iloc[:, 0].astype("string").str.strip()
    values = dataframe.loc[:, sample_columns].to_numpy(dtype=float)
    return {
        str(identifier): values[index]
        for index, identifier in enumerate(identifiers)
    }


def _safe_std(rows: Mapping[str, np.ndarray]) -> float:
    values = np.concatenate(tuple(rows.values())) if rows else np.array([], dtype=float)
    value = float(np.nanstd(values)) if values.size else float("nan")
    return value if np.isfinite(value) and value > 0 else 1.0


def _levels(rows: Mapping[str, np.ndarray], spec: DatasetSpec) -> np.ndarray | None:
    if spec.kind == "continuous":
        return None
    if spec.levels is not None:
        return np.asarray(spec.levels, dtype=float)
    if spec.kind == "binary":
        return np.asarray((0.0, 1.0), dtype=float)

    values = np.concatenate(tuple(rows.values())) if rows else np.array([], dtype=float)
    levels = np.unique(values[np.isfinite(values)])
    if len(levels) == 0:
        raise ValueError(f"Could not infer levels for discrete dataset {spec.name!r}.")
    return levels.astype(float)


def _snap_to_levels(values: np.ndarray, levels: np.ndarray, *, dataset_name: str) -> np.ndarray:
    """Snap close floating values to declared density centers.

    ``densitymap`` uses exact dictionary lookup for discrete observations.
    Snapping permits harmless CSV floating-point noise while still rejecting a
    value that is not described by the dataset specification.
    """
    output = np.asarray(values, dtype=float).copy()
    finite = np.isfinite(output)
    if not finite.any():
        return output

    finite_values = output[finite]
    distances = np.abs(finite_values[:, np.newaxis] - levels[np.newaxis, :])
    indices = np.argmin(distances, axis=1)
    matched = np.isclose(finite_values, levels[indices], rtol=1e-8, atol=1e-12)
    if not matched.all():
        bad_values = np.unique(finite_values[~matched])[:5].tolist()
        raise ValueError(
            f"Dataset {dataset_name!r} contains values outside its declared discrete levels: {bad_values!r}."
        )
    output[finite] = levels[indices]
    return output


def _canonicalize_discrete_rows(
    rows: Mapping[str, np.ndarray],
    levels: np.ndarray | None,
    spec: DatasetSpec,
) -> dict[str, np.ndarray]:
    if levels is None:
        return {identifier: np.asarray(values, dtype=float) for identifier, values in rows.items()}
    return {
        identifier: _snap_to_levels(values, levels, dataset_name=spec.name)
        for identifier, values in rows.items()
    }


def _pair_sigma(
    left_rows: Mapping[str, np.ndarray],
    right_rows: Mapping[str, np.ndarray],
    left_spec: DatasetSpec,
    right_spec: DatasetSpec,
) -> float:
    left_scale = _safe_std(left_rows)
    right_scale = _safe_std(right_rows)
    if left_spec.kind == "continuous" and right_spec.kind == "continuous":
        return float(np.sqrt((left_scale**2 + right_scale**2) / 2))
    if left_spec.kind == "continuous":
        return left_scale
    if right_spec.kind == "continuous":
        return right_scale
    return 1.0


def _build_context(
    pair: _MatrixPair,
    prepared: Mapping[str, pd.DataFrame],
    specs: Mapping[str, DatasetSpec],
) -> _PairContext:
    left, right = prepared[pair.left], prepared[pair.right]
    left_samples, right_samples = _sample_lookup(left), _sample_lookup(right)
    shared_names = [name for name in left_samples if name in right_samples]
    if not shared_names:
        raise ValueError(
            f"Matrices {pair.left!r} and {pair.right!r} do not share any sample columns."
        )

    left_rows = _row_lookup(left, [left_samples[name] for name in shared_names])
    right_rows = _row_lookup(right, [right_samples[name] for name in shared_names])
    left_spec, right_spec = specs[pair.left], specs[pair.right]
    left_levels = _levels(left_rows, left_spec)
    right_levels = _levels(right_rows, right_spec)
    left_rows = _canonicalize_discrete_rows(left_rows, left_levels, left_spec)
    right_rows = _canonicalize_discrete_rows(right_rows, right_levels, right_spec)

    return _PairContext(
        pair=pair,
        left_spec=left_spec,
        right_spec=right_spec,
        left_rows=left_rows,
        right_rows=right_rows,
        left_scale=_safe_std(left_rows),
        right_scale=_safe_std(right_rows),
        sigma=_pair_sigma(left_rows, right_rows, left_spec, right_spec),
        left_levels=left_levels,
        right_levels=right_levels,
        sample_count=len(shared_names),
    )


def _finite_pair(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def _continuous_axis(values: np.ndarray, scale: float, bins: int) -> tuple[np.ndarray, np.ndarray]:
    limit = scale * bins
    centers = np.linspace(-limit, limit, num=bins * 2, endpoint=False)[1::2]
    return np.clip(values, -limit, limit), centers


def _density_axis(
    values: np.ndarray,
    *,
    spec: DatasetSpec,
    levels: np.ndarray | None,
    scale: float,
    bins: int,
) -> tuple[np.ndarray, np.ndarray, bool]:
    if spec.kind == "continuous":
        capped, centers = _continuous_axis(values, scale, bins)
        return capped, centers, False
    if levels is None:
        raise RuntimeError("Discrete matrix context is missing its levels.")
    return values, levels, True


def _density_columns(context: _PairContext, bins: int) -> list[str]:
    left_width = bins if context.left_levels is None else len(context.left_levels)
    right_width = bins if context.right_levels is None else len(context.right_levels)
    return [
        f"{context.pair.left}.{context.pair.right}.{index}"
        for index in range(left_width * right_width)
    ]


def _density_block(
    context: _PairContext,
    left_ids: Sequence[str | None],
    right_ids: Sequence[str | None],
    *,
    config: PairFeatureConfig,
    index: pd.Index,
) -> pd.DataFrame:
    columns = _density_columns(context, config.density_bins)
    values = np.full((len(left_ids), len(columns)), np.nan, dtype=float)
    for row_index, (left_id, right_id) in enumerate(zip(left_ids, right_ids)):
        if left_id is None or right_id is None:
            continue
        x = context.left_rows.get(left_id)
        y = context.right_rows.get(right_id)
        if x is None or y is None:
            continue
        x, y = _finite_pair(x, y)
        if not len(x):
            continue
        x, x_centers, x_discrete = _density_axis(
            x,
            spec=context.left_spec,
            levels=context.left_levels,
            scale=context.left_scale,
            bins=config.density_bins,
        )
        y, y_centers, y_discrete = _density_axis(
            y,
            spec=context.right_spec,
            levels=context.right_levels,
            scale=context.right_scale,
            bins=config.density_bins,
        )
        matrix = densitymap(
            x,
            y,
            x_centers,
            y_centers,
            xdiscrete=x_discrete,
            ydiscrete=y_discrete,
            sigma=context.sigma,
        )
        if not isinstance(matrix, str):
            values[row_index] = matrix.ravel()

    if config.log_offset is None or config.log_offset == "auto":
        offset = 1 / (context.sample_count + 1)
    else:
        offset = float(config.log_offset)
    return pd.DataFrame(np.log(values + offset), index=index, columns=columns)


def _association(x: np.ndarray, y: np.ndarray, *, method: str, context: _PairContext) -> float:
    x, y = _finite_pair(x, y)
    if len(x) < 3:
        return float("nan")
    if method == "pearson":
        if np.nanstd(x) == 0 or np.nanstd(y) == 0:
            return float("nan")
        return float(stats.pearsonr(x, y).statistic)
    if method == "spearman":
        if np.nanstd(x) == 0 or np.nanstd(y) == 0:
            return float("nan")
        return float(stats.spearmanr(x, y).statistic)
    if method == "mi":
        return compute_mutual_information(
            x,
            y,
            x_discrete=context.left_spec.kind != "continuous",
            y_discrete=context.right_spec.kind != "continuous",
        )
    return bicor(x, y)


def _statistic_block(
    context: _PairContext,
    left_ids: Sequence[str | None],
    right_ids: Sequence[str | None],
    *,
    method: str,
    index: pd.Index,
) -> pd.DataFrame:
    values = np.full(len(left_ids), np.nan, dtype=float)
    for row_index, (left_id, right_id) in enumerate(zip(left_ids, right_ids)):
        if left_id is None or right_id is None:
            continue
        x = context.left_rows.get(left_id)
        y = context.right_rows.get(right_id)
        if x is not None and y is not None:
            values[row_index] = _association(x, y, method=method, context=context)
    return pd.DataFrame({f"{context.pair.left}.{context.pair.right}": values}, index=index)


def _append_blocks(
    pair_table: pd.DataFrame,
    blocks: Sequence[pd.DataFrame],
    *,
    drop_missing: bool,
) -> pd.DataFrame:
    feature_columns = [column for block in blocks for column in block.columns]
    if not feature_columns:
        raise ValueError("No feature blocks were generated.")
    if len(set(feature_columns)) != len(feature_columns):
        raise ValueError("Generated feature names are not unique.")
    collisions = [column for column in feature_columns if column in pair_table.columns]
    if collisions:
        raise ValueError(
            "Pair-table columns collide with generated feature names: "
            f"{', '.join(map(str, collisions[:5]))}."
        )

    output = pd.concat([pair_table.copy(), *blocks], axis=1)
    if drop_missing:
        output = output.loc[output.loc[:, feature_columns].notna().all(axis=1)].copy()
    return output


def _validate_methods(methods: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(dict.fromkeys(str(method).strip().lower() for method in methods))
    if not normalized:
        raise ValueError("Provide at least one feature method.")
    invalid = sorted(set(normalized) - set(PAIR_FEATURE_METHODS))
    if invalid:
        raise ValueError(f"Unsupported feature methods: {', '.join(invalid)}.")
    return normalized


def build_pair_features(
    datasets: Mapping[str, pd.DataFrame],
    pair_table: pd.DataFrame,
    *,
    left_id_column: object,
    right_id_column: object,
    dataset_specs: Mapping[str, DatasetSpec] | Sequence[DatasetSpec] | None = None,
    specs: Mapping[str, DatasetSpec] | Sequence[DatasetSpec] | None = None,
    pairs: Sequence[object] | None = None,
    methods: Sequence[str] = ("pdm",),
    config: PairFeatureConfig | None = None,
    progress: bool = False,
) -> dict[str, pd.DataFrame]:
    """Build PDM and/or scalar features for an ordered pair table.

    Parameters
    ----------
    datasets:
        Mapping from arbitrary dataset names to wide entity-by-sample tables.
        The inputs are data frames, not paths.  Each table is prepared using
        its :class:`~pdm_learn.datasets.DatasetSpec` before construction.
    pair_table:
        One output row per observation.  It may contain arbitrary metadata;
        every input column is retained in each returned table.
    left_id_column, right_id_column:
        Headers in ``pair_table`` identifying the left and right entities.
        Header matching ignores surrounding whitespace.
    dataset_specs:
        Optional mapping or sequence of ``DatasetSpec`` objects.  Unspecified
        matrices default to continuous, row-centered preparation.
    specs:
        Short alias for ``dataset_specs``, matching
        :func:`pdm_learn.features.build_features`.  Pass only one spelling.
    pairs:
        Ordered matrix pairs to build.  A pair maps the pair table's left ID
        to the pair's left matrix and its right ID to the right matrix.  This
        argument overrides ``config.pairs``.  If neither is supplied, all
        ordered matrix pairs (including self-pairs) are built.
    methods:
        Any subset of ``"pdm"``, ``"pearson"``, ``"spearman"``, ``"mi"``,
        and ``"bicor"``.
    config:
        Pair-feature configuration.  ``drop_missing`` only examines generated
        feature columns, so optional pair metadata may itself be missing.
    progress:
        Show one progress bar per requested feature family.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Tables keyed by method name.  Generated PDM columns are named
        ``"<left dataset>.<right dataset>.<flat bin index>"``.  Scalar
        columns are named ``"<left dataset>.<right dataset>"``.

    Notes
    -----
    Pair rows whose entity IDs do not exist in the appropriate matrices are
    never fabricated.  Set ``PairFeatureConfig(drop_missing=False)`` to keep
    those rows with ``NaN`` generated values for auditing.
    """
    if not isinstance(pair_table, pd.DataFrame):
        raise TypeError("pair_table must be a pandas DataFrame.")
    if config is not None and not isinstance(config, PairFeatureConfig):
        raise TypeError("config must be a PairFeatureConfig.")
    config = config or PairFeatureConfig()
    if dataset_specs is not None and specs is not None:
        raise ValueError("Pass matrix specifications through either specs or dataset_specs, not both.")
    if pairs is not None and config.pairs is not None:
        raise ValueError("Pass matrix pairs through either pairs or config.pairs, not both.")

    normalized_datasets = _normalize_datasets(datasets)
    resolved_specs = resolve_dataset_specs(
        normalized_datasets,
        dataset_specs if dataset_specs is not None else specs,
    )
    prepared = {
        name: prepare_matrix(dataframe, resolved_specs[name])
        for name, dataframe in normalized_datasets.items()
    }
    resolved_pairs = _resolve_matrix_pairs(normalized_datasets, pairs if pairs is not None else config.pairs)
    method_names = _validate_methods(methods)

    left_column = _resolve_pair_column(pair_table, left_id_column, role="left")
    right_column = _resolve_pair_column(pair_table, right_id_column, role="right")
    left_ids = [_clean_entity_id(value) for value in pair_table[left_column]]
    right_ids = [_clean_entity_id(value) for value in pair_table[right_column]]
    contexts = [_build_context(pair, prepared, resolved_specs) for pair in resolved_pairs]

    output: dict[str, pd.DataFrame] = {}
    if "pdm" in method_names:
        iterator = tqdm(contexts, desc="Pair PDM feature blocks", disable=not progress)
        blocks = []
        for context in iterator:
            iterator.set_postfix_str(f"{context.pair.left} vs {context.pair.right}")
            blocks.append(
                _density_block(
                    context,
                    left_ids,
                    right_ids,
                    config=config,
                    index=pair_table.index,
                )
            )
        output["pdm"] = _append_blocks(pair_table, blocks, drop_missing=config.drop_missing)

    for method in method_names:
        if method == "pdm":
            continue
        iterator = tqdm(contexts, desc=f"Pair {method} feature blocks", disable=not progress)
        blocks = []
        for context in iterator:
            iterator.set_postfix_str(f"{context.pair.left} vs {context.pair.right}")
            blocks.append(
                _statistic_block(
                    context,
                    left_ids,
                    right_ids,
                    method=method,
                    index=pair_table.index,
                )
            )
        output[method] = _append_blocks(pair_table, blocks, drop_missing=config.drop_missing)
    return output


def build_pair_density_features(
    datasets: Mapping[str, pd.DataFrame],
    pair_table: pd.DataFrame,
    **kwargs: object,
) -> pd.DataFrame:
    """Build only log-transformed PDM features for a pair table."""
    return build_pair_features(datasets, pair_table, methods=("pdm",), **kwargs)["pdm"]


def build_pair_statistic_features(
    datasets: Mapping[str, pd.DataFrame],
    pair_table: pd.DataFrame,
    *,
    method: str = "pearson",
    **kwargs: object,
) -> pd.DataFrame:
    """Build one scalar association feature family for a pair table."""
    method_name = str(method).strip().lower()
    if method_name not in STATISTIC_METHODS:
        raise ValueError(f"method must be one of: {', '.join(STATISTIC_METHODS)}.")
    return build_pair_features(datasets, pair_table, methods=(method_name,), **kwargs)[method_name]


__all__ = [
    "PAIR_FEATURE_METHODS",
    "STATISTIC_METHODS",
    "PairFeatureConfig",
    "bicor",
    "build_pair_density_features",
    "build_pair_features",
    "build_pair_statistic_features",
    "compute_mutual_information",
]
