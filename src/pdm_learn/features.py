"""Domain-neutral feature builders for aligned matrix datasets.

The public API works with a mapping of named matrices.  Each matrix has one
entity identifier column and one or more sample columns.  Dataset metadata is
described with :class:`~pdm_learn.datasets.DatasetSpec`; the feature builders
do not know or care whether an entity is a gene, protein, compound, or another
row-level object.

For example::

    datasets = load_matrices({"expression": "expr.csv", "crispr": "crispr.csv"})
    features = build_features(
        datasets,
        specs=[DatasetSpec("expression"), DatasetSpec("crispr")],
        pairs=[FeaturePair("expression", "crispr")],
    )

``pdm_learn.workflows`` contains domain-specific recipes.  This module has no
dependency on those workflows.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mutual_info_score
from tqdm.auto import tqdm

from .datasets import DatasetSpec, load_matrices, prepare_matrix
from .preprocessing import densitymap, drop_nan


@dataclass(frozen=True)
class FeaturePair:
    """An ordered pair of named datasets used to create a feature block."""

    left: str
    right: str

    def __post_init__(self) -> None:
        left, right = str(self.left).strip(), str(self.right).strip()
        if not left or not right:
            raise ValueError("Feature pair dataset names must not be empty.")
        object.__setattr__(self, "left", left)
        object.__setattr__(self, "right", right)


@dataclass(frozen=True)
class FeatureConfig:
    """Configuration shared by density-map and statistic feature builders.

    Parameters
    ----------
    pairs:
        Ordered dataset pairs.  If omitted, every unordered pair of supplied
        datasets is used.
    density_bins:
        Number of centers per continuous dataset axis.
    log_offset:
        A positive density offset before log transformation, or ``"auto"``.
        ``"auto"`` uses the inverse number of aligned columns, matching the
        historical PDM workflow.
    drop_missing:
        Remove rows that cannot produce a complete requested feature table.
    entity_universe:
        ``"union"`` retains every entity seen in a selected dataset;
        ``"intersection"`` retains only entities shared by all selected
        datasets.  Missing binary rows are treated as all-zero observations.
    """

    pairs: tuple[FeaturePair, ...] | None = None
    density_bins: int = 7
    log_offset: float | str | None = "auto"
    drop_missing: bool = True
    entity_universe: str = "union"

    def __post_init__(self) -> None:
        if isinstance(self.density_bins, bool) or not isinstance(self.density_bins, int) or self.density_bins < 1:
            raise ValueError("density_bins must be a positive integer.")
        if self.entity_universe not in {"union", "intersection"}:
            raise ValueError("entity_universe must be 'union' or 'intersection'.")
        if isinstance(self.log_offset, str) and self.log_offset != "auto":
            raise ValueError("log_offset must be a number, None, or 'auto'.")
        if isinstance(self.log_offset, bool):
            raise TypeError("log_offset must be a number, None, or 'auto'.")
        if isinstance(self.log_offset, (int, float)) and self.log_offset <= 0:
            raise ValueError("log_offset must be positive.")


STATISTIC_METHODS = ("pearson", "spearman", "mi", "bicor")
"""Scalar feature methods accepted by :func:`build_features`."""

FEATURE_METHODS = ("pdm", *STATISTIC_METHODS)
"""All feature methods accepted by :func:`build_features`."""


def _coerce_dataset_spec(name: str, value: object | None) -> DatasetSpec:
    """Normalize friendly mapping and legacy metadata inputs to ``DatasetSpec``."""
    if isinstance(value, DatasetSpec):
        return value
    if value is None:
        return DatasetSpec(name)
    if isinstance(value, str):
        return DatasetSpec(name, kind=value)
    if isinstance(value, Mapping):
        return DatasetSpec(
            name,
            kind=str(value.get("kind", "continuous")),
            levels=value.get("levels"),
            normalize=value.get("normalize", "row_center"),
            id_column=value.get("id_column"),
        )

    # Compatibility with the historical OncogeneDatasetSpec without importing
    # the workflow module (which would make the core depend on a recipe).
    if hasattr(value, "kind"):
        legacy_kind = str(getattr(value, "kind"))
        legacy_normalize = getattr(value, "normalize", None)
        # The historical ``None`` meant "use the kind default".  The generic
        # API uses explicit ``None`` to mean no normalization, so translate
        # only this legacy-object spelling here.
        if legacy_normalize is None and legacy_kind == "continuous":
            legacy_normalize = "row_center"
        return DatasetSpec(
            name,
            kind=legacy_kind,
            levels=getattr(value, "levels", None),
            normalize=legacy_normalize,
            id_column=getattr(value, "id_column", None),
        )
    raise TypeError(f"Unsupported dataset specification for {name!r}: {type(value).__name__}.")


def _normalize_dataset_specs(
    datasets: Mapping[str, pd.DataFrame],
    specs: Mapping[str, object] | Sequence[DatasetSpec] | None,
) -> dict[str, DatasetSpec]:
    if specs is None:
        values: Mapping[str, object] = {}
    elif isinstance(specs, Mapping):
        normalized_values = {str(name).strip(): value for name, value in specs.items()}
        if len(normalized_values) != len(specs):
            raise ValueError("Dataset specification names must be unique after trimming whitespace.")
        values = normalized_values
    else:
        values = {str(spec.name): spec for spec in specs}

    normalized = {str(name): _coerce_dataset_spec(str(name), values.get(name)) for name in datasets}
    missing = set(values) - set(datasets)
    if missing:
        raise ValueError(f"Dataset specs reference missing datasets: {', '.join(sorted(map(str, missing)))}")
    return normalized


def _coerce_feature_pair(value: FeaturePair | Sequence[object] | object) -> FeaturePair:
    if isinstance(value, FeaturePair):
        return value
    if hasattr(value, "left") and hasattr(value, "right"):
        return FeaturePair(str(getattr(value, "left")).strip(), str(getattr(value, "right")).strip())
    if isinstance(value, str):
        separator = ":" if ":" in value else ","
        if separator not in value:
            raise ValueError("String feature pairs must use 'left:right' or 'left,right'.")
        left, right = value.split(separator, maxsplit=1)
        return FeaturePair(left.strip(), right.strip())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 2:
        return FeaturePair(str(value[0]).strip(), str(value[1]).strip())
    raise TypeError("Feature pairs must be FeaturePair instances or two-item sequences.")


def _normalize_feature_pairs(
    datasets: Mapping[str, pd.DataFrame],
    pairs: Sequence[FeaturePair | Sequence[object] | object] | None,
) -> tuple[FeaturePair, ...]:
    if pairs is None:
        names = list(datasets)
        normalized = tuple(
            FeaturePair(left, right)
            for index, left in enumerate(names)
            for right in names[index + 1 :]
        )
    else:
        normalized = tuple(_coerce_feature_pair(pair) for pair in pairs)
    if not normalized:
        raise ValueError("Provide at least one feature pair.")
    missing = {
        name
        for pair in normalized
        for name in (pair.left, pair.right)
        if name not in datasets
    }
    if missing:
        raise ValueError(f"Feature pairs reference missing datasets: {', '.join(sorted(missing))}")
    return normalized


def _entity_values(dataframe: pd.DataFrame) -> list[str]:
    return dataframe.iloc[:, 0].astype(str).str.strip().tolist()


def _feature_entities(
    prepared: Mapping[str, pd.DataFrame],
    pairs: Sequence[FeaturePair],
    universe: str,
) -> list[str]:
    names = sorted({name for pair in pairs for name in (pair.left, pair.right)})
    entity_sets = [set(_entity_values(prepared[name])) for name in names]
    if not entity_sets:
        return []
    entities = set.union(*entity_sets) if universe == "union" else set.intersection(*entity_sets)
    return sorted(entities)


def _numeric_matrix(dataframe: pd.DataFrame) -> np.ndarray:
    return dataframe.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)


def _align_pair(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_spec: DatasetSpec,
    right_spec: DatasetSpec,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align one pair by sample columns and compatible entity rows.

    A missing binary row represents an absent event and is therefore filled
    with zeros against a non-binary reference (or across two binary matrices).
    Other missing rows remain unavailable rather than being imputed.
    """
    left_id, right_id = left.columns[0], right.columns[0]
    left_columns = {str(column).strip(): column for column in left.columns[1:]}
    right_columns = {str(column).strip(): column for column in right.columns[1:]}
    shared_samples = sorted(set(left_columns) & set(right_columns))
    left_out = left.loc[:, [left_id, *[left_columns[name] for name in shared_samples]]].copy()
    right_out = right.loc[:, [right_id, *[right_columns[name] for name in shared_samples]]].copy()
    left_out.columns = [left_id, *shared_samples]
    right_out.columns = [right_id, *shared_samples]

    left_out[left_id] = left_out[left_id].astype(str).str.strip()
    right_out[right_id] = right_out[right_id].astype(str).str.strip()
    left_ids = set(left_out[left_id])
    right_ids = set(right_out[right_id])
    if left_spec.kind == "binary" and right_spec.kind == "binary":
        entity_ids = sorted(left_ids | right_ids)
    elif left_spec.kind == "binary":
        entity_ids = sorted(right_ids)
    elif right_spec.kind == "binary":
        entity_ids = sorted(left_ids)
    else:
        entity_ids = sorted(left_ids & right_ids)

    def reindex(frame: pd.DataFrame, identifier: object, ids: Sequence[str], fill_binary: bool) -> pd.DataFrame:
        indexed = frame.drop_duplicates(subset=identifier).set_index(identifier)
        indexed = indexed.reindex(ids)
        values = indexed.apply(pd.to_numeric, errors="coerce")
        if fill_binary:
            values = values.fillna(0.0)
        output = values.reset_index()
        output.columns = [identifier, *shared_samples]
        return output

    left_out = reindex(left_out, left_id, entity_ids, left_spec.kind == "binary")
    right_out = reindex(right_out, right_id, entity_ids, right_spec.kind == "binary")
    return left_out, right_out


def _safe_std(values: pd.DataFrame) -> float:
    std = float(np.nanstd(_numeric_matrix(values)))
    return std if np.isfinite(std) and std > 0 else 1.0


def _pair_sigma(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_spec: DatasetSpec,
    right_spec: DatasetSpec,
) -> float:
    if left_spec.kind == "continuous" and right_spec.kind == "continuous":
        left_variance = float(np.nanvar(_numeric_matrix(left)))
        right_variance = float(np.nanvar(_numeric_matrix(right)))
        value = float(np.sqrt((left_variance + right_variance) / 2))
        return value if np.isfinite(value) and value > 0 else 1.0
    if left_spec.kind == "continuous":
        return _safe_std(left)
    if right_spec.kind == "continuous":
        return _safe_std(right)
    return 1.0


def _levels(dataframe: pd.DataFrame, spec: DatasetSpec) -> np.ndarray:
    if spec.kind == "continuous":
        raise ValueError("Continuous datasets do not have fixed levels.")
    if spec.levels is not None:
        levels = np.asarray(spec.levels, dtype=float)
    elif spec.kind == "binary":
        levels = np.asarray((0.0, 1.0), dtype=float)
    else:
        values = _numeric_matrix(dataframe)
        levels = np.unique(values[np.isfinite(values)])
    if len(levels) == 0:
        raise ValueError(f"Could not infer levels for discrete dataset {spec.name!r}.")
    return np.asarray(levels, dtype=float)


def _continuous_centers(values: np.ndarray, bins: int, std: float) -> tuple[np.ndarray, np.ndarray]:
    """Cap one row and return centers using its dataset-wide scale."""
    capped = np.clip(values.astype(float), -std * bins, std * bins)
    centers = np.linspace(-std * bins, std * bins, num=bins * 2, endpoint=False)[1::2]
    return capped, centers


def _snap_to_levels(values: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Make discrete values safe for exact lookup in the density primitive."""
    output = np.asarray(values, dtype=float).copy()
    finite = np.isfinite(output)
    if finite.any():
        distances = np.abs(output[finite, np.newaxis] - levels[np.newaxis, :])
        output[finite] = levels[np.argmin(distances, axis=1)]
    return output


def _row_lookup(dataframe: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        str(row.iloc[0]).strip(): row.iloc[1:].to_numpy(dtype=float)
        for _, row in dataframe.iterrows()
    }


def compute_mutual_information(
    x: Sequence[float],
    y: Sequence[float],
    *,
    x_discrete: bool = False,
    y_discrete: bool = False,
    random_state: int = 0,
) -> float:
    """Compute mutual information while respecting continuous/discrete inputs."""
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    finite = np.isfinite(x_array) & np.isfinite(y_array)
    x_array, y_array = x_array[finite], y_array[finite]
    if len(x_array) < 3:
        return float("nan")
    try:
        if x_discrete and y_discrete:
            return float(mutual_info_score(_discrete_codes(x_array), _discrete_codes(y_array)))
        if x_discrete:
            return float(mutual_info_score(_discrete_codes(x_array), _bin_continuous(y_array)))
        if y_discrete:
            return float(mutual_info_score(_bin_continuous(x_array), _discrete_codes(y_array)))
        return float(
            mutual_info_regression(
                x_array.reshape(-1, 1),
                y_array,
                discrete_features=False,
                n_neighbors=max(1, min(3, len(x_array) - 1)),
                random_state=random_state,
            )[0]
        )
    except ValueError:
        return float("nan")


def _bin_continuous(values: np.ndarray, bins: int = 10) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) == 0 or np.nanstd(values) == 0:
        return np.zeros(len(values), dtype=int)
    edges = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)))
    return np.zeros(len(values), dtype=int) if len(edges) <= 2 else np.digitize(values, edges[1:-1], right=False)


def _discrete_codes(values: np.ndarray) -> np.ndarray:
    """Encode arbitrary numeric discrete levels as categorical codes."""
    _, codes = np.unique(np.asarray(values, dtype=float), return_inverse=True)
    return codes


def bicor(
    x: Sequence[float],
    y: Sequence[float],
    c: float = 9.0,
    epsilon: float = 1e-9,
) -> float:
    """Return the biweight midcorrelation of two numeric vectors."""
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    finite = np.isfinite(x_array) & np.isfinite(y_array)
    x_array, y_array = x_array[finite], y_array[finite]
    if len(x_array) < 3:
        return float("nan")
    x_median, y_median = np.median(x_array), np.median(y_array)
    x_mad = max(float(np.median(np.abs(x_array - x_median))), epsilon)
    y_mad = max(float(np.median(np.abs(y_array - y_median))), epsilon)
    ux, uy = (x_array - x_median) / (c * x_mad), (y_array - y_median) / (c * y_mad)
    wx, wy = (1 - ux**2) ** 2, (1 - uy**2) ** 2
    wx[np.abs(ux) >= 1] = 0
    wy[np.abs(uy) >= 1] = 0
    x_weighted, y_weighted = (x_array - x_median) * wx, (y_array - y_median) * wy
    denominator = max(float(np.sqrt(np.sum(x_weighted**2) * np.sum(y_weighted**2))), epsilon)
    return float(np.sum(x_weighted * y_weighted) / denominator)


def _build_density_table(
    prepared: Mapping[str, pd.DataFrame],
    specs: Mapping[str, DatasetSpec],
    pairs: Sequence[FeaturePair],
    *,
    output_id_column: str,
    config: FeatureConfig,
    progress: bool,
) -> pd.DataFrame:
    entities = _feature_entities(prepared, pairs, config.entity_universe)
    output = pd.DataFrame({output_id_column: entities})
    entity_index = {entity: index for index, entity in enumerate(entities)}
    pair_iterator = tqdm(pairs, desc="PDM feature blocks", disable=not progress)
    for pair in pair_iterator:
        pair_iterator.set_postfix_str(f"{pair.left} vs {pair.right}")
        left, right = _align_pair(prepared[pair.left], prepared[pair.right], specs[pair.left], specs[pair.right])
        left_spec, right_spec = specs[pair.left], specs[pair.right]
        left_levels = None if left_spec.kind == "continuous" else _levels(left, left_spec)
        right_levels = None if right_spec.kind == "continuous" else _levels(right, right_spec)
        left_width = config.density_bins if left_levels is None else len(left_levels)
        right_width = config.density_bins if right_levels is None else len(right_levels)
        columns = [f"{pair.left}.{pair.right}.{index}" for index in range(left_width * right_width)]
        block = pd.DataFrame(np.nan, index=range(len(entities)), columns=columns)
        left_rows, right_rows = _row_lookup(left), _row_lookup(right)
        left_std, right_std = _safe_std(left), _safe_std(right)
        sigma = _pair_sigma(left, right, left_spec, right_spec)
        for entity in set(left_rows) & set(right_rows):
            if entity not in entity_index:
                continue
            x, y = left_rows[entity], right_rows[entity]
            if left_levels is None:
                x, x_centers = _continuous_centers(x, config.density_bins, left_std)
            else:
                x, x_centers = _snap_to_levels(x, left_levels), left_levels
            if right_levels is None:
                y, y_centers = _continuous_centers(y, config.density_bins, right_std)
            else:
                y, y_centers = _snap_to_levels(y, right_levels), right_levels
            x, y = drop_nan(x, y)
            if len(x) == 0:
                continue
            matrix = densitymap(
                x,
                y,
                x_centers,
                y_centers,
                xdiscrete=left_levels is not None,
                ydiscrete=right_levels is not None,
                sigma=sigma,
            )
            if not isinstance(matrix, str):
                block.iloc[entity_index[entity]] = matrix.flatten()
        if config.log_offset in {None, "auto"}:
            offset = 1 / max(len(left.columns), 1)
        else:
            offset = float(config.log_offset)
        output = pd.concat([output, np.log(block.astype(float) + offset)], axis=1)
    return output.dropna().reset_index(drop=True) if config.drop_missing else output.reset_index(drop=True)


def _build_statistic_table(
    prepared: Mapping[str, pd.DataFrame],
    specs: Mapping[str, DatasetSpec],
    pairs: Sequence[FeaturePair],
    *,
    output_id_column: str,
    method: str,
    config: FeatureConfig,
    progress: bool,
) -> pd.DataFrame:
    entities = _feature_entities(prepared, pairs, config.entity_universe)
    output = pd.DataFrame({output_id_column: entities})
    entity_index = {entity: index for index, entity in enumerate(entities)}
    pair_iterator = tqdm(pairs, desc=f"{method} feature blocks", disable=not progress)
    for pair in pair_iterator:
        pair_iterator.set_postfix_str(f"{pair.left} vs {pair.right}")
        left, right = _align_pair(prepared[pair.left], prepared[pair.right], specs[pair.left], specs[pair.right])
        block = pd.DataFrame(np.nan, index=range(len(entities)), columns=[f"{pair.left}.{pair.right}"])
        left_rows, right_rows = _row_lookup(left), _row_lookup(right)
        for entity in set(left_rows) & set(right_rows):
            if entity not in entity_index:
                continue
            x, y = drop_nan(left_rows[entity], right_rows[entity])
            if len(x) < 3:
                continue
            if method == "pearson":
                if np.nanstd(x) == 0 or np.nanstd(y) == 0:
                    continue
                value = stats.pearsonr(x, y).statistic
            elif method == "spearman":
                if np.nanstd(x) == 0 or np.nanstd(y) == 0:
                    continue
                value = stats.spearmanr(x, y).statistic
            elif method == "mi":
                value = compute_mutual_information(
                    x,
                    y,
                    x_discrete=specs[pair.left].kind != "continuous",
                    y_discrete=specs[pair.right].kind != "continuous",
                )
            else:
                value = bicor(x, y)
            block.iloc[entity_index[entity], 0] = value
        output = pd.concat([output, block], axis=1)
    return output.dropna().reset_index(drop=True) if config.drop_missing else output.reset_index(drop=True)


def build_features(
    datasets: Mapping[str, pd.DataFrame],
    *,
    specs: Mapping[str, object] | Sequence[DatasetSpec] | None = None,
    pairs: Sequence[FeaturePair | Sequence[object] | object] | None = None,
    methods: Sequence[str] = FEATURE_METHODS,
    config: FeatureConfig | None = None,
    density_bins: int | None = None,
    log_offset: float | str | None = "auto",
    drop_missing: bool | None = None,
    entity_universe: str | None = None,
    output_id_column: str | None = None,
    progress: bool = False,
) -> dict[str, pd.DataFrame]:
    """Build PDM and scalar association features from named matrices.

    Inputs are in-memory DataFrames.  File paths belong in :func:`load_matrices`
    or an application/workflow layer.  ``specs`` accepts either a mapping keyed
    by dataset name or a sequence of :class:`DatasetSpec` instances.
    """
    if not datasets:
        raise ValueError("Provide at least one named dataset.")
    normalized_datasets = {str(name).strip(): dataframe for name, dataframe in datasets.items()}
    if len(normalized_datasets) != len(datasets) or any(not name for name in normalized_datasets):
        raise ValueError("Dataset names must be unique, non-empty strings.")
    normalized_specs = _normalize_dataset_specs(normalized_datasets, specs)
    prepared = {
        name: prepare_matrix(dataframe, normalized_specs[name])
        for name, dataframe in normalized_datasets.items()
    }
    base_config = config or FeatureConfig()
    resolved_pairs = _normalize_feature_pairs(
        normalized_datasets,
        pairs if pairs is not None else base_config.pairs,
    )
    resolved_config = FeatureConfig(
        pairs=resolved_pairs,
        density_bins=base_config.density_bins if density_bins is None else density_bins,
        log_offset=base_config.log_offset if log_offset == "auto" else log_offset,
        drop_missing=base_config.drop_missing if drop_missing is None else drop_missing,
        entity_universe=base_config.entity_universe if entity_universe is None else entity_universe,
    )
    method_names = tuple(dict.fromkeys(str(method).lower() for method in methods))
    invalid = sorted(set(method_names) - set(FEATURE_METHODS))
    if invalid:
        raise ValueError(f"Unsupported feature methods: {', '.join(invalid)}")
    if not method_names:
        raise ValueError("Provide at least one feature method.")
    default_id = str(prepared[resolved_pairs[0].left].columns[0])
    identifier = output_id_column or default_id
    output: dict[str, pd.DataFrame] = {}
    if "pdm" in method_names:
        output["pdm"] = _build_density_table(
            prepared,
            normalized_specs,
            resolved_pairs,
            output_id_column=identifier,
            config=resolved_config,
            progress=progress,
        )
    for method in method_names:
        if method == "pdm":
            continue
        output[method] = _build_statistic_table(
            prepared,
            normalized_specs,
            resolved_pairs,
            output_id_column=identifier,
            method=method,
            config=resolved_config,
            progress=progress,
        )
    return output


def build_density_features(
    datasets: Mapping[str, pd.DataFrame],
    **kwargs: object,
) -> pd.DataFrame:
    """Build only log-transformed pairwise density-map features."""
    return build_features(datasets, methods=("pdm",), **kwargs)["pdm"]


def build_statistic_features(
    datasets: Mapping[str, pd.DataFrame],
    *,
    method: str = "pearson",
    **kwargs: object,
) -> pd.DataFrame:
    """Build one scalar association feature table."""
    method_name = method.lower()
    if method_name not in STATISTIC_METHODS:
        raise ValueError(f"method must be one of: {', '.join(STATISTIC_METHODS)}")
    return build_features(datasets, methods=(method_name,), **kwargs)[method_name]


# ---------------------------------------------------------------------------
# Compatibility API introduced before the core refactor.

DatasetPair = FeaturePair
"""Deprecated compatibility alias for :class:`FeaturePair`."""


def load_gene_matrices(paths: Mapping[str, str | Path]) -> dict[str, pd.DataFrame]:
    """Compatibility alias for :func:`pdm_learn.datasets.load_matrices`."""
    return load_matrices(paths)


def build_feature_tables(
    datasets: Mapping[str, pd.DataFrame],
    *,
    pairs: Sequence[FeaturePair | Sequence[object] | object] | None = None,
    dataset_specs: Mapping[str, object] | Sequence[DatasetSpec] | None = None,
    methods: Sequence[str] = STATISTIC_METHODS,
    gene_universe: str = "union",
    gene_column_name: str = "gene name",
    boxes: int = 7,
    log_offset: float | None = None,
    drop_missing: bool = True,
    progress: bool = False,
) -> dict[str, pd.DataFrame]:
    """Compatibility wrapper for the pre-0.3 generic feature API."""
    scalar_methods = tuple(str(method).lower() for method in methods)
    invalid = sorted(set(scalar_methods) - set(STATISTIC_METHODS))
    if invalid:
        raise ValueError(f"Unsupported statistic methods: {', '.join(invalid)}")
    return build_features(
        datasets,
        specs=dataset_specs,
        pairs=pairs,
        methods=("pdm", *scalar_methods),
        density_bins=boxes,
        log_offset=log_offset,
        drop_missing=drop_missing,
        entity_universe=gene_universe,
        output_id_column=gene_column_name,
        progress=progress,
    )


def load_labels(path: str | Path) -> set[str]:
    """Compatibility alias for :func:`pdm_learn.labels.load_label_set`."""
    from .labels import load_label_set

    return load_label_set(path)


def split_labeled_feature_tables(
    feature_tables: Mapping[str, pd.DataFrame],
    positive_labels: Iterable[object],
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Split every feature table into positive and unlabeled rows."""
    from .labels import split_labeled_rows

    return {
        str(name): split_labeled_rows(table, positive_labels, drop_missing=True)
        for name, table in feature_tables.items()
    }


def rank_labeled_genes(
    feature_table: pd.DataFrame,
    positive_labels: Iterable[object],
    *,
    trials: int = 100,
    model: str = "XGB",
    ks_test: bool = True,
    features_left: int | None = None,
    progress: bool = False,
) -> pd.DataFrame:
    """Compatibility alias for :func:`pdm_learn.modeling.rank_candidates`."""
    from .modeling import rank_candidates

    return rank_candidates(
        feature_table,
        positive_labels,
        trials=trials,
        model=model,
        ks_test=ks_test,
        features_left=features_left,
        progress=progress,
    )


__all__ = [
    "DatasetPair",
    "DatasetSpec",
    "FEATURE_METHODS",
    "FeatureConfig",
    "FeaturePair",
    "STATISTIC_METHODS",
    "bicor",
    "build_density_features",
    "build_feature_tables",
    "build_features",
    "build_statistic_features",
    "compute_mutual_information",
    "load_gene_matrices",
    "load_labels",
    "rank_labeled_genes",
    "split_labeled_feature_tables",
]
