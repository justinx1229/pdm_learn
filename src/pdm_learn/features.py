"""Feature construction for aligned entity-by-sample matrices.

This module is deliberately domain neutral.  A matrix may describe genes,
proteins, compounds, or any other entities; callers provide the identifier and
value semantics through :class:`pdm_learn.datasets.DatasetSpec`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mutual_info_score
from tqdm.auto import tqdm

from .datasets import DatasetSpec, DatasetSpecs, prepare_matrix, resolve_dataset_specs


@dataclass(frozen=True)
class FeaturePair:
    """An ordered pair of named matrices that defines one feature block."""

    left: str
    right: str

    def __post_init__(self) -> None:
        left, right = str(self.left).strip(), str(self.right).strip()
        if not left or not right:
            raise ValueError("Feature-pair dataset names must not be empty.")
        object.__setattr__(self, "left", left)
        object.__setattr__(self, "right", right)


@dataclass(frozen=True)
class FeatureConfig:
    """Shared configuration for density-map and scalar features.

    ``pairs`` is optional because :func:`build_features` also accepts it as a
    direct argument.  ``entity_universe`` controls the rows that are eligible
    for output before incomplete feature rows are optionally removed.
    """

    pairs: tuple[FeaturePair, ...] | None = None
    density_bins: int = 7
    log_offset: float | str | None = "auto"
    drop_missing: bool = True
    entity_universe: str = "union"

    def __post_init__(self) -> None:
        if self.pairs is not None:
            pairs = tuple(self.pairs)
            if not pairs or any(not isinstance(pair, FeaturePair) for pair in pairs):
                raise TypeError("pairs must be a non-empty sequence of FeaturePair objects.")
            object.__setattr__(self, "pairs", pairs)
        if (
            isinstance(self.density_bins, bool)
            or not isinstance(self.density_bins, int)
            or self.density_bins < 1
        ):
            raise ValueError("density_bins must be a positive integer.")
        if self.log_offset is not None and self.log_offset != "auto":
            if isinstance(self.log_offset, bool) or not isinstance(self.log_offset, (int, float)):
                raise TypeError("log_offset must be a positive number, None, or 'auto'.")
            if not np.isfinite(self.log_offset) or self.log_offset <= 0:
                raise ValueError("log_offset must be a positive finite number.")
        if not isinstance(self.drop_missing, bool):
            raise TypeError("drop_missing must be a boolean.")
        if self.entity_universe not in {"union", "intersection"}:
            raise ValueError("entity_universe must be 'union' or 'intersection'.")


STATISTIC_METHODS = ("pearson", "spearman", "mi", "bicor")
"""Scalar association methods accepted by :func:`build_features`."""

FEATURE_METHODS = ("pdm", *STATISTIC_METHODS)
"""All feature methods accepted by :func:`build_features`."""


def drop_missing_observations(
    x: Sequence[float],
    y: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return finite, paired observations from two equally sized vectors."""
    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    if x_values.shape != y_values.shape:
        raise ValueError("Feature vectors must have matching shapes.")
    keep = np.isfinite(x_values) & np.isfinite(y_values)
    return x_values[keep], y_values[keep]


def density_map(
    x: Sequence[float],
    y: Sequence[float],
    x_centers: Sequence[float],
    y_centers: Sequence[float],
    *,
    x_discrete: bool = False,
    y_discrete: bool = False,
    sigma: float = 1.0,
) -> np.ndarray:
    """Estimate a normalized pairwise density map.

    Continuous axes use a Gaussian kernel centered at each paired
    observation. Discrete axes assign observations directly to their declared
    centers. The returned matrix has shape ``(len(y_centers), len(x_centers))``.
    """
    x_values, y_values = drop_missing_observations(x, y)
    x_axis = np.asarray(x_centers, dtype=float)
    y_axis = np.asarray(y_centers, dtype=float)
    if x_axis.ndim != 1 or y_axis.ndim != 1 or not len(x_axis) or not len(y_axis):
        raise ValueError("Density centers must be non-empty one-dimensional sequences.")
    if len(np.unique(x_axis)) != len(x_axis) or len(np.unique(y_axis)) != len(y_axis):
        raise ValueError("Density centers must be unique.")
    if (not x_discrete or not y_discrete) and (not np.isfinite(sigma) or sigma <= 0):
        raise ValueError("sigma must be a positive finite number for continuous axes.")

    output = np.zeros((len(y_axis), len(x_axis)), dtype=float)
    if not len(x_values):
        return output

    def indices_for(values: np.ndarray, centers: np.ndarray) -> np.ndarray:
        distances = np.abs(values[:, np.newaxis] - centers[np.newaxis, :])
        indices = np.argmin(distances, axis=1)
        if not np.all(np.isclose(values, centers[indices], rtol=1e-8, atol=1e-12)):
            raise ValueError("Discrete observations must match a declared density center.")
        return indices

    def normalized(weights: np.ndarray) -> np.ndarray:
        total = float(weights.sum())
        if total > 0 and np.isfinite(total):
            return weights / total
        fallback = np.zeros_like(weights, dtype=float)
        fallback.flat[np.argmax(weights)] = 1.0
        return fallback

    x_indices = indices_for(x_values, x_axis) if x_discrete else None
    y_indices = indices_for(y_values, y_axis) if y_discrete else None
    inverse_sigma_squared = 1.0 / float(sigma) ** 2
    for index, (x_value, y_value) in enumerate(zip(x_values, y_values, strict=True)):
        if x_discrete and y_discrete:
            output[y_indices[index], x_indices[index]] += 1.0
        elif x_discrete:
            weights = np.exp(-0.5 * inverse_sigma_squared * (y_value - y_axis) ** 2)
            output[:, x_indices[index]] += normalized(weights)
        elif y_discrete:
            weights = np.exp(-0.5 * inverse_sigma_squared * (x_value - x_axis) ** 2)
            output[y_indices[index], :] += normalized(weights)
        else:
            weights = np.exp(
                -0.5
                * inverse_sigma_squared
                * ((x_value - x_axis[np.newaxis, :]) ** 2 + (y_value - y_axis[:, np.newaxis]) ** 2)
            )
            output += normalized(weights)
    return output / len(x_values)


def _normalize_datasets(datasets: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    if not isinstance(datasets, Mapping) or not datasets:
        raise ValueError("Provide at least one named matrix.")
    normalized: dict[str, pd.DataFrame] = {}
    for raw_name, dataframe in datasets.items():
        name = str(raw_name).strip()
        if not name:
            raise ValueError("Dataset names must not be empty.")
        if name in normalized:
            raise ValueError(f"Duplicate dataset name after trimming whitespace: {name!r}.")
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(f"Dataset {name!r} must be a pandas DataFrame.")
        normalized[name] = dataframe
    return normalized


def _resolve_pairs(
    datasets: Mapping[str, pd.DataFrame],
    pairs: Sequence[FeaturePair] | None,
) -> tuple[FeaturePair, ...]:
    if pairs is None:
        names = tuple(datasets)
        resolved = tuple(
            FeaturePair(left, right)
            for index, left in enumerate(names)
            for right in names[index + 1 :]
        )
    else:
        resolved = tuple(pairs)
        if any(not isinstance(pair, FeaturePair) for pair in resolved):
            raise TypeError("pairs must contain FeaturePair objects.")
    if not resolved:
        raise ValueError("Provide at least one FeaturePair.")
    missing = sorted(
        {name for pair in resolved for name in (pair.left, pair.right) if name not in datasets}
    )
    if missing:
        raise ValueError(f"Feature pairs reference missing datasets: {', '.join(missing)}.")
    if len(set(resolved)) != len(resolved):
        raise ValueError("Feature pairs must be unique.")
    return resolved


def _entity_values(dataframe: pd.DataFrame) -> set[str]:
    return set(dataframe.iloc[:, 0].astype("string").str.strip())


def _feature_entities(
    prepared: Mapping[str, pd.DataFrame],
    pairs: Sequence[FeaturePair],
    universe: str,
) -> list[str]:
    names = sorted({name for pair in pairs for name in (pair.left, pair.right)})
    entity_sets = [_entity_values(prepared[name]) for name in names]
    entities = set.union(*entity_sets) if universe == "union" else set.intersection(*entity_sets)
    return sorted(entities)


def _numeric_values(dataframe: pd.DataFrame) -> np.ndarray:
    return dataframe.iloc[:, 1:].to_numpy(dtype=float)


def _align_pair(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_spec: DatasetSpec,
    right_spec: DatasetSpec,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align a pair of prepared matrices by common sample headers and entities."""
    left_id, right_id = left.columns[0], right.columns[0]
    left_samples = {str(column).strip(): column for column in left.columns[1:]}
    right_samples = {str(column).strip(): column for column in right.columns[1:]}
    shared_samples = sorted(set(left_samples) & set(right_samples))
    if not shared_samples:
        raise ValueError(
            f"Matrices {left_spec.name!r} and {right_spec.name!r} do not share any sample columns."
        )

    left_selected = left.loc[:, [left_id, *[left_samples[name] for name in shared_samples]]].copy()
    right_selected = right.loc[
        :, [right_id, *[right_samples[name] for name in shared_samples]]
    ].copy()
    left_selected.columns = [left_id, *shared_samples]
    right_selected.columns = [right_id, *shared_samples]
    left_ids = set(left_selected[left_id].astype("string").str.strip())
    right_ids = set(right_selected[right_id].astype("string").str.strip())
    if left_spec.kind == "binary" and right_spec.kind == "binary":
        entity_ids = sorted(left_ids | right_ids)
    elif left_spec.kind == "binary":
        entity_ids = sorted(right_ids)
    elif right_spec.kind == "binary":
        entity_ids = sorted(left_ids)
    else:
        entity_ids = sorted(left_ids & right_ids)

    def reindex(frame: pd.DataFrame, identifier: object, fill_binary: bool) -> pd.DataFrame:
        indexed = frame.copy()
        indexed[identifier] = indexed[identifier].astype("string").str.strip()
        values = (
            indexed.set_index(identifier).reindex(entity_ids).apply(pd.to_numeric, errors="coerce")
        )
        if fill_binary:
            values = values.fillna(0.0)
        result = values.reset_index()
        result.columns = [identifier, *shared_samples]
        return result

    return (
        reindex(left_selected, left_id, left_spec.kind == "binary"),
        reindex(right_selected, right_id, right_spec.kind == "binary"),
    )


def _safe_scale(values: np.ndarray) -> float:
    scale = float(np.nanstd(values))
    return scale if np.isfinite(scale) and scale > 0 else 1.0


def _pair_sigma(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_spec: DatasetSpec,
    right_spec: DatasetSpec,
) -> float:
    left_scale, right_scale = (
        _safe_scale(_numeric_values(left)),
        _safe_scale(_numeric_values(right)),
    )
    if left_spec.kind == "continuous" and right_spec.kind == "continuous":
        return float(np.sqrt((left_scale**2 + right_scale**2) / 2))
    return (
        left_scale
        if left_spec.kind == "continuous"
        else right_scale
        if right_spec.kind == "continuous"
        else 1.0
    )


def _levels(dataframe: pd.DataFrame, spec: DatasetSpec) -> np.ndarray | None:
    if spec.kind == "continuous":
        return None
    if spec.levels is not None:
        return np.asarray(spec.levels, dtype=float)
    if spec.kind == "binary":
        return np.asarray((0.0, 1.0), dtype=float)
    values = _numeric_values(dataframe)
    levels = np.unique(values[np.isfinite(values)])
    if not len(levels):
        raise ValueError(f"Could not infer discrete levels for dataset {spec.name!r}.")
    return levels


def _continuous_axis(values: np.ndarray, bins: int, scale: float) -> tuple[np.ndarray, np.ndarray]:
    limit = scale * bins
    centers = np.linspace(-limit, limit, num=bins * 2, endpoint=False)[1::2]
    return np.clip(values, -limit, limit), centers


def _snap_to_levels(values: np.ndarray, levels: np.ndarray, dataset_name: str) -> np.ndarray:
    output = np.asarray(values, dtype=float).copy()
    finite = np.isfinite(output)
    if not finite.any():
        return output
    distances = np.abs(output[finite, np.newaxis] - levels[np.newaxis, :])
    indices = np.argmin(distances, axis=1)
    if not np.all(np.isclose(output[finite], levels[indices], rtol=1e-8, atol=1e-12)):
        raise ValueError(
            f"Dataset {dataset_name!r} contains values outside its declared discrete levels."
        )
    output[finite] = levels[indices]
    return output


def _row_lookup(dataframe: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        str(row.iloc[0]).strip(): row.iloc[1:].to_numpy(dtype=float)
        for _, row in dataframe.iterrows()
    }


def _bin_continuous(values: np.ndarray, bins: int = 10) -> np.ndarray:
    if not len(values) or np.nanstd(values) == 0:
        return np.zeros(len(values), dtype=int)
    edges = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)))
    return np.zeros(len(values), dtype=int) if len(edges) <= 2 else np.digitize(values, edges[1:-1])


def _discrete_codes(values: np.ndarray) -> np.ndarray:
    return np.unique(values, return_inverse=True)[1]


def compute_mutual_information(
    x: Sequence[float],
    y: Sequence[float],
    *,
    x_discrete: bool = False,
    y_discrete: bool = False,
    random_state: int = 0,
) -> float:
    """Compute mutual information for continuous and/or discrete observations."""
    x_values, y_values = drop_missing_observations(x, y)
    if len(x_values) < 3:
        return float("nan")
    try:
        if x_discrete and y_discrete:
            return float(mutual_info_score(_discrete_codes(x_values), _discrete_codes(y_values)))
        if x_discrete:
            return float(mutual_info_score(_discrete_codes(x_values), _bin_continuous(y_values)))
        if y_discrete:
            return float(mutual_info_score(_bin_continuous(x_values), _discrete_codes(y_values)))
        return float(
            mutual_info_regression(
                x_values.reshape(-1, 1),
                y_values,
                n_neighbors=max(1, min(3, len(x_values) - 1)),
                random_state=random_state,
            )[0]
        )
    except ValueError:
        return float("nan")


def bicor(
    x: Sequence[float],
    y: Sequence[float],
    *,
    tuning_constant: float = 9.0,
    epsilon: float = 1e-9,
) -> float:
    """Return the biweight midcorrelation of two numeric vectors."""
    x_values, y_values = drop_missing_observations(x, y)
    if len(x_values) < 3:
        return float("nan")
    x_median, y_median = np.median(x_values), np.median(y_values)
    x_mad = max(float(np.median(np.abs(x_values - x_median))), epsilon)
    y_mad = max(float(np.median(np.abs(y_values - y_median))), epsilon)
    x_u = (x_values - x_median) / (tuning_constant * x_mad)
    y_u = (y_values - y_median) / (tuning_constant * y_mad)
    x_weights = np.where(np.abs(x_u) < 1, (1 - x_u**2) ** 2, 0.0)
    y_weights = np.where(np.abs(y_u) < 1, (1 - y_u**2) ** 2, 0.0)
    x_weighted = (x_values - x_median) * x_weights
    y_weighted = (y_values - y_median) * y_weights
    denominator = max(float(np.sqrt(np.sum(x_weighted**2) * np.sum(y_weighted**2))), epsilon)
    return float(np.sum(x_weighted * y_weighted) / denominator)


def _association(
    x: np.ndarray,
    y: np.ndarray,
    *,
    method: str,
    left_spec: DatasetSpec,
    right_spec: DatasetSpec,
) -> float:
    x_values, y_values = drop_missing_observations(x, y)
    if len(x_values) < 3:
        return float("nan")
    if method == "pearson":
        return (
            float("nan")
            if np.std(x_values) == 0 or np.std(y_values) == 0
            else float(stats.pearsonr(x_values, y_values).statistic)
        )
    if method == "spearman":
        return (
            float("nan")
            if np.std(x_values) == 0 or np.std(y_values) == 0
            else float(stats.spearmanr(x_values, y_values).statistic)
        )
    if method == "mi":
        return compute_mutual_information(
            x_values,
            y_values,
            x_discrete=left_spec.is_discrete,
            y_discrete=right_spec.is_discrete,
        )
    return bicor(x_values, y_values)


def _build_density_table(
    prepared: Mapping[str, pd.DataFrame],
    specs: Mapping[str, DatasetSpec],
    pairs: Sequence[FeaturePair],
    *,
    identifier: str,
    config: FeatureConfig,
    progress: bool,
) -> pd.DataFrame:
    entities = _feature_entities(prepared, pairs, config.entity_universe)
    output = pd.DataFrame({identifier: entities})
    entity_index = {entity: index for index, entity in enumerate(entities)}
    for pair in tqdm(pairs, desc="PDM feature blocks", disable=not progress):
        left, right = _align_pair(
            prepared[pair.left], prepared[pair.right], specs[pair.left], specs[pair.right]
        )
        left_spec, right_spec = specs[pair.left], specs[pair.right]
        left_levels, right_levels = _levels(left, left_spec), _levels(right, right_spec)
        left_width = config.density_bins if left_levels is None else len(left_levels)
        right_width = config.density_bins if right_levels is None else len(right_levels)
        columns = [f"{pair.left}.{pair.right}.{index}" for index in range(left_width * right_width)]
        block = pd.DataFrame(np.nan, index=range(len(entities)), columns=columns)
        left_rows, right_rows = _row_lookup(left), _row_lookup(right)
        left_scale, right_scale = (
            _safe_scale(_numeric_values(left)),
            _safe_scale(_numeric_values(right)),
        )
        sigma = _pair_sigma(left, right, left_spec, right_spec)
        for entity in set(left_rows) & set(right_rows):
            if entity not in entity_index:
                continue
            x, y = left_rows[entity], right_rows[entity]
            if left_levels is None:
                x, x_centers = _continuous_axis(x, config.density_bins, left_scale)
                x_discrete = False
            else:
                x, x_centers, x_discrete = (
                    _snap_to_levels(x, left_levels, left_spec.name),
                    left_levels,
                    True,
                )
            if right_levels is None:
                y, y_centers = _continuous_axis(y, config.density_bins, right_scale)
                y_discrete = False
            else:
                y, y_centers, y_discrete = (
                    _snap_to_levels(y, right_levels, right_spec.name),
                    right_levels,
                    True,
                )
            values = density_map(
                x,
                y,
                x_centers,
                y_centers,
                x_discrete=x_discrete,
                y_discrete=y_discrete,
                sigma=sigma,
            )
            if values.sum() > 0:
                block.iloc[entity_index[entity]] = values.ravel()
        offset = (
            1 / max(left.shape[1] - 1, 1)
            if config.log_offset in {None, "auto"}
            else float(config.log_offset)
        )
        output = pd.concat([output, np.log(block + offset)], axis=1)
    return (
        output.dropna().reset_index(drop=True)
        if config.drop_missing
        else output.reset_index(drop=True)
    )


def _build_statistic_table(
    prepared: Mapping[str, pd.DataFrame],
    specs: Mapping[str, DatasetSpec],
    pairs: Sequence[FeaturePair],
    *,
    identifier: str,
    method: str,
    config: FeatureConfig,
    progress: bool,
) -> pd.DataFrame:
    entities = _feature_entities(prepared, pairs, config.entity_universe)
    output = pd.DataFrame({identifier: entities})
    entity_index = {entity: index for index, entity in enumerate(entities)}
    for pair in tqdm(pairs, desc=f"{method} feature blocks", disable=not progress):
        left, right = _align_pair(
            prepared[pair.left], prepared[pair.right], specs[pair.left], specs[pair.right]
        )
        block = pd.DataFrame(
            np.nan, index=range(len(entities)), columns=[f"{pair.left}.{pair.right}"]
        )
        left_rows, right_rows = _row_lookup(left), _row_lookup(right)
        for entity in set(left_rows) & set(right_rows):
            if entity in entity_index:
                block.iloc[entity_index[entity], 0] = _association(
                    left_rows[entity],
                    right_rows[entity],
                    method=method,
                    left_spec=specs[pair.left],
                    right_spec=specs[pair.right],
                )
        output = pd.concat([output, block], axis=1)
    return (
        output.dropna().reset_index(drop=True)
        if config.drop_missing
        else output.reset_index(drop=True)
    )


def build_features(
    datasets: Mapping[str, pd.DataFrame],
    *,
    specs: DatasetSpecs = None,
    pairs: Sequence[FeaturePair] | None = None,
    methods: Sequence[str] = FEATURE_METHODS,
    config: FeatureConfig | None = None,
    output_id_column: str | None = None,
    progress: bool = False,
) -> dict[str, pd.DataFrame]:
    """Build PDM and scalar association features from in-memory matrices.

    File loading is intentionally separate; use :func:`pdm_learn.load_matrices`
    before calling this function.  The return value maps each requested method
    name to a feature table.
    """
    normalized_datasets = _normalize_datasets(datasets)
    if config is not None and not isinstance(config, FeatureConfig):
        raise TypeError("config must be a FeatureConfig.")
    configuration = config or FeatureConfig()
    if pairs is not None and configuration.pairs is not None:
        raise ValueError("Pass feature pairs through either pairs or config.pairs, not both.")
    resolved_pairs = _resolve_pairs(
        normalized_datasets, pairs if pairs is not None else configuration.pairs
    )
    resolved_specs = resolve_dataset_specs(normalized_datasets, specs)
    prepared = {
        name: prepare_matrix(table, resolved_specs[name])
        for name, table in normalized_datasets.items()
    }
    method_names = tuple(dict.fromkeys(str(method).strip().lower() for method in methods))
    invalid = sorted(set(method_names) - set(FEATURE_METHODS))
    if invalid:
        raise ValueError(f"Unsupported feature methods: {', '.join(invalid)}.")
    if not method_names:
        raise ValueError("Provide at least one feature method.")
    identifier = output_id_column or str(prepared[resolved_pairs[0].left].columns[0])

    output: dict[str, pd.DataFrame] = {}
    if "pdm" in method_names:
        output["pdm"] = _build_density_table(
            prepared,
            resolved_specs,
            resolved_pairs,
            identifier=identifier,
            config=configuration,
            progress=progress,
        )
    for method in method_names:
        if method != "pdm":
            output[method] = _build_statistic_table(
                prepared,
                resolved_specs,
                resolved_pairs,
                identifier=identifier,
                method=method,
                config=configuration,
                progress=progress,
            )
    return output


def build_density_features(datasets: Mapping[str, pd.DataFrame], **kwargs: object) -> pd.DataFrame:
    """Build only log-transformed density-map features."""
    return build_features(datasets, methods=("pdm",), **kwargs)["pdm"]


def build_statistic_features(
    datasets: Mapping[str, pd.DataFrame],
    *,
    method: str = "pearson",
    **kwargs: object,
) -> pd.DataFrame:
    """Build one scalar association feature table."""
    method_name = str(method).strip().lower()
    if method_name not in STATISTIC_METHODS:
        raise ValueError(f"method must be one of: {', '.join(STATISTIC_METHODS)}.")
    return build_features(datasets, methods=(method_name,), **kwargs)[method_name]


__all__ = [
    "FEATURE_METHODS",
    "STATISTIC_METHODS",
    "FeatureConfig",
    "FeaturePair",
    "bicor",
    "build_density_features",
    "build_features",
    "build_statistic_features",
    "compute_mutual_information",
    "density_map",
    "drop_missing_observations",
]
