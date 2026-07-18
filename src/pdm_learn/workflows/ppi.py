"""The maintained DepMap protein-interaction recipe.

This module contains the small amount of PPI-specific policy that belongs
outside the reusable library: the five DepMap assays, their default input
paths, and the conventional ``Gene1``/``Gene2`` pair columns.  Feature
construction itself is delegated to :mod:`pdm_learn.pair_features`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from importlib import import_module
from pathlib import Path
from typing import Any

from ..pair_features import PairFeatureConfig, build_pair_features
from ..presets import DepMapPreset, depmap_ppi


PRESET = depmap_ppi
"""Default dataset, pair, and parameter configuration for the PPI recipe."""

_LEGACY_EXPORTS = frozenset(
    {
        "canonical_pair",
        "build_control_feature_tables",
        "cancer_complex_pair_set",
        "derive_shared_trimmed_inputs",
        "evaluate_ppi_controls",
        "load_biogrid_pairs",
        "load_cancer_complexes",
        "load_ppi_inputs",
        "parse_gene_list",
        "rank_ppi_pairs_cancer",
        "sample_negative_control_pairs",
        "sample_positive_control_pairs",
    }
)


def _legacy_module() -> Any:
    """Load the compatibility implementation only for a legacy attribute."""
    return import_module("pdm_learn.ppi")


def load_inputs(
    data_dir: str | Path,
    input_paths: Mapping[str, str | Path] | None = None,
    *,
    preset: DepMapPreset = PRESET,
) -> dict[str, Any]:
    """Load the shared-trimmed DepMap matrices used by the PPI recipe."""
    from ..datasets import load_matrices

    paths = preset.resolve_input_paths(data_dir, input_paths, names="dataset")
    return load_matrices(paths, specs=preset.make_dataset_specs())


def build_features(
    datasets: Mapping[str, Any],
    pair_table: Any,
    *,
    left_id_column: object = "Gene1",
    right_id_column: object = "Gene2",
    specs: Mapping[str, object] | Sequence[object] | None = None,
    pairs: Sequence[object] | None = None,
    methods: Sequence[str] = ("pdm",),
    density_bins: int | None = None,
    log_offset: float | str | None = None,
    drop_missing: bool | None = None,
    progress: bool = False,
    preset: DepMapPreset = PRESET,
) -> dict[str, Any]:
    """Build features for a protein-pair table using generic pair builders."""
    resolved_specs = specs if specs is not None else preset.make_dataset_specs()
    resolved_pairs = pairs if pairs is not None else preset.make_feature_pairs()
    config = PairFeatureConfig(
        pairs=None,
        density_bins=int(density_bins if density_bins is not None else preset.defaults["density_bins"]),
        log_offset=preset.defaults.get("log_offset", "auto") if log_offset is None else log_offset,
        drop_missing=bool(preset.defaults["drop_missing"] if drop_missing is None else drop_missing),
    )
    return build_pair_features(
        datasets,
        pair_table,
        left_id_column=left_id_column,
        right_id_column=right_id_column,
        specs=resolved_specs,
        pairs=resolved_pairs,
        methods=methods,
        config=config,
        progress=progress,
    )


def rank_candidates(
    feature_table: Any,
    cancer_complexes: Any,
    **kwargs: object,
) -> Any:
    """Rank candidate interactions using the established cancer-complex recipe."""
    return _legacy_module().rank_ppi_pairs_cancer(feature_table, cancer_complexes, **kwargs)


def __getattr__(name: str) -> Any:
    """Expose selected historical helpers without importing them eagerly."""
    if name in _LEGACY_EXPORTS:
        value = getattr(_legacy_module(), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _LEGACY_EXPORTS)


__all__ = ["PRESET", "build_features", "load_inputs", "rank_candidates", *_LEGACY_EXPORTS]
