"""The maintained DepMap oncogene recipe.

This module owns the DepMap-specific defaults: assay names, shared-trimmed
file locations, and the historical ten feature pairs.  Computation belongs in
the generic package API.  Selected historical helpers are re-exported lazily
so older projects can migrate without a flag day.

For a new analysis, prefer this compact flow:

.. code-block:: python

    from pdm_learn.workflows import oncogene

    datasets = oncogene.load_inputs("data")
    feature_tables = oncogene.build_features(datasets)
    ranking = oncogene.rank_candidates(feature_tables["pdm"], positive_genes)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from importlib import import_module
from pathlib import Path
from typing import Any

from ..presets import DepMapPreset, depmap_oncogene


PRESET = depmap_oncogene
"""The default dataset, pair, and parameter configuration for this recipe."""

_STATISTIC_METHODS = frozenset({"pearson", "spearman", "mi", "bicor"})
_FEATURE_METHODS = frozenset({"pdm", *_STATISTIC_METHODS})

_LEGACY_EXPORTS = frozenset(
    {
        "COPY_NUMBER_LEVELS",
        "DEFAULT_ONCOGENE_DATASET_SPECS",
        "DEFAULT_ONCOGENE_PAIR_SPECS",
        "ONCOGENE_HEATMAP_DIMENSIONS",
        "ONCOGENE_PAIRS",
        "GeneDatasetSpec",
        "GenePairSpec",
        "OncogeneDatasetSpec",
        "OncogenePairSpec",
        "bicor",
        "build_gene_correlation_features",
        "build_gene_density_features",
        "build_gene_statistic_features",
        "build_oncogene_correlation_features",
        "build_oncogene_density_features",
        "build_oncogene_statistic_features",
        "compute_mutual_information",
        "evaluate_method_curves",
        "load_gene_feature_sets",
        "load_oncogene_feature_sets",
        "load_oncogene_inputs",
        "load_shared_trimmed_oncogene_inputs",
        "oncogene_gene_list",
        "plot_method_curves",
        "rank_candidate_genes",
        "rank_candidate_oncogenes",
        "save_gene_feature_tables",
        "save_oncogene_feature_tables",
        "snap_copy_number_levels",
        "standardize_oncogene_matrix",
        "standardize_oncogene_mutations",
        "trim_oncogene_input_tables",
    }
)


def _legacy_module() -> Any:
    """Import the old implementation only when a compatibility path is used."""
    return import_module("pdm_learn.oncogene")


def _normalize_methods(methods: Sequence[str] | None, preset: DepMapPreset) -> tuple[str, ...]:
    selected = methods if methods is not None else preset.defaults["methods"]
    normalized = tuple(dict.fromkeys(str(method).strip().lower() for method in selected))
    if not normalized:
        raise ValueError("Provide at least one feature method.")
    invalid = sorted(set(normalized) - _FEATURE_METHODS)
    if invalid:
        raise ValueError(f"Unsupported feature methods: {', '.join(invalid)}")
    return normalized


def load_inputs(
    data_dir: str | Path,
    input_paths: Mapping[str, str | Path] | None = None,
    *,
    preset: DepMapPreset = PRESET,
) -> dict[str, Any]:
    """Load the five shared-trimmed DepMap assay matrices.

    File locations are resolved by ``preset``.  Overrides may use either
    familiar file keys (``"expression"``) or neutral dataset names
    (``"gene_exp"``), making this recipe independent of the old loader's
    hard-coded layout.
    """
    from ..datasets import load_matrices

    paths = preset.resolve_input_paths(data_dir, input_paths, names="dataset")
    return load_matrices(paths, specs=preset.make_dataset_specs())


def build_features(
    datasets: Mapping[str, Any],
    *,
    preset: DepMapPreset = PRESET,
    specs: Mapping[str, object] | Sequence[object] | None = None,
    pairs: Sequence[object] | None = None,
    methods: Sequence[str] | None = None,
    density_bins: int | None = None,
    log_offset: float | str | None = None,
    drop_missing: bool | None = None,
    gene_universe: str | None = None,
    id_column: str = "gene name",
    progress: bool = False,
) -> dict[str, Any]:
    """Build requested feature tables with the DepMap oncogene defaults.

    The result maps method names to tables; ``"pdm"`` is the density-map
    table and the optional scalar tables are ``"pearson"``, ``"spearman"``,
    ``"mi"``, and ``"bicor"``.  The recipe contributes only its DepMap
    defaults; numerical work is delegated to the generic feature API.
    """
    from ..features import build_features as build_core_features

    selected_methods = _normalize_methods(methods, preset)
    resolved_specs = specs if specs is not None else preset.make_dataset_specs()
    resolved_pairs = pairs if pairs is not None else preset.make_feature_pairs()
    resolved_bins = int(density_bins if density_bins is not None else preset.defaults["density_bins"])
    if resolved_bins < 1:
        raise ValueError("density_bins must be positive.")
    resolved_offset = preset.defaults.get("log_offset", "auto") if log_offset is None else log_offset
    use_drop_missing = bool(preset.defaults["drop_missing"] if drop_missing is None else drop_missing)
    use_universe = str(preset.defaults.get("gene_universe", "union") if gene_universe is None else gene_universe)
    return build_core_features(
        datasets,
        specs=resolved_specs,
        pairs=resolved_pairs,
        methods=selected_methods,
        density_bins=resolved_bins,
        log_offset=resolved_offset,
        drop_missing=use_drop_missing,
        entity_universe=use_universe,
        output_id_column=id_column,
        progress=progress,
    )


def rank_candidates(
    feature_table: Any,
    positive_labels: Sequence[object] | set[object],
    *,
    trials: int = 100,
    model: str = "XGB",
    ks_test: bool = True,
    features_left: int | None = None,
    progress: bool = False,
) -> Any:
    """Rank unlabeled genes from a feature table and known positive labels."""
    from ..modeling import rank_candidates as rank_core_candidates

    return rank_core_candidates(
        feature_table,
        positive_labels,
        trials=trials,
        model=model,
        ks_test=ks_test,
        features_left=features_left,
        progress=progress,
    )


def __getattr__(name: str) -> Any:
    """Expose selected historical names without importing them eagerly."""
    if name in _LEGACY_EXPORTS:
        value = getattr(_legacy_module(), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _LEGACY_EXPORTS)


__all__ = [
    "PRESET",
    "build_features",
    "load_inputs",
    "rank_candidates",
    *_LEGACY_EXPORTS,
]
