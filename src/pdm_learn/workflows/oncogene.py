"""Optional DepMap recipe for row-level candidate feature construction.

This module supplies only DepMap file names, matrix specifications, and the
established assay-pair selection. All computation remains in the generic
``pdm_learn`` API.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from ..datasets import load_matrices
from ..features import FEATURE_METHODS, FeatureConfig
from ..features import build_features as build_core_features
from ..presets import DepMapPreset, depmap_oncogene

PRESET = depmap_oncogene
"""Default configuration for the DepMap oncogene recipe."""


def load_inputs(
    data_dir: str | Path,
    input_paths: Mapping[str, str | Path] | None = None,
    *,
    preset: DepMapPreset = PRESET,
) -> dict[str, pd.DataFrame]:
    """Load the recipe's standardized DepMap matrices from ``data_dir``."""
    return load_matrices(
        preset.resolve_input_paths(data_dir, input_paths),
        specs=preset.make_dataset_specs(),
    )


def build_features(
    datasets: Mapping[str, pd.DataFrame],
    *,
    methods: Sequence[str] = FEATURE_METHODS,
    config: FeatureConfig | None = None,
    output_id_column: str | None = None,
    progress: bool = False,
    preset: DepMapPreset = PRESET,
) -> dict[str, pd.DataFrame]:
    """Build the standard DepMap row-level PDM and association features."""
    feature_config = config or FeatureConfig(
        density_bins=int(preset.defaults["density_bins"]),
        log_offset=preset.defaults["log_offset"],
        drop_missing=bool(preset.defaults["drop_missing"]),
        entity_universe=str(preset.defaults["entity_universe"]),
    )
    return build_core_features(
        datasets,
        specs=preset.make_dataset_specs(),
        pairs=preset.make_feature_pairs(),
        methods=methods,
        config=feature_config,
        output_id_column=output_id_column,
        progress=progress,
    )


__all__ = ["PRESET", "build_features", "load_inputs"]
