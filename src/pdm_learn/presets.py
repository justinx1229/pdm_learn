"""Ready-to-use configuration for the maintained DepMap recipes.

The core :mod:`pdm_learn` API deliberately does not know about DepMap file
names or assay naming conventions.  This module is the small boundary where
those conventions live.  It contains immutable, dependency-free metadata and
does not read files or import pandas.

``depmap_oncogene`` and ``depmap_ppi`` are preset objects. They resolve input
paths and create the generic ``DatasetSpec`` and ``FeaturePair`` objects used
by the two optional workflows.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

COPY_NUMBER_LEVELS = (0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0)
"""Copy-number states used by the shared DepMap input tables."""

TRIMMED_INPUT_FILENAMES = MappingProxyType(
    {
        "expression": "Gene_Expression_Trimmed.csv",
        "copy_number": "Copy_Number_Trimmed.csv",
        "shrna": "shRNA_Trimmed.csv",
        "mutation": "Gene_Mutation_Trimmed.csv",
        "crispr": "CRISPR_Trimmed.csv",
    }
)
"""Names of the normalized input tables below ``DepMap_Trimmed``."""


@dataclass(frozen=True)
class DatasetPreset:
    """Metadata for one named matrix in a workflow preset.

    ``normalize`` uses the generic API vocabulary.  ``"row_center"`` is
    appropriate for continuous DepMap assays, while ``None`` preserves
    discrete and binary observations.
    """

    name: str
    kind: Literal["continuous", "discrete", "binary"] = "continuous"
    levels: tuple[float, ...] | None = None
    normalize: str | None = "row_center"
    id_column: str | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Dataset preset names must not be empty.")
        if self.kind not in {"continuous", "discrete", "binary"}:
            raise ValueError(f"Unsupported dataset kind: {self.kind!r}")
        if self.levels is not None and not self.levels:
            raise ValueError("Discrete levels must be non-empty when provided.")

    def as_dict(self) -> dict[str, object]:
        """Return keyword arguments suitable for a generic ``DatasetSpec``."""
        return {
            "name": self.name,
            "kind": self.kind,
            "levels": self.levels,
            "normalize": self.normalize,
            "id_column": self.id_column,
        }

    def to_dataset_spec(self) -> Any:
        """Create a :class:`pdm_learn.datasets.DatasetSpec` on demand.

        The import is local to keep this metadata module lightweight.
        """
        from .datasets import DatasetSpec

        return DatasetSpec(**self.as_dict())


@dataclass(frozen=True)
class FeaturePairPreset:
    """An ordered pair of named matrices to use for a feature block."""

    left: str
    right: str

    def __post_init__(self) -> None:
        if not self.left.strip() or not self.right.strip():
            raise ValueError("Feature-pair dataset names must not be empty.")

    def to_feature_pair(self) -> Any:
        """Create a :class:`pdm_learn.features.FeaturePair` on demand."""
        from .features import FeaturePair

        return FeaturePair(self.left, self.right)


@dataclass(frozen=True)
class DepMapPreset:
    """Immutable configuration shared by a DepMap workflow.

    ``input_names`` maps user-facing file keys (such as ``"expression"``) to
    the neutral names used in feature construction (such as ``"gene_exp"``).
    ``resolve_input_paths`` accepts either spelling and always returns neutral
    dataset names for the generic loader.
    """

    name: str
    datasets: tuple[DatasetPreset, ...]
    pairs: tuple[FeaturePairPreset, ...]
    filenames: Mapping[str, str]
    input_names: Mapping[str, str]
    defaults: Mapping[str, object] = field(default_factory=dict)
    input_directory: str = "DepMap_Trimmed"

    def __post_init__(self) -> None:
        dataset_names = tuple(dataset.name for dataset in self.datasets)
        if not self.name.strip():
            raise ValueError("Preset names must not be empty.")
        if len(set(dataset_names)) != len(dataset_names):
            raise ValueError("Dataset names must be unique within a preset.")
        if set(self.filenames) != set(self.input_names):
            raise ValueError("filenames and input_names must use the same input keys.")
        if set(self.input_names.values()) != set(dataset_names):
            raise ValueError("input_names must map exactly to the preset dataset names.")

        unknown_pairs = {
            name
            for pair in self.pairs
            for name in (pair.left, pair.right)
            if name not in dataset_names
        }
        if unknown_pairs:
            raise ValueError(
                "Feature pairs reference datasets not defined by the preset: "
                f"{', '.join(sorted(unknown_pairs))}"
            )

        object.__setattr__(self, "filenames", MappingProxyType(dict(self.filenames)))
        object.__setattr__(self, "input_names", MappingProxyType(dict(self.input_names)))
        object.__setattr__(self, "defaults", MappingProxyType(dict(self.defaults)))

    @property
    def dataset_names(self) -> tuple[str, ...]:
        """The generic matrix names in the order expected by this recipe."""
        return tuple(dataset.name for dataset in self.datasets)

    @property
    def input_keys(self) -> tuple[str, ...]:
        """The user-facing input-file keys accepted by this preset."""
        return tuple(self.input_names)

    def make_dataset_specs(self) -> tuple[Any, ...]:
        """Lazily create generic ``DatasetSpec`` instances for this preset."""
        return tuple(dataset.to_dataset_spec() for dataset in self.datasets)

    def make_feature_pairs(self) -> tuple[Any, ...]:
        """Lazily create generic ``FeaturePair`` instances for this preset."""
        return tuple(pair.to_feature_pair() for pair in self.pairs)

    def resolve_input_paths(
        self,
        data_dir: str | Path,
        input_paths: Mapping[str, str | Path] | None = None,
    ) -> dict[str, Path]:
        """Resolve default paths plus optional overrides.

        Parameters
        ----------
        data_dir:
            Directory containing the preset's ``input_directory``.
        input_paths:
            Optional paths keyed by either a user-facing input key or generic
            dataset name.  Only known preset inputs are accepted.
        """
        root = Path(data_dir) / self.input_directory
        resolved = {input_key: root / filename for input_key, filename in self.filenames.items()}
        reverse_names = {
            dataset_name: input_key for input_key, dataset_name in self.input_names.items()
        }

        for name, path in (input_paths or {}).items():
            key = str(name).strip()
            input_key = key if key in self.input_names else reverse_names.get(key)
            if input_key is None:
                accepted = ", ".join((*self.input_keys, *self.dataset_names))
                raise ValueError(
                    f"Unknown input {key!r} for {self.name}; expected one of: {accepted}."
                )
            resolved[input_key] = Path(path)

        return {self.input_names[input_key]: path for input_key, path in resolved.items()}


_DEPMAP_DATASETS = (
    DatasetPreset("gene_exp", "continuous", normalize="row_center"),
    DatasetPreset("copy_num", "discrete", COPY_NUMBER_LEVELS, normalize=None),
    DatasetPreset("shRNA", "continuous", normalize="row_center"),
    DatasetPreset("gene_mut", "binary", (0.0, 1.0), normalize=None),
    DatasetPreset("CRISPR", "continuous", normalize="row_center"),
)

_DEPMAP_INPUT_NAMES = {
    "expression": "gene_exp",
    "copy_number": "copy_num",
    "shrna": "shRNA",
    "mutation": "gene_mut",
    "crispr": "CRISPR",
}

_ONCOGENE_PAIRS = (
    FeaturePairPreset("gene_exp", "copy_num"),
    FeaturePairPreset("gene_exp", "shRNA"),
    FeaturePairPreset("gene_mut", "gene_exp"),
    FeaturePairPreset("gene_exp", "CRISPR"),
    FeaturePairPreset("shRNA", "copy_num"),
    FeaturePairPreset("gene_mut", "copy_num"),
    FeaturePairPreset("CRISPR", "copy_num"),
    FeaturePairPreset("gene_mut", "shRNA"),
    FeaturePairPreset("shRNA", "CRISPR"),
    FeaturePairPreset("gene_mut", "CRISPR"),
)

_PPI_PAIRS = tuple(
    FeaturePairPreset(left.name, right.name)
    for left in _DEPMAP_DATASETS
    for right in _DEPMAP_DATASETS
)

_COMMON_DEFAULTS = {
    "density_bins": 7,
    "log_offset": "auto",
    "drop_missing": True,
    "entity_universe": "union",
}

depmap_oncogene = DepMapPreset(
    name="depmap_oncogene",
    datasets=_DEPMAP_DATASETS,
    pairs=_ONCOGENE_PAIRS,
    filenames=TRIMMED_INPUT_FILENAMES,
    input_names=_DEPMAP_INPUT_NAMES,
    defaults={
        **_COMMON_DEFAULTS,
        "methods": ("pdm", "pearson", "spearman", "mi", "bicor"),
    },
)
"""Preset for gene-level oncogene candidate ranking with shared DepMap tables."""

depmap_ppi = DepMapPreset(
    name="depmap_ppi",
    datasets=_DEPMAP_DATASETS,
    pairs=_PPI_PAIRS,
    filenames=TRIMMED_INPUT_FILENAMES,
    input_names=_DEPMAP_INPUT_NAMES,
    defaults={
        **_COMMON_DEFAULTS,
        "pair_columns": ("Gene1", "Gene2"),
    },
)
"""Preset for pair-table PPI feature construction with shared DepMap tables."""

__all__ = [
    "COPY_NUMBER_LEVELS",
    "DatasetPreset",
    "DepMapPreset",
    "FeaturePairPreset",
    "TRIMMED_INPUT_FILENAMES",
    "depmap_oncogene",
    "depmap_ppi",
]
