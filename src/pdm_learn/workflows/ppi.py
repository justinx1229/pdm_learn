"""Optional DepMap recipe for ordered protein-pair feature construction.

The recipe contains only the conventional DepMap defaults. Pair-table feature
construction is provided by :mod:`pdm_learn.pair_features` and works for any
two entity types.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
from itertools import combinations
from pathlib import Path

import pandas as pd

from ..datasets import load_matrices
from ..pair_features import PairFeatureConfig, build_pair_features
from ..presets import DepMapPreset, depmap_ppi

PRESET = depmap_ppi
"""Default configuration for the DepMap protein-pair recipe."""

GENES_COLUMN = "Representative Genes (Core Members)"
COMPLEX_NAME_COLUMN = "Complex Name"


def canonical_pair(gene_a: object, gene_b: object) -> str:
    """Return an order-independent identifier for a gene pair."""
    left, right = sorted((str(gene_a).strip(), str(gene_b).strip()))
    return f"{left}.{right}"


def parse_gene_list(value: object) -> list[str]:
    """Parse the cancer-complex workbook's list-like gene column."""
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            value = value.split(",")
    if not isinstance(value, (list, tuple, set)):
        value = [value]
    return [str(gene).strip() for gene in value if str(gene).strip()]


def load_biogrid_pairs(path: str | Path) -> pd.DataFrame:
    """Load a BioGRID pair table and add stable pair identifiers.

    The input may name its two gene columns ``Gene_A``/``Gene_B`` or
    ``Gene1``/``Gene2``.  The original orientation is retained for feature
    construction; ``canonical_pair`` is the order-independent label key.
    """
    pairs = pd.read_csv(path).rename(columns={"Gene_A": "Gene1", "Gene_B": "Gene2"})
    if not {"Gene1", "Gene2"}.issubset(pairs.columns):
        raise ValueError("BioGRID input must contain Gene_A/Gene_B or Gene1/Gene2 columns.")
    pairs = pairs.dropna(subset=["Gene1", "Gene2"]).copy()
    pairs["Gene1"] = pairs["Gene1"].astype("string").str.strip()
    pairs["Gene2"] = pairs["Gene2"].astype("string").str.strip()
    pairs = pairs[pairs["Gene1"].ne("") & pairs["Gene2"].ne("")].reset_index(drop=True)
    gene_pairs = zip(pairs["Gene1"], pairs["Gene2"], strict=True)
    pairs["pair"] = [f"{left}.{right}" for left, right in gene_pairs]
    pairs["canonical_pair"] = [
        canonical_pair(left, right)
        for left, right in zip(pairs["Gene1"], pairs["Gene2"], strict=True)
    ]
    return pairs


def load_cancer_complexes(path: str | Path) -> pd.DataFrame:
    """Load the curated cancer-complex table from CSV or Excel."""
    source = Path(path)
    if source.suffix.lower() in {".xls", ".xlsx"}:
        complexes = pd.read_excel(source)
    else:
        complexes = pd.read_csv(source)
    if "Unnamed: 0" in complexes.columns:
        complexes = complexes.drop(columns="Unnamed: 0")
    if GENES_COLUMN not in complexes.columns:
        raise ValueError(f"Expected column {GENES_COLUMN!r} in cancer-complex table.")
    if COMPLEX_NAME_COLUMN not in complexes.columns:
        complexes = complexes.copy()
        complexes[COMPLEX_NAME_COLUMN] = [f"Complex {index + 1}" for index in range(len(complexes))]
    return complexes


def cancer_complex_pair_set(
    cancer_complexes: pd.DataFrame,
    *,
    genes_column: str = GENES_COLUMN,
) -> set[str]:
    """Return all order-independent gene pairs represented by curated complexes."""
    if genes_column not in cancer_complexes.columns:
        raise ValueError(f"Cancer-complex table does not contain {genes_column!r}.")
    pairs: set[str] = set()
    for genes in cancer_complexes[genes_column].map(parse_gene_list):
        for gene_a, gene_b in combinations(dict.fromkeys(genes), 2):
            pairs.add(canonical_pair(gene_a, gene_b))
    return pairs


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
    pair_table: pd.DataFrame,
    *,
    left_id_column: str = "Gene1",
    right_id_column: str = "Gene2",
    methods: Sequence[str] = ("pdm",),
    config: PairFeatureConfig | None = None,
    progress: bool = False,
    preset: DepMapPreset = PRESET,
) -> dict[str, pd.DataFrame]:
    """Build the standard DepMap feature blocks for an ordered pair table."""
    feature_config = config or PairFeatureConfig(
        density_bins=int(preset.defaults["density_bins"]),
        log_offset=preset.defaults["log_offset"],
        drop_missing=bool(preset.defaults["drop_missing"]),
    )
    return build_pair_features(
        datasets,
        pair_table,
        left_id_column=left_id_column,
        right_id_column=right_id_column,
        specs=preset.make_dataset_specs(),
        pairs=preset.make_feature_pairs(),
        methods=methods,
        config=feature_config,
        progress=progress,
    )


__all__ = [
    "COMPLEX_NAME_COLUMN",
    "GENES_COLUMN",
    "PRESET",
    "build_features",
    "cancer_complex_pair_set",
    "canonical_pair",
    "load_biogrid_pairs",
    "load_cancer_complexes",
    "load_inputs",
    "parse_gene_list",
]
