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

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

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


def control_gene_universe(datasets: Mapping[str, pd.DataFrame]) -> list[str]:
    """Return genes available in every assay used by the PPI feature recipe."""
    if not datasets:
        raise ValueError("Provide at least one dataset to derive the control gene universe.")
    gene_sets = [
        set(table.iloc[:, 0].astype("string").str.strip())
        for table in datasets.values()
    ]
    return sorted(set.intersection(*gene_sets))


def sample_positive_control_pairs(
    cancer_complexes: pd.DataFrame,
    *,
    gene_universe: Sequence[str],
    max_pairs_per_complex: int = 5,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Sample disjoint, available within-complex pairs for one positive control set."""
    if max_pairs_per_complex < 1:
        raise ValueError("max_pairs_per_complex must be a positive integer.")
    generator = rng or np.random.default_rng()
    allowed = set(gene_universe)
    rows: list[dict[str, str]] = []
    for _, complex_row in cancer_complexes.iterrows():
        genes = [
            gene
            for gene in dict.fromkeys(parse_gene_list(complex_row[GENES_COLUMN]))
            if gene in allowed
        ]
        generator.shuffle(genes)
        pair_count = min(max_pairs_per_complex, len(genes) // 2)
        for index in range(pair_count):
            left, right = genes[2 * index], genes[2 * index + 1]
            rows.append(
                {
                    COMPLEX_NAME_COLUMN: str(complex_row[COMPLEX_NAME_COLUMN]).strip(),
                    "Gene1": left,
                    "Gene2": right,
                    "pair": f"{left}.{right}",
                    "canonical_pair": canonical_pair(left, right),
                }
            )
    if not rows:
        raise ValueError("No positive control pairs could be sampled from the supplied complexes.")
    return pd.DataFrame(rows)


def sample_negative_control_pairs(
    cancer_complexes: pd.DataFrame,
    *,
    gene_universe: Sequence[str],
    candidate_pairs: pd.DataFrame,
    count: int,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Sample unique, non-complex pairs from a supplied candidate set.

    These are exclusion-based negative controls: pairs absent from the curated
    complex catalogue, not a claim of experimentally verified noninteraction.
    ``candidate_pairs`` must explicitly define the population being sampled
    (for example, BioGRID pairs or a separately generated random-pair table).
    """
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("count must be a positive integer.")
    if not isinstance(candidate_pairs, pd.DataFrame):
        raise TypeError("candidate_pairs must be a pandas DataFrame.")
    if not {"Gene1", "Gene2"}.issubset(candidate_pairs.columns):
        raise ValueError("candidate_pairs must contain Gene1 and Gene2 columns.")

    candidates = candidate_pairs.copy()
    candidates["Gene1"] = candidates["Gene1"].astype("string").str.strip()
    candidates["Gene2"] = candidates["Gene2"].astype("string").str.strip()
    candidates = candidates.loc[
        candidates["Gene1"].notna()
        & candidates["Gene2"].notna()
        & candidates["Gene1"].ne("")
        & candidates["Gene2"].ne("")
        & candidates["Gene1"].ne(candidates["Gene2"])
    ].copy()
    candidates["canonical_pair"] = [
        canonical_pair(left, right)
        for left, right in zip(candidates["Gene1"], candidates["Gene2"], strict=True)
    ]
    candidates["pair"] = [
        f"{left}.{right}"
        for left, right in zip(candidates["Gene1"], candidates["Gene2"], strict=True)
    ]

    allowed = set(gene_universe)
    excluded = cancer_complex_pair_set(cancer_complexes)
    eligible = candidates.loc[
        candidates["Gene1"].isin(allowed)
        & candidates["Gene2"].isin(allowed)
        & ~candidates["canonical_pair"].isin(excluded)
    ].drop_duplicates(subset="canonical_pair", keep="first")
    if count > len(eligible):
        raise ValueError(
            "Requested "
            f"{count} negative controls, but the supplied candidate table contains only "
            f"{len(eligible)} eligible non-complex pairs."
        )
    generator = rng or np.random.default_rng()
    selected_indices = generator.choice(len(eligible), size=count, replace=False)
    return eligible.iloc[selected_indices].reset_index(drop=True)


def build_control_feature_tables(
    datasets: Mapping[str, pd.DataFrame],
    cancer_complexes: pd.DataFrame,
    *,
    negative_candidate_pairs: pd.DataFrame,
    controls: int = 10,
    max_pairs_per_complex: int = 5,
    negative_pairs: int = 5_000,
    random_state: int = 42,
    progress: bool = False,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    """Generate matched PDM feature tables for reproducible PPI control sets."""
    if isinstance(controls, bool) or not isinstance(controls, int) or controls < 1:
        raise ValueError("controls must be a positive integer.")
    universe = control_gene_universe(datasets)
    positives: list[pd.DataFrame] = []
    negatives: list[pd.DataFrame] = []
    for index in tqdm(
        range(controls), desc="PPI control sets completed", unit="set", disable=not progress
    ):
        positive_pairs = sample_positive_control_pairs(
            cancer_complexes,
            gene_universe=universe,
            max_pairs_per_complex=max_pairs_per_complex,
            rng=np.random.default_rng(random_state + index),
        )
        negative_pairs_table = sample_negative_control_pairs(
            cancer_complexes,
            gene_universe=universe,
            candidate_pairs=negative_candidate_pairs,
            count=negative_pairs,
            rng=np.random.default_rng(random_state + 10_000 + index),
        )
        positives.append(
            build_symmetric_density_features(datasets, positive_pairs, progress=False)
        )
        negatives.append(
            build_symmetric_density_features(datasets, negative_pairs_table, progress=False)
        )
    return positives, negatives


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


def build_symmetric_density_features(
    datasets: Mapping[str, pd.DataFrame],
    pair_table: pd.DataFrame,
    *,
    progress: bool = False,
    preset: DepMapPreset = PRESET,
) -> pd.DataFrame:
    """Build one orientation-invariant PDM row for each undirected gene pair."""
    if not {"Gene1", "Gene2"}.issubset(pair_table.columns):
        raise ValueError("PPI pairs must contain Gene1 and Gene2 columns.")
    if "_pair_row" in pair_table.columns:
        raise ValueError("PPI pairs cannot use the reserved '_pair_row' column.")

    forward_pairs = pair_table.copy()
    if "pair" not in forward_pairs.columns:
        forward_pairs["pair"] = [
            f"{left}.{right}"
            for left, right in zip(forward_pairs["Gene1"], forward_pairs["Gene2"], strict=True)
        ]
    if "canonical_pair" not in forward_pairs.columns:
        forward_pairs["canonical_pair"] = [
            canonical_pair(left, right)
            for left, right in zip(forward_pairs["Gene1"], forward_pairs["Gene2"], strict=True)
        ]
    forward_pairs["_pair_row"] = range(len(forward_pairs))

    reverse_pairs = forward_pairs.copy()
    reverse_pairs["Gene1"] = forward_pairs["Gene2"].astype(object).tolist()
    reverse_pairs["Gene2"] = forward_pairs["Gene1"].astype(object).tolist()
    reverse_pairs["pair"] = [
        f"{left}.{right}"
        for left, right in zip(reverse_pairs["Gene1"], reverse_pairs["Gene2"], strict=True)
    ]

    forward = build_features(
        datasets,
        forward_pairs,
        methods=("pdm",),
        progress=progress,
        preset=preset,
    )["pdm"]
    reverse = build_features(
        datasets,
        reverse_pairs,
        methods=("pdm",),
        progress=progress,
        preset=preset,
    )["pdm"]
    feature_columns = pdm_feature_columns(forward, preset=preset)
    if not feature_columns:
        raise ValueError("PPI feature construction did not generate any PDM columns.")
    reverse_by_row = reverse.set_index("_pair_row")
    if not forward["_pair_row"].isin(reverse_by_row.index).all():
        raise ValueError("Forward and reverse PPI feature rows do not align.")
    forward.loc[:, feature_columns] = (
        forward.loc[:, feature_columns].to_numpy(dtype=float)
        + reverse_by_row.loc[forward["_pair_row"], feature_columns].to_numpy(dtype=float)
    ) / 2.0
    return (
        forward.drop(columns="_pair_row")
        .drop_duplicates(subset="canonical_pair", keep="first")
        .reset_index(drop=True)
    )


def pdm_feature_columns(
    feature_table: pd.DataFrame,
    *,
    preset: DepMapPreset = PRESET,
) -> list[object]:
    """Return only PDM columns generated by this recipe, excluding source metadata."""
    prefixes = tuple(f"{pair.left}.{pair.right}." for pair in preset.pairs)
    return [
        column
        for column in feature_table.columns
        if isinstance(column, str)
        and column.startswith(prefixes)
        and column.rsplit(".", maxsplit=1)[-1].isdigit()
    ]


__all__ = [
    "COMPLEX_NAME_COLUMN",
    "GENES_COLUMN",
    "PRESET",
    "build_features",
    "build_control_feature_tables",
    "build_symmetric_density_features",
    "cancer_complex_pair_set",
    "canonical_pair",
    "control_gene_universe",
    "load_biogrid_pairs",
    "load_cancer_complexes",
    "load_inputs",
    "parse_gene_list",
    "pdm_feature_columns",
    "sample_negative_control_pairs",
    "sample_positive_control_pairs",
]
