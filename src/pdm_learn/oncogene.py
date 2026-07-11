from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mutual_info_score
from tqdm.auto import tqdm

from pdm_learn.modeling import KFold_PR, LOOCV
from pdm_learn.preprocessing import densitymap, drop_nan


COPY_NUMBER_LEVELS = np.array([0, 1, 2, 3, 4, 6, 8], dtype=float)
ONCOGENE_TRIMMED_FILENAMES = {
    "expression": "CCLE_gene_expression_trimmed_Wei.csv",
    "copy_number": "CCLE_gene_cn_trimmed_Wei.csv",
    "shrna": "shRNA_Broad_Trimmed_Wei.csv",
    "mutation": "CCLE_gene_mutation_trimmed_Wei.csv",
    "crispr": "Avana_gene_effect_20Q3_Trimmed_Wei.csv",
}
SHARED_TRIMMED_FILENAMES = {
    "expression": "Gene_Expression_Trimmed.csv",
    "copy_number": "Copy_Number_Trimmed.csv",
    "shrna": "shRNA_Trimmed.csv",
    "mutation": "Gene_Mutation_Trimmed.csv",
    "crispr": "CRISPR_Trimmed.csv",
}
ONCOGENE_HEATMAP_DIMENSIONS = (
    (7, 7),
    (7, 7),
    (7, 2),
    (7, 7),
    (7, 7),
    (7, 2),
    (7, 7),
    (7, 2),
    (7, 7),
    (7, 2),
)
ONCOGENE_PAIRS = (
    ("gene_exp", "copy_num", ("cn",)),
    ("gene_exp", "shRNA", ()),
    ("gene_mut", "gene_exp", ("mut",)),
    ("gene_exp", "CRISPR", ()),
    ("shRNA", "copy_num", ("cn",)),
    ("gene_mut", "copy_num", ("cn", "mut")),
    ("CRISPR", "copy_num", ("cn",)),
    ("gene_mut", "shRNA", ("mut",)),
    ("shRNA", "CRISPR", ()),
    ("gene_mut", "CRISPR", ("mut",)),
)


@dataclass(frozen=True)
class OncogeneDatasetSpec:
    """Feature-building metadata for one gene-by-sample matrix."""

    name: str
    kind: str = "continuous"
    levels: tuple[float, ...] | None = None
    normalize: bool | None = None

    @property
    def is_continuous(self) -> bool:
        return self.kind == "continuous"

    @property
    def is_discrete(self) -> bool:
        return not self.is_continuous


@dataclass(frozen=True)
class OncogenePairSpec:
    left: str
    right: str


DEFAULT_ONCOGENE_DATASET_SPECS = {
    "gene_exp": OncogeneDatasetSpec("gene_exp", "continuous"),
    "copy_num": OncogeneDatasetSpec("copy_num", "discrete", tuple(COPY_NUMBER_LEVELS), normalize=False),
    "shRNA": OncogeneDatasetSpec("shRNA", "continuous"),
    "gene_mut": OncogeneDatasetSpec("gene_mut", "binary", (0.0, 1.0), normalize=False),
    "CRISPR": OncogeneDatasetSpec("CRISPR", "continuous"),
}
DEFAULT_ONCOGENE_PAIR_SPECS = tuple(
    OncogenePairSpec(left, right)
    for left, right, _ in ONCOGENE_PAIRS
)
LEGACY_ONCOGENE_INPUT_NAMES = {
    "expression": "gene_exp",
    "copy_number": "copy_num",
    "shrna": "shRNA",
    "mutation": "gene_mut",
    "crispr": "CRISPR",
}
STATISTIC_METHOD_FILENAMES = {
    "pearson": "pearson.csv",
    "spearman": "spearman.csv",
    "mi": "mi.csv",
    "bicor": "bicor.csv",
}


def _standardize_gene_names(dataframe: pd.DataFrame) -> pd.DataFrame:
    output = dataframe.copy()
    output.iloc[:, 0] = output.iloc[:, 0].astype(str).str.strip()
    return output


def _first_matching_column(dataframe: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    normalized = {str(column).strip().lower(): column for column in dataframe.columns}
    for candidate in candidates:
        match = normalized.get(candidate.lower())
        if match is not None:
            return str(match)
    return None


def _clean_gene_symbol(value: object) -> str:
    text = str(value).strip()
    if text.endswith(")") and " (" in text:
        text = text.rsplit(" (", maxsplit=1)[0]
    return text.strip()


def _deduplicate_gene_rows(dataframe: pd.DataFrame) -> pd.DataFrame:
    gene_column = dataframe.columns[0]
    numeric = dataframe.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    output = pd.concat([dataframe[[gene_column]], numeric], axis=1)
    output = output.groupby(gene_column, as_index=False, sort=True).mean(numeric_only=True)
    return output


def standardize_oncogene_matrix(
    dataframe: pd.DataFrame,
    *,
    orientation: str = "genes-as-rows",
    gene_column: str | None = None,
    sample_column: str | None = None,
) -> pd.DataFrame:
    """Standardize a raw CCLE/DepMap matrix to genes x cell lines.

    The downstream oncogene workflow expects one gene-name column followed by
    cell-line columns. Raw DepMap releases are not always oriented the same way,
    so this helper supports either genes-as-rows or samples-as-rows input.
    """
    if orientation not in {"genes-as-rows", "samples-as-rows"}:
        raise ValueError("orientation must be 'genes-as-rows' or 'samples-as-rows'")

    input_df = dataframe.copy()
    input_df.columns = [str(column).strip() for column in input_df.columns]

    if orientation == "genes-as-rows":
        if gene_column is None:
            gene_column = _first_matching_column(
                input_df,
                ("gene name", "gene", "hugo_symbol", "hugo symbol", "symbol", "description", "unnamed: 0"),
            )
        gene_column = gene_column or str(input_df.columns[0])
        if gene_column not in input_df.columns:
            raise ValueError(f"Gene column {gene_column!r} was not found.")

        output = input_df.copy()
        output.insert(0, "gene name", output.pop(gene_column).map(_clean_gene_symbol))
        output = output.loc[output["gene name"].ne("")].reset_index(drop=True)
        output.columns = [str(column).strip() for column in output.columns]
        return _deduplicate_gene_rows(output)

    if sample_column is None:
        sample_column = _first_matching_column(
            input_df,
            ("cell line", "cell_line", "cellline", "ccle_name", "modelid", "model_id", "depmap_id", "unnamed: 0"),
        )
    sample_column = sample_column or str(input_df.columns[0])
    if sample_column not in input_df.columns:
        raise ValueError(f"Sample/cell-line column {sample_column!r} was not found.")

    sample_names = input_df.pop(sample_column).astype(str).str.strip()
    numeric = input_df.apply(pd.to_numeric, errors="coerce")
    numeric.index = sample_names
    output = numeric.T.reset_index()
    output.insert(0, "gene name", output.pop("index").map(_clean_gene_symbol))
    output.columns = ["gene name", *[str(column).strip() for column in output.columns[1:]]]
    return _deduplicate_gene_rows(output)


def standardize_oncogene_mutations(
    dataframe: pd.DataFrame,
    *,
    gene_column: str | None = None,
    cell_line_column: str | None = None,
) -> pd.DataFrame:
    input_df = dataframe.copy()
    input_df.columns = [str(column).strip() for column in input_df.columns]
    gene_column = gene_column or _first_matching_column(
        input_df,
        ("Hugo_Symbol", "hugo symbol", "hugo_symbol", "gene", "gene name", "symbol"),
    )
    cell_line_column = cell_line_column or _first_matching_column(
        input_df,
        ("Cell line", "cell_line", "cellline", "CCLE_Name", "model_id", "ModelID", "DepMap_ID"),
    )
    if gene_column is None or gene_column not in input_df.columns:
        raise ValueError("Could not identify the mutation gene column.")
    if cell_line_column is None or cell_line_column not in input_df.columns:
        raise ValueError("Could not identify the mutation cell-line column.")

    output = input_df.copy()
    output["Hugo_Symbol"] = output[gene_column].map(_clean_gene_symbol)
    output["Cell line"] = output[cell_line_column].astype(str).str.strip()
    output = output.loc[output["Hugo_Symbol"].ne("") & output["Cell line"].ne("")]
    keep_columns = ["Hugo_Symbol", "Cell line"]
    extra_columns = [column for column in output.columns if column not in keep_columns]
    return output[keep_columns + extra_columns].reset_index(drop=True)


def trim_oncogene_input_tables(
    raw_paths: Mapping[str, str | Path],
    output_dir: str | Path,
    *,
    matrix_orientation: str = "genes-as-rows",
    gene_list: Sequence[str] | None = None,
    align_cell_lines: bool = True,
    gene_column: str | None = None,
    sample_column: str | None = None,
    mutation_gene_column: str | None = None,
    mutation_cell_line_column: str | None = None,
) -> dict[str, Path]:
    required = {"expression", "copy_number", "shrna", "mutation", "crispr"}
    missing = required - set(raw_paths)
    if missing:
        raise ValueError(f"Missing raw paths for: {', '.join(sorted(missing))}")

    matrices = {
        name: standardize_oncogene_matrix(
            pd.read_csv(raw_paths[name]),
            orientation=matrix_orientation,
            gene_column=gene_column,
            sample_column=sample_column,
        )
        for name in ("expression", "copy_number", "shrna", "crispr")
    }
    mutations = standardize_oncogene_mutations(
        pd.read_csv(raw_paths["mutation"], low_memory=False),
        gene_column=mutation_gene_column,
        cell_line_column=mutation_cell_line_column,
    )

    if gene_list is not None:
        allowed_genes = {str(gene).strip() for gene in gene_list}
        matrices = {
            name: dataframe.loc[dataframe.iloc[:, 0].isin(allowed_genes)].reset_index(drop=True)
            for name, dataframe in matrices.items()
        }
        mutations = mutations.loc[mutations["Hugo_Symbol"].isin(allowed_genes)].reset_index(drop=True)

    if align_cell_lines:
        shared_columns = set(matrices["expression"].columns[1:])
        for dataframe in matrices.values():
            shared_columns &= set(dataframe.columns[1:])
        ordered_columns = ["gene name", *sorted(shared_columns)]
        matrices = {name: dataframe.loc[:, ordered_columns].copy() for name, dataframe in matrices.items()}
        mutations = mutations.loc[mutations["Cell line"].isin(shared_columns)].reset_index(drop=True)

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    outputs = {
        "expression": root / ONCOGENE_TRIMMED_FILENAMES["expression"],
        "copy_number": root / ONCOGENE_TRIMMED_FILENAMES["copy_number"],
        "shrna": root / ONCOGENE_TRIMMED_FILENAMES["shrna"],
        "mutation": root / ONCOGENE_TRIMMED_FILENAMES["mutation"],
        "crispr": root / ONCOGENE_TRIMMED_FILENAMES["crispr"],
    }
    matrices["expression"].to_csv(outputs["expression"], index=False)
    matrices["copy_number"].to_csv(outputs["copy_number"], index=False)
    matrices["shrna"].to_csv(outputs["shrna"], index=False)
    matrices["crispr"].to_csv(outputs["crispr"], index=False)
    mutations.to_csv(outputs["mutation"], index=False)
    return outputs


def snap_copy_number_levels(
    dataframe: pd.DataFrame,
    levels: Sequence[float] = COPY_NUMBER_LEVELS,
) -> pd.DataFrame:
    output = _standardize_gene_names(dataframe)
    level_array = np.asarray(levels, dtype=float)
    numeric = output.iloc[:, 1:].astype(float).to_numpy() * 2
    indices = np.abs(numeric[..., np.newaxis] - level_array).argmin(axis=-1)
    output.iloc[:, 1:] = level_array[indices]
    return output


def _process_continuous_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    output = dataframe.copy()
    output.sort_index(inplace=True)
    first_column = output.iloc[:, 0].astype(str).str.strip()
    output = output.drop(columns=output.columns[0])
    output = output.sort_index(axis=1)
    output.insert(0, first_column.name, first_column)
    return output


def _normalize_rows(dataframe: pd.DataFrame) -> pd.DataFrame:
    output = dataframe.copy()
    numeric = output.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    centered = numeric.sub(numeric.mean(axis=1), axis=0)
    return pd.concat(
        [
            output.iloc[:, :1].reset_index(drop=True),
            centered.reset_index(drop=True),
        ],
        axis=1,
    )


def _trim_pair_dataframes(
    dataframe_1: pd.DataFrame,
    dataframe_2: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df1 = _process_continuous_dataframe(dataframe_1)
    df2 = _process_continuous_dataframe(dataframe_2)

    mask = df1.iloc[:, 0].isin(df2.iloc[:, 0])
    df1 = df1.loc[mask].copy()
    mask = df2.iloc[:, 0].isin(df1.iloc[:, 0])
    df2 = df2.loc[mask].copy()

    shared_columns = df1.columns.str.strip().isin(df2.columns.str.strip())
    df1 = df1.loc[:, shared_columns].copy()
    shared_columns = df2.columns.str.strip().isin(df1.columns.str.strip())
    df2 = df2.loc[:, shared_columns].copy()
    return df1, df2


def _mutation_to_reference(
    mutation: pd.DataFrame,
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_out = dataframe.copy()
    df_out.iloc[:, 0] = df_out.iloc[:, 0].astype(str).str.strip()

    first_column = str(df_out.columns[0]).strip()
    row_names = df_out.iloc[:, 0].astype(str).str.strip()
    data_columns = pd.Index(df_out.columns.astype(str).str.strip()[1:])

    mutation_columns = {str(column).strip() for column in mutation.columns}
    if {"Hugo_Symbol", "Cell line"}.issubset(mutation_columns):
        raise ValueError(
            "Oncogene analysis now requires the shared trimmed binary mutation matrix "
            "from data/DepMap_Trimmed/Gene_Mutation_Trimmed.csv."
        )

    mutation_matrix = mutation.copy()
    mutation_matrix.columns = mutation_matrix.columns.astype(str).str.strip()
    mutation_matrix.iloc[:, 0] = mutation_matrix.iloc[:, 0].astype(str).str.strip()
    mutation_matrix = mutation_matrix.drop_duplicates(subset=mutation_matrix.columns[0]).set_index(mutation_matrix.columns[0])
    mutation_matrix = mutation_matrix.apply(pd.to_numeric, errors="coerce")
    mutation_matrix = mutation_matrix.reindex(index=row_names, columns=data_columns, fill_value=0.0)
    mutation_matrix = mutation_matrix.fillna(0.0).gt(0).astype(float)
    mutation_matrix = mutation_matrix.reset_index()
    mutation_matrix.columns = [first_column, *data_columns]
    return mutation_matrix, df_out


def _extract_density_values(
    values: np.ndarray,
    *,
    cutoff: bool = False,
    std: float = 1.0,
    max_value: float = 7.0,
    boxes: int = 7,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    if not cutoff:
        return np.asarray(values, dtype=float)

    capped = np.asarray(values, dtype=float).copy()
    capped[capped > std * max_value] = std * max_value
    capped[capped < -std * max_value] = -std * max_value
    centers = np.linspace(-std * max_value, std * max_value, num=boxes * 2, endpoint=False)[1::2]
    return capped, centers


def _row_lookup(dataframe: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        str(row.iloc[0]).strip(): row.iloc[1:].to_numpy(dtype=float)
        for _, row in dataframe.iterrows()
    }


def _normalize_oncogene_input_paths(
    input_paths: Mapping[str, str | Path],
) -> dict[str, Path]:
    return {
        LEGACY_ONCOGENE_INPUT_NAMES.get(name, name): Path(path)
        for name, path in input_paths.items()
    }


def load_oncogene_inputs(
    data_dir: str | Path,
    *,
    input_paths: Mapping[str, str | Path] | None = None,
) -> dict[str, pd.DataFrame]:
    return load_shared_trimmed_oncogene_inputs(data_dir, input_paths=input_paths)


def load_shared_trimmed_oncogene_inputs(
    data_dir: str | Path,
    *,
    input_paths: Mapping[str, str | Path] | None = None,
) -> dict[str, pd.DataFrame]:
    root = Path(data_dir)
    if input_paths:
        paths = _normalize_oncogene_input_paths(input_paths)
    else:
        paths = {
            "gene_exp": root / "DepMap_Trimmed" / SHARED_TRIMMED_FILENAMES["expression"],
            "copy_num": root / "DepMap_Trimmed" / SHARED_TRIMMED_FILENAMES["copy_number"],
            "shRNA": root / "DepMap_Trimmed" / SHARED_TRIMMED_FILENAMES["shrna"],
            "gene_mut": root / "DepMap_Trimmed" / SHARED_TRIMMED_FILENAMES["mutation"],
            "CRISPR": root / "DepMap_Trimmed" / SHARED_TRIMMED_FILENAMES["crispr"],
        }

    datasets = {}
    for name, path in paths.items():
        dataframe = _standardize_gene_names(pd.read_csv(path, low_memory=False))
        dataframe.name = name
        datasets[name] = dataframe
    return datasets


def oncogene_gene_list(
    datasets: Mapping[str, pd.DataFrame],
    *,
    dataset_names: Sequence[str] | None = None,
    mode: str = "union",
) -> list[str]:
    if mode not in {"union", "intersection"}:
        raise ValueError("mode must be 'union' or 'intersection'")
    names = list(dataset_names) if dataset_names is not None else list(datasets)
    if not names:
        return []

    gene_sets = [
        set(datasets[name].iloc[:, 0].astype(str).str.strip())
        for name in names
    ]
    genes = set.union(*gene_sets) if mode == "union" else set.intersection(*gene_sets)
    return sorted(genes)


def _coerce_dataset_spec(name: str, value: object | None = None) -> OncogeneDatasetSpec:
    if isinstance(value, OncogeneDatasetSpec):
        return value
    if isinstance(value, str):
        return OncogeneDatasetSpec(name, value)
    if isinstance(value, Mapping):
        kind = str(value.get("kind", "continuous"))
        levels = value.get("levels")
        normalize = value.get("normalize")
        level_tuple = None if levels is None else tuple(float(level) for level in levels)
        return OncogeneDatasetSpec(name, kind, level_tuple, normalize)
    if name in DEFAULT_ONCOGENE_DATASET_SPECS:
        return DEFAULT_ONCOGENE_DATASET_SPECS[name]
    return OncogeneDatasetSpec(name)


def _normalize_dataset_specs(
    datasets: Mapping[str, pd.DataFrame],
    dataset_specs: Mapping[str, OncogeneDatasetSpec | Mapping[str, object] | str] | None,
) -> dict[str, OncogeneDatasetSpec]:
    dataset_specs = dataset_specs or {}
    normalized = {}
    for name in datasets:
        normalized[name] = _coerce_dataset_spec(name, dataset_specs.get(name))
        if normalized[name].kind not in {"continuous", "binary", "discrete"}:
            raise ValueError(f"Dataset {name!r} has unsupported kind {normalized[name].kind!r}.")
    return normalized


def _coerce_pair_spec(value: OncogenePairSpec | Sequence[str]) -> OncogenePairSpec:
    if isinstance(value, OncogenePairSpec):
        return value
    if isinstance(value, str):
        separator = ":" if ":" in value else ","
        if separator not in value:
            raise ValueError("String pair specs must use 'left:right' or 'left,right'.")
        left, right = value.split(separator, maxsplit=1)
        return OncogenePairSpec(left.strip(), right.strip())
    if len(value) < 2:
        raise ValueError("Pair specs must contain at least two dataset names.")
    return OncogenePairSpec(str(value[0]), str(value[1]))


def _normalize_pair_specs(
    datasets: Mapping[str, pd.DataFrame],
    pairs: Sequence[OncogenePairSpec | Sequence[str]] | None,
) -> tuple[OncogenePairSpec, ...]:
    if pairs is None:
        if all(name in datasets for name in DEFAULT_ONCOGENE_DATASET_SPECS):
            return DEFAULT_ONCOGENE_PAIR_SPECS
        names = list(datasets)
        return tuple(
            OncogenePairSpec(left, right)
            for left_index, left in enumerate(names)
            for right in names[left_index + 1:]
        )

    normalized = tuple(_coerce_pair_spec(pair) for pair in pairs)
    missing = {
        name
        for pair in normalized
        for name in (pair.left, pair.right)
        if name not in datasets
    }
    if missing:
        raise ValueError(f"Pair specs reference missing datasets: {', '.join(sorted(missing))}")
    return normalized


def _spec_levels(
    dataframe: pd.DataFrame,
    spec: OncogeneDatasetSpec,
) -> np.ndarray:
    if spec.is_continuous:
        raise ValueError("Continuous datasets do not have fixed density levels.")
    if spec.levels is not None:
        return np.asarray(spec.levels, dtype=float)
    values = dataframe.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    levels = np.unique(values[np.isfinite(values)])
    if len(levels) == 0:
        raise ValueError(f"Could not infer discrete levels for dataset {spec.name!r}.")
    return levels.astype(float)


def _should_normalize(spec: OncogeneDatasetSpec) -> bool:
    if spec.normalize is not None:
        return spec.normalize
    return spec.is_continuous


def _binary_matrix_to_reference(
    binary: pd.DataFrame,
    reference: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference_out = reference.copy()
    reference_out.iloc[:, 0] = reference_out.iloc[:, 0].astype(str).str.strip()

    first_column = str(reference_out.columns[0]).strip()
    row_names = reference_out.iloc[:, 0].astype(str).str.strip()
    data_columns = pd.Index(reference_out.columns.astype(str).str.strip()[1:])

    binary_matrix = binary.copy()
    binary_matrix.columns = binary_matrix.columns.astype(str).str.strip()
    binary_matrix.iloc[:, 0] = binary_matrix.iloc[:, 0].astype(str).str.strip()
    binary_matrix = binary_matrix.drop_duplicates(subset=binary_matrix.columns[0]).set_index(binary_matrix.columns[0])
    binary_matrix = binary_matrix.apply(pd.to_numeric, errors="coerce")
    binary_matrix = binary_matrix.reindex(index=row_names, columns=data_columns, fill_value=0.0)
    binary_matrix = binary_matrix.fillna(0.0).gt(0).astype(float)
    binary_matrix = binary_matrix.reset_index()
    binary_matrix.columns = [first_column, *data_columns]
    return binary_matrix, reference_out


def _prepare_pair_dataframes(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_spec: OncogeneDatasetSpec,
    right_spec: OncogeneDatasetSpec,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if left_spec.kind == "binary" and right_spec.kind != "binary":
        left_out, right_out = _binary_matrix_to_reference(left, right)
    elif right_spec.kind == "binary" and left_spec.kind != "binary":
        right_out, left_out = _binary_matrix_to_reference(right, left)
    else:
        left_out, right_out = _trim_pair_dataframes(left, right)

    if _should_normalize(left_spec):
        left_out = _normalize_rows(left_out)
    if _should_normalize(right_spec):
        right_out = _normalize_rows(right_out)
    return left_out, right_out


def _safe_std(values: pd.DataFrame | np.ndarray) -> float:
    array = values.iloc[:, 1:].to_numpy(dtype=float) if isinstance(values, pd.DataFrame) else np.asarray(values, dtype=float)
    std = float(np.nanstd(array))
    if not np.isfinite(std) or std == 0:
        return 1.0
    return std


def _safe_variance(values: pd.DataFrame) -> float:
    variance = float(np.nanvar(values.iloc[:, 1:].to_numpy(dtype=float)))
    if not np.isfinite(variance) or variance == 0:
        return 1.0
    return variance


def _pair_sigma(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    left_spec: OncogeneDatasetSpec,
    right_spec: OncogeneDatasetSpec,
) -> float:
    if left_spec.is_continuous and right_spec.is_continuous:
        return float(np.sqrt((_safe_variance(left_df) + _safe_variance(right_df)) / 2))
    if left_spec.is_continuous:
        return _safe_std(left_df)
    if right_spec.is_continuous:
        return _safe_std(right_df)
    return 1.0


def _bin_continuous(values: np.ndarray, num_bins: int = 10) -> np.ndarray:
    finite_values = np.asarray(values, dtype=float)
    if len(finite_values) == 0:
        return np.array([], dtype=int)
    if np.nanstd(finite_values) == 0:
        return np.zeros(len(finite_values), dtype=int)

    quantiles = np.linspace(0, 1, num_bins + 1)
    edges = np.unique(np.quantile(finite_values, quantiles))
    if len(edges) <= 2:
        return np.zeros(len(finite_values), dtype=int)
    return np.digitize(finite_values, edges[1:-1], right=False)


def compute_mutual_information(
    x: Sequence[float],
    y: Sequence[float],
    *,
    x_discrete: bool = False,
    y_discrete: bool = False,
    random_state: int = 0,
) -> float:
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    mask = np.isfinite(x_array) & np.isfinite(y_array)
    x_array = x_array[mask]
    y_array = y_array[mask]

    if len(x_array) < 3:
        return float("nan")

    if x_discrete and y_discrete:
        return float(mutual_info_score(np.rint(x_array).astype(int), np.rint(y_array).astype(int)))
    if x_discrete:
        return float(mutual_info_score(np.rint(x_array).astype(int), _bin_continuous(y_array)))
    if y_discrete:
        return float(mutual_info_score(_bin_continuous(x_array), np.rint(y_array).astype(int)))
    return float(
        mutual_info_regression(
            x_array.reshape(-1, 1),
            y_array,
            discrete_features=False,
            random_state=random_state,
        )[0]
    )


def bicor(
    x: Sequence[float],
    y: Sequence[float],
    c: float = 9.0,
    epsilon: float = 1e-9,
) -> float:
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    mask = np.isfinite(x_array) & np.isfinite(y_array)
    x_array = x_array[mask]
    y_array = y_array[mask]

    if len(x_array) < 3:
        return float("nan")

    x_med = np.median(x_array)
    y_med = np.median(y_array)
    x_mad = np.median(np.abs(x_array - x_med))
    y_mad = np.median(np.abs(y_array - y_med))
    x_mad = max(float(x_mad), epsilon)
    y_mad = max(float(y_mad), epsilon)

    ux = (x_array - x_med) / (c * x_mad)
    uy = (y_array - y_med) / (c * y_mad)
    wx = (1 - ux**2) ** 2
    wy = (1 - uy**2) ** 2
    wx[np.abs(ux) >= 1] = 0
    wy[np.abs(uy) >= 1] = 0

    x_weighted = (x_array - x_med) * wx
    y_weighted = (y_array - y_med) * wy
    numerator = np.sum(x_weighted * y_weighted)
    denominator = np.sqrt(np.sum(x_weighted**2) * np.sum(y_weighted**2))
    denominator = max(float(denominator), epsilon)
    return float(numerator / denominator)


def build_oncogene_density_features(
    datasets: Mapping[str, pd.DataFrame],
    *,
    pairs: Sequence[OncogenePairSpec | Sequence[str]] | None = None,
    dataset_specs: Mapping[str, OncogeneDatasetSpec | Mapping[str, object] | str] | None = None,
    gene_universe: str = "union",
    gene_column_name: str = "gene name",
    boxes: int = 7,
    copy_number_levels: Sequence[float] = COPY_NUMBER_LEVELS,
    log_offset: float | None = None,
    drop_missing: bool = True,
    progress: bool = False,
) -> pd.DataFrame:
    specs = _normalize_dataset_specs(datasets, dataset_specs)
    if "copy_num" in specs and specs["copy_num"].levels == tuple(COPY_NUMBER_LEVELS):
        specs["copy_num"] = OncogeneDatasetSpec("copy_num", "discrete", tuple(copy_number_levels), normalize=False)
    pair_specs = _normalize_pair_specs(datasets, pairs)
    pair_dataset_names = sorted({name for pair in pair_specs for name in (pair.left, pair.right)})
    genes = oncogene_gene_list(datasets, dataset_names=pair_dataset_names, mode=gene_universe)
    dataset = pd.DataFrame({gene_column_name: genes})
    gene_index = {gene: index for index, gene in enumerate(genes)}

    pair_iterator = tqdm(pair_specs, desc="Gene PDM feature blocks", disable=not progress)
    for pair in pair_iterator:
        left_name = pair.left
        right_name = pair.right
        pair_label = f"{left_name} vs {right_name}"
        pair_iterator.set_postfix_str(pair_label)

        left_spec = specs[left_name]
        right_spec = specs[right_name]
        left_df, right_df = _prepare_pair_dataframes(
            datasets[left_name],
            datasets[right_name],
            left_spec,
            right_spec,
        )

        left_levels = None if left_spec.is_continuous else _spec_levels(left_df, left_spec)
        right_levels = None if right_spec.is_continuous else _spec_levels(right_df, right_spec)
        left_width = boxes if left_spec.is_continuous else len(left_levels)
        right_width = boxes if right_spec.is_continuous else len(right_levels)
        column_count = left_width * right_width

        columns = [f"{left_name}.{right_name}.{i}" for i in range(column_count)]
        temp = pd.DataFrame(np.nan, index=range(len(genes)), columns=columns)

        left_rows = _row_lookup(left_df)
        right_rows = _row_lookup(right_df)
        left_std = _safe_std(left_df)
        right_std = _safe_std(right_df)
        sigma = _pair_sigma(left_df, right_df, left_spec, right_spec)

        common_genes = [gene for gene in left_df.iloc[:, 0].astype(str).str.strip() if gene in right_rows]
        for gene_name in tqdm(common_genes, desc=pair_label, disable=not progress, leave=False):
            if gene_name not in gene_index:
                continue
            if left_spec.is_continuous:
                x, x_centers = _extract_density_values(
                    left_rows[gene_name],
                    cutoff=True,
                    std=left_std,
                    max_value=boxes,
                    boxes=boxes,
                )
            else:
                x = _extract_density_values(left_rows[gene_name], cutoff=False)
                x_centers = left_levels

            if right_spec.is_continuous:
                y, y_centers = _extract_density_values(
                    right_rows[gene_name],
                    cutoff=True,
                    std=right_std,
                    max_value=boxes,
                    boxes=boxes,
                )
            else:
                y = _extract_density_values(right_rows[gene_name], cutoff=False)
                y_centers = right_levels

            x, y = drop_nan(x, y)
            if len(x) == 0:
                continue
            matrix = densitymap(
                x,
                y,
                x_centers,
                y_centers,
                xdiscrete=left_spec.is_discrete,
                ydiscrete=right_spec.is_discrete,
                sigma=sigma,
            )
            if isinstance(matrix, str):
                continue
            temp.iloc[gene_index[gene_name]] = matrix.flatten()

        offset = log_offset if log_offset is not None else 1 / max(len(left_df.columns), 1)
        temp = temp.astype(float) + offset
        dataset = pd.concat([dataset, temp.map(np.log)], axis=1)

    if drop_missing:
        dataset = dataset.dropna()
    return dataset.reset_index(drop=True)


def build_oncogene_correlation_features(
    datasets: Mapping[str, pd.DataFrame],
    *,
    method: str = "pearson",
) -> pd.DataFrame:
    if method not in {"pearson", "spearman"}:
        raise ValueError("method must be 'pearson' or 'spearman'")
    return build_oncogene_statistic_features(datasets, method=method)


def build_oncogene_statistic_features(
    datasets: Mapping[str, pd.DataFrame],
    *,
    method: str = "pearson",
    pairs: Sequence[OncogenePairSpec | Sequence[str]] | None = None,
    dataset_specs: Mapping[str, OncogeneDatasetSpec | Mapping[str, object] | str] | None = None,
    gene_universe: str = "union",
    gene_column_name: str = "gene name",
    progress: bool = False,
) -> pd.DataFrame:
    if method not in {"pearson", "spearman", "mi", "bicor"}:
        raise ValueError("method must be 'pearson', 'spearman', 'mi', or 'bicor'")

    specs = _normalize_dataset_specs(datasets, dataset_specs)
    pair_specs = _normalize_pair_specs(datasets, pairs)
    pair_dataset_names = sorted({name for pair in pair_specs for name in (pair.left, pair.right)})
    genes = oncogene_gene_list(datasets, dataset_names=pair_dataset_names, mode=gene_universe)
    dataset = pd.DataFrame({gene_column_name: genes})
    gene_index = {gene: index for index, gene in enumerate(genes)}

    pair_iterator = tqdm(
        pair_specs,
        desc=f"Gene {method} feature blocks",
        disable=not progress,
    )
    for pair in pair_iterator:
        left_name = pair.left
        right_name = pair.right
        pair_label = f"{left_name} vs {right_name}"
        pair_iterator.set_postfix_str(pair_label)
        temp = pd.DataFrame(np.nan, index=range(len(genes)), columns=[f"{left_name}.{right_name}"])

        left_spec = specs[left_name]
        right_spec = specs[right_name]
        left_df, right_df = _prepare_pair_dataframes(
            datasets[left_name],
            datasets[right_name],
            left_spec,
            right_spec,
        )

        left_rows = _row_lookup(left_df)
        right_rows = _row_lookup(right_df)
        x_discrete = left_spec.is_discrete
        y_discrete = right_spec.is_discrete

        common_genes = [gene for gene in left_df.iloc[:, 0].astype(str).str.strip() if gene in right_rows]
        for gene_name in tqdm(common_genes, desc=pair_label, disable=not progress, leave=False):
            if gene_name not in gene_index:
                continue
            x = left_rows[gene_name]
            y = right_rows[gene_name]
            x, y = drop_nan(x, y)
            if len(x) == 0:
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
                    x_discrete=x_discrete,
                    y_discrete=y_discrete,
                )
            else:
                value = bicor(x, y)
            temp.iloc[gene_index[gene_name], 0] = value
        dataset = pd.concat([dataset, temp], axis=1)

    return dataset.reset_index(drop=True)


def save_oncogene_feature_tables(
    data_dir: str | Path,
    *,
    input_paths: Mapping[str, str | Path] | None = None,
    output_dir: str | Path | None = None,
    pairs: Sequence[OncogenePairSpec | Sequence[str]] | None = None,
    dataset_specs: Mapping[str, OncogeneDatasetSpec | Mapping[str, object] | str] | None = None,
    feature_methods: Sequence[str] = ("pearson", "spearman", "mi", "bicor"),
    density_name: str = "dataset_trimmed_v3.csv",
    pearson_name: str = "pearson.csv",
    spearman_name: str = "spearman.csv",
    mi_name: str = "mi.csv",
    bicor_name: str = "bicor.csv",
    verbose: bool = False,
    progress: bool = False,
) -> dict[str, Path]:
    root = Path(data_dir)
    output_root = Path(output_dir) if output_dir is not None else root / "Trimmed data"
    output_root.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("Loading input tables...")
    datasets = load_oncogene_inputs(root, input_paths=input_paths)
    density_path = output_root / density_name
    method_names = {
        "pearson": pearson_name,
        "spearman": spearman_name,
        "mi": mi_name,
        "bicor": bicor_name,
    }

    if verbose:
        print(f"Building PDM density features -> {density_path}")
    build_oncogene_density_features(
        datasets,
        pairs=pairs,
        dataset_specs=dataset_specs,
        progress=progress,
    ).to_csv(density_path, index=False)

    outputs = {"density": density_path}
    for method in feature_methods:
        if method not in method_names:
            raise ValueError("feature_methods may contain only 'pearson', 'spearman', 'mi', or 'bicor'")
        path = output_root / method_names[method]
        if verbose:
            print(f"Building {method} features -> {path}")
        build_oncogene_statistic_features(
            datasets,
            method=method,
            pairs=pairs,
            dataset_specs=dataset_specs,
            progress=progress,
        ).to_csv(path, index=False)
        outputs[method] = path

    return outputs


def load_oncogene_feature_sets(
    data_dir: str | Path,
    *,
    feature_dir: str | Path | None = None,
    dataset_paths: Mapping[str, str | Path] | None = None,
    density_name: str = "dataset_trimmed_v3.csv",
    pearson_name: str = "pearson.csv",
    spearman_name: str = "spearman.csv",
    mi_name: str = "mi.csv",
    bicor_name: str = "bicor.csv",
    oncogene_name: str = "oncogene.txt",
    oncogene_path: str | Path | None = None,
    label_path: str | Path | None = None,
    positive_label_extras: Sequence[str] = (),
) -> tuple[dict[str, tuple[pd.DataFrame, pd.DataFrame]], pd.DataFrame]:
    root = Path(data_dir)
    feature_root = Path(feature_dir) if feature_dir is not None else root / "Trimmed data"
    paths = (
        {name: Path(path) for name, path in dataset_paths.items()}
        if dataset_paths is not None
        else {
            "PDM": feature_root / density_name,
            "Pearson": feature_root / pearson_name,
            "Spearman": feature_root / spearman_name,
            "MutualInfo": feature_root / mi_name,
            "Bicor": feature_root / bicor_name,
        }
    )
    resolved_label_path = label_path or oncogene_path
    label_file = Path(resolved_label_path) if resolved_label_path is not None else root / oncogene_name
    positive_set = set(pd.read_csv(label_file).iloc[:, 0].astype(str).str.strip())
    positive_set.update(str(value).strip() for value in positive_label_extras if str(value).strip())

    data_dict: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    combined = pd.DataFrame()

    for name, path in paths.items():
        if not path.exists():
            continue
        dataset = pd.read_csv(path).dropna().reset_index(drop=True)
        label = dataset.iloc[:, 0].astype(str).str.strip().isin(positive_set)
        positive = dataset.loc[label].reset_index(drop=True)
        negative = dataset.loc[~label].reset_index(drop=True)
        data_dict[name] = (positive, negative)
        if combined.empty and name == "PDM":
            combined = dataset.assign(positive_label=label.to_numpy())

    return data_dict, combined


def rank_candidate_genes(
    positive: pd.DataFrame,
    negative: pd.DataFrame,
    *,
    trials: int = 100,
    model: str = "GBR",
    ks_test: bool = True,
    features_left: int | None = None,
    id_column_name: str | None = None,
    progress: bool = False,
) -> pd.DataFrame:
    from pdm_learn.modeling import core_predict

    scores = core_predict(
        positive.iloc[:, 1:].to_numpy(),
        negative.iloc[:, 1:].to_numpy(),
        trials,
        model=model,
        ks_test=ks_test,
        features_left=features_left,
        progress=progress,
        progress_desc="Gene ranking trials",
    )
    output = pd.DataFrame(
        {
            id_column_name or str(negative.columns[0]): negative.iloc[:, 0].to_numpy(),
            "score": scores,
        }
    )
    return output.sort_values("score", ascending=False).reset_index(drop=True)


def rank_candidate_oncogenes(
    positive: pd.DataFrame,
    negative: pd.DataFrame,
    *,
    trials: int = 100,
    model: str = "GBR",
    ks_test: bool = True,
    features_left: int | None = None,
    progress: bool = False,
) -> pd.DataFrame:
    return rank_candidate_genes(
        positive,
        negative,
        trials=trials,
        model=model,
        ks_test=ks_test,
        features_left=features_left,
        progress=progress,
    )


GeneDatasetSpec = OncogeneDatasetSpec
GenePairSpec = OncogenePairSpec
build_gene_density_features = build_oncogene_density_features
build_gene_correlation_features = build_oncogene_correlation_features
build_gene_statistic_features = build_oncogene_statistic_features
save_gene_feature_tables = save_oncogene_feature_tables
load_gene_feature_sets = load_oncogene_feature_sets


def evaluate_method_curves(
    data_dict: Mapping[str, tuple[pd.DataFrame, pd.DataFrame]],
    *,
    trials: int,
    model: str = "XGB",
    ks_test: bool | Mapping[str, bool] = False,
    features_left: int | Mapping[str, int | None] | None = None,
    metric: str = "pr",
) -> dict[str, tuple[float, np.ndarray, np.ndarray]]:
    results: dict[str, tuple[float, np.ndarray, np.ndarray]] = {}
    for name, (positive, negative) in data_dict.items():
        feature_limit = (
            features_left.get(name)
            if isinstance(features_left, Mapping)
            else features_left
        )
        use_ks_test = ks_test.get(name, False) if isinstance(ks_test, Mapping) else ks_test
        pos_values = positive.iloc[:, 1:].to_numpy()
        neg_values = negative.iloc[:, 1:].to_numpy()
        if metric == "pr":
            area, x_values, y_values = KFold_PR(
                pos_values,
                neg_values,
                trials,
                model=model,
                ks_test=use_ks_test,
                features_left=feature_limit,
            )
        elif metric == "loocv":
            area, x_values, y_values = LOOCV(
                pos_values,
                neg_values,
                trials,
                model=model,
                ks_test=use_ks_test,
                features_left=feature_limit,
                equation=True,
            )
        else:
            raise ValueError("metric must be 'pr' or 'loocv'")
        results[name] = (area, np.asarray(x_values), np.asarray(y_values))
    return results


def plot_method_curves(
    results: Mapping[str, tuple[float, np.ndarray, np.ndarray]],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    sns.set_theme(
        style="whitegrid",
        context="paper",
        font_scale=1.2,
        rc={
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "semibold",
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "grid.alpha": 0.2,
            "grid.linestyle": "--",
        },
    )
    palette = sns.color_palette("colorblind", n_colors=max(len(results), 1))
    style_cycle = ["solid", "dashed", "dotted", "dashdot"]
    plt.figure(figsize=(8, 5))
    for index, (name, (area, x_values, y_values)) in enumerate(results.items()):
        sns.lineplot(
            x=x_values,
            y=y_values,
            label=f"{name} ({area:.3f})",
            linewidth=2.2,
            linestyle=style_cycle[index % len(style_cycle)],
            color=palette[index % len(palette)],
        )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(title="", frameon=False)
    sns.despine()
    plt.tight_layout()
    plt.show()
