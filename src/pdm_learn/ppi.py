from __future__ import annotations

import ast
from collections.abc import Iterable, Sequence
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

from pdm_learn.modeling import _build_predictor, _predict_scores
from pdm_learn.preprocessing import build_density_map, density_centers


PPI_DATASET_ORDER = ("gene_exp", "copy_num", "shRNA", "gene_mut", "CRISPR")
PPI_METADATA_COLUMNS = {"Gene1", "Gene2", "Interaction_Type", "Confidence_Score", "pair"}
GENES_COLUMN = "Representative Genes (Core Members)"
COMPLEX_NAME_COLUMN = "Complex Name"
SHARED_TRIMMED_FILENAMES = {
    "expression": "Gene_Expression_Trimmed.csv",
    "copy_number": "Copy_Number_Trimmed.csv",
    "shrna": "shRNA_Trimmed.csv",
    "mutation": "Gene_Mutation_Trimmed.csv",
    "crispr": "CRISPR_Trimmed.csv",
}
SHARED_SOURCE_FILENAMES = {
    "expression": "CCLE_gene_expression_trimmed_Wei.csv",
    "copy_number": "CCLE_gene_cn_trimmed_Wei.csv",
    "shrna": "shRNA_Broad_Trimmed_Wei.csv",
    "mutation": "CCLE_gene_mutation_trimmed_Wei.csv",
    "crispr": "Avana_gene_effect_20Q3_Trimmed_Wei.csv",
}


def canonical_pair(gene_a: object, gene_b: object) -> str:
    left, right = sorted((str(gene_a).strip(), str(gene_b).strip()))
    return f"{left}.{right}"


def parse_gene_list(value: object) -> list[str]:
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            value = value.split(",")
    if not isinstance(value, (list, tuple, set)):
        value = [value]
    return [str(gene).strip() for gene in value if str(gene).strip()]


def _standardize_gene_index(dataframe: pd.DataFrame) -> pd.DataFrame:
    output = dataframe.copy()
    output.columns = [str(column).strip() for column in output.columns]
    output.iloc[:, 0] = output.iloc[:, 0].astype(str).str.strip()
    return output.drop_duplicates(subset=output.columns[0]).set_index(output.columns[0])


def _filter_to_shared_genes(dataframe: pd.DataFrame, shared_genes: Sequence[str]) -> pd.DataFrame:
    indexed = _standardize_gene_index(dataframe)
    output = indexed.loc[list(shared_genes)].reset_index()
    output = output.rename(columns={output.columns[0]: dataframe.columns[0]})
    return output.loc[:, [output.columns[0], *sorted(output.columns[1:])]]


def _row_center_matrix(dataframe: pd.DataFrame) -> pd.DataFrame:
    output = dataframe.copy()
    numeric = output.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    output.iloc[:, 1:] = numeric.sub(numeric.mean(axis=1), axis=0)
    return output


def _snap_copy_number_matrix(
    dataframe: pd.DataFrame,
    *,
    levels: Sequence[float] = (0, 1, 2, 3, 4, 6, 8),
) -> pd.DataFrame:
    output = dataframe.copy()
    level_array = np.asarray(levels, dtype=float)
    numeric = output.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float) * 2
    distances = np.abs(numeric[..., np.newaxis] - level_array)
    distances[np.isnan(distances)] = np.inf
    indices = np.argmin(distances, axis=-1)
    all_missing = np.isinf(distances).all(axis=-1)
    output.iloc[:, 1:] = level_array[indices]
    output.iloc[:, 1:] = output.iloc[:, 1:].mask(all_missing)
    return output


def _mutation_matrix_from_long_table(
    mutation: pd.DataFrame,
    *,
    shared_genes: Sequence[str],
    gene_column: str = "Hugo_Symbol",
    cell_line_column: str = "Cell line",
) -> pd.DataFrame:
    if gene_column not in mutation.columns:
        raise ValueError(f"Mutation gene column {gene_column!r} was not found.")
    if cell_line_column not in mutation.columns:
        raise ValueError(f"Mutation cell-line column {cell_line_column!r} was not found.")

    shared_gene_list = list(shared_genes)
    shared_gene_set = set(shared_gene_list)
    pairs = mutation[[gene_column, cell_line_column]].copy()
    pairs[gene_column] = pairs[gene_column].astype(str).str.strip()
    pairs[cell_line_column] = pairs[cell_line_column].astype(str).str.strip()
    pairs = pairs[pairs[gene_column].isin(shared_gene_set) & pairs[cell_line_column].ne("")]

    cell_lines = sorted(pairs[cell_line_column].unique())
    matrix = pd.crosstab(pairs[gene_column], pairs[cell_line_column]).gt(0).astype(float)
    matrix = matrix.reindex(index=shared_gene_list, columns=cell_lines, fill_value=0.0)
    matrix = matrix.reset_index()
    matrix.columns = ["gene name", *cell_lines]
    return matrix


def derive_shared_trimmed_inputs(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    copy_number_levels: Sequence[float] = (0, 1, 2, 3, 4, 6, 8),
) -> dict[str, Path]:
    """Build shared ``DepMap_Trimmed`` tables from ``trimmed_Wei`` inputs.

    The shared inputs are derived by taking the gene intersection across
    expression, copy-number, shRNA, and CRISPR, row-centering continuous assays,
    discretizing copy number after doubling, and converting the long mutation
    table to a binary gene-by-cell-line matrix.
    """
    source_root = Path(source_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    expression = pd.read_csv(source_root / SHARED_SOURCE_FILENAMES["expression"])
    copy_number = pd.read_csv(source_root / SHARED_SOURCE_FILENAMES["copy_number"])
    shrna = pd.read_csv(source_root / SHARED_SOURCE_FILENAMES["shrna"])
    crispr = pd.read_csv(source_root / SHARED_SOURCE_FILENAMES["crispr"])
    mutation = pd.read_csv(source_root / SHARED_SOURCE_FILENAMES["mutation"], low_memory=False)

    gene_sets = [
        set(_standardize_gene_index(dataframe).index)
        for dataframe in (expression, copy_number, shrna, crispr)
    ]
    shared_genes = sorted(set.intersection(*gene_sets))

    outputs = {
        "expression": output_root / SHARED_TRIMMED_FILENAMES["expression"],
        "copy_number": output_root / SHARED_TRIMMED_FILENAMES["copy_number"],
        "shrna": output_root / SHARED_TRIMMED_FILENAMES["shrna"],
        "mutation": output_root / SHARED_TRIMMED_FILENAMES["mutation"],
        "crispr": output_root / SHARED_TRIMMED_FILENAMES["crispr"],
    }

    _row_center_matrix(_filter_to_shared_genes(expression, shared_genes)).to_csv(outputs["expression"], index=False)
    _snap_copy_number_matrix(
        _filter_to_shared_genes(copy_number, shared_genes),
        levels=copy_number_levels,
    ).to_csv(outputs["copy_number"], index=False)
    _row_center_matrix(_filter_to_shared_genes(shrna, shared_genes)).to_csv(outputs["shrna"], index=False)
    _mutation_matrix_from_long_table(mutation, shared_genes=shared_genes).to_csv(outputs["mutation"], index=False)
    _row_center_matrix(_filter_to_shared_genes(crispr, shared_genes)).to_csv(outputs["crispr"], index=False)
    return outputs


def load_ppi_inputs(
    data_dir: str | Path,
    *,
    input_paths: dict[str, str | Path] | None = None,
) -> dict[str, pd.DataFrame]:
    root = Path(data_dir)
    input_paths = input_paths or {}
    paths = {
        "gene_exp": Path(input_paths.get("expression", root / "DepMap_Trimmed" / "Gene_Expression_Trimmed.csv")),
        "copy_num": Path(input_paths.get("copy_number", root / "DepMap_Trimmed" / "Copy_Number_Trimmed.csv")),
        "shRNA": Path(input_paths.get("shrna", root / "DepMap_Trimmed" / "shRNA_Trimmed.csv")),
        "gene_mut": Path(input_paths.get("mutation", root / "DepMap_Trimmed" / "Gene_Mutation_Trimmed.csv")),
        "CRISPR": Path(input_paths.get("crispr", root / "DepMap_Trimmed" / "CRISPR_Trimmed.csv")),
    }
    datasets = {}
    for name, path in paths.items():
        dataframe = pd.read_csv(path)
        dataframe.name = name
        dataframe.iloc[:, 0] = dataframe.iloc[:, 0].astype(str).str.strip()
        datasets[name] = dataframe
    return datasets


def ppi_gene_universe(datasets: dict[str, pd.DataFrame]) -> list[str]:
    shared = (
        set(datasets["gene_exp"].iloc[:, 0].astype(str).str.strip())
        & set(datasets["copy_num"].iloc[:, 0].astype(str).str.strip())
        & set(datasets["shRNA"].iloc[:, 0].astype(str).str.strip())
        & set(datasets["CRISPR"].iloc[:, 0].astype(str).str.strip())
    )
    return sorted(shared)


def ppi_density_parameters(datasets: dict[str, pd.DataFrame]):
    ordered = [datasets[name] for name in PPI_DATASET_ORDER]
    density_points = (
        density_centers(datasets["gene_exp"], 7),
        [0, 1, 2, 3, 4, 6, 8],
        density_centers(datasets["shRNA"], 7),
        [0, 1],
        density_centers(datasets["CRISPR"], 7),
    )
    continuous = (True, False, True, False, True)
    return ordered, density_points, continuous


def load_biogrid_pairs(path: str | Path) -> pd.DataFrame:
    pairs = pd.read_csv(path).rename(columns={"Gene_A": "Gene1", "Gene_B": "Gene2"})
    if not {"Gene1", "Gene2"}.issubset(pairs.columns):
        raise ValueError("Biogrid input must contain Gene_A/Gene_B or Gene1/Gene2 columns.")
    pairs["Gene1"] = pairs["Gene1"].astype(str).str.strip()
    pairs["Gene2"] = pairs["Gene2"].astype(str).str.strip()
    pairs = pairs.dropna(subset=["Gene1", "Gene2"]).reset_index(drop=True)
    pairs["pair"] = [f"{left}.{right}" for left, right in zip(pairs["Gene1"], pairs["Gene2"])]
    return pairs


def build_ppi_feature_table(
    datasets: dict[str, pd.DataFrame],
    pair_table: pd.DataFrame,
    *,
    gene1_column: str = "Gene1",
    gene2_column: str = "Gene2",
    progress: bool = False,
    progress_desc: str = "PPI PDM feature blocks",
) -> pd.DataFrame:
    genes = set(ppi_gene_universe(datasets))
    pairs = pair_table.copy()
    pairs[gene1_column] = pairs[gene1_column].astype(str).str.strip()
    pairs[gene2_column] = pairs[gene2_column].astype(str).str.strip()
    pairs = pairs[pairs[gene1_column].isin(genes) & pairs[gene2_column].isin(genes)].reset_index(drop=True)
    if "pair" not in pairs.columns:
        pairs["pair"] = [f"{left}.{right}" for left, right in zip(pairs[gene1_column], pairs[gene2_column])]
    if pairs.empty:
        raise ValueError("No PPI pairs remain after filtering to the shared DepMap gene universe.")

    ordered, density_points, continuous = ppi_density_parameters(datasets)
    features = build_density_map(
        ordered,
        pairs[[gene1_column, gene2_column]].to_numpy().tolist(),
        density_points,
        continuous,
        progress=progress,
        progress_desc=progress_desc,
    )
    return pd.concat([pairs.reset_index(drop=True), features.drop(columns=["pair"])], axis=1)


def load_cancer_complexes(path: str | Path) -> pd.DataFrame:
    complexes = pd.read_excel(path)
    if "Unnamed: 0" in complexes.columns:
        complexes = complexes.drop(columns=["Unnamed: 0"])
    if GENES_COLUMN not in complexes.columns:
        raise ValueError(f"Expected column {GENES_COLUMN!r} in cancer-complex workbook.")
    if COMPLEX_NAME_COLUMN not in complexes.columns:
        complexes[COMPLEX_NAME_COLUMN] = [f"Complex {index + 1}" for index in range(len(complexes))]
    return complexes


def cancer_complex_pair_set(
    cancer_complexes: pd.DataFrame,
    *,
    genes_column: str = GENES_COLUMN,
) -> set[str]:
    pairs = set()
    for genes in cancer_complexes[genes_column].map(parse_gene_list):
        for gene_a, gene_b in combinations(dict.fromkeys(genes), 2):
            pairs.add(canonical_pair(gene_a, gene_b))
    return pairs


def sample_positive_control_pairs(
    cancer_complexes: pd.DataFrame,
    *,
    gene_universe: Iterable[str],
    max_pairs_per_complex: int = 5,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    generator = rng or np.random.default_rng()
    allowed = set(gene_universe)
    rows = []
    for _, row in cancer_complexes.iterrows():
        genes = [gene for gene in dict.fromkeys(parse_gene_list(row[GENES_COLUMN])) if gene in allowed]
        generator.shuffle(genes)
        pair_count = min(max_pairs_per_complex, len(genes) // 2)
        for index in range(pair_count):
            gene_a = genes[2 * index]
            gene_b = genes[2 * index + 1]
            rows.append(
                {
                    "Complex": str(row[COMPLEX_NAME_COLUMN]).strip(),
                    "Gene1": gene_a,
                    "Gene2": gene_b,
                    "pair": f"{gene_a}.{gene_b}",
                }
            )
    if not rows:
        raise ValueError("No positive control pairs could be sampled from the complex workbook.")
    return pd.DataFrame(rows)


def sample_negative_control_pairs(
    cancer_complexes: pd.DataFrame,
    *,
    gene_universe: Sequence[str],
    count: int,
    rng: np.random.Generator | None = None,
    max_attempts: int | None = None,
) -> pd.DataFrame:
    generator = rng or np.random.default_rng()
    genes = np.asarray(sorted(set(gene_universe)), dtype=object)
    excluded = cancer_complex_pair_set(cancer_complexes)
    max_attempts = max_attempts or count * 100
    rows = []
    seen = set()
    attempts = 0
    while len(rows) < count and attempts < max_attempts:
        attempts += 1
        gene_a, gene_b = generator.choice(genes, size=2, replace=False)
        pair_key = canonical_pair(gene_a, gene_b)
        if pair_key in excluded or pair_key in seen:
            continue
        seen.add(pair_key)
        rows.append({"Gene1": gene_a, "Gene2": gene_b, "pair": f"{gene_a}.{gene_b}"})
    if len(rows) < count:
        raise ValueError(f"Only sampled {len(rows)} negative pairs after {attempts} attempts.")
    return pd.DataFrame(rows)


def build_control_feature_tables(
    datasets: dict[str, pd.DataFrame],
    cancer_complexes: pd.DataFrame,
    *,
    controls: int = 10,
    max_pairs_per_complex: int = 5,
    negative_pairs: int = 5000,
    random_state: int = 42,
    progress: bool = False,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame], list[pd.DataFrame], list[pd.DataFrame]]:
    gene_universe = ppi_gene_universe(datasets)
    positive_pairs = []
    negative_pairs_out = []
    positive_features = []
    negative_features = []

    for index in tqdm(range(controls), desc="PPI control sets", disable=not progress):
        rng = np.random.default_rng(random_state + index)
        pos_pairs = sample_positive_control_pairs(
            cancer_complexes,
            gene_universe=gene_universe,
            max_pairs_per_complex=max_pairs_per_complex,
            rng=rng,
        )
        neg_pairs = sample_negative_control_pairs(
            cancer_complexes,
            gene_universe=gene_universe,
            count=negative_pairs,
            rng=np.random.default_rng(random_state + 10_000 + index),
        )
        positive_pairs.append(pos_pairs)
        negative_pairs_out.append(neg_pairs)
        positive_features.append(
            build_ppi_feature_table(
                datasets,
                pos_pairs,
                progress=progress,
                progress_desc=f"Control {index + 1} positive PDM blocks",
            )
        )
        negative_features.append(
            build_ppi_feature_table(
                datasets,
                neg_pairs,
                progress=progress,
                progress_desc=f"Control {index + 1} negative PDM blocks",
            )
        )

    return positive_pairs, negative_pairs_out, positive_features, negative_features


def _feature_columns(dataframe: pd.DataFrame) -> list[str]:
    return [
        column
        for column in dataframe.columns
        if column not in PPI_METADATA_COLUMNS | {"Complex", "Y"}
    ]


def evaluate_ppi_controls(
    positive_controls: Sequence[pd.DataFrame],
    negative_controls: Sequence[pd.DataFrame],
    *,
    model: str = "XGB",
    features_left: int | None = 50,
    heldout_complexes: int = 5,
    random_state: int = 42,
    progress: bool = False,
) -> pd.DataFrame:
    rows = []
    control_iterator = tqdm(
        zip(positive_controls, negative_controls),
        total=min(len(positive_controls), len(negative_controls)),
        desc="PPI control benchmarks",
        disable=not progress,
    )
    for index, (positive, negative) in enumerate(control_iterator, start=1):
        pos = positive.dropna().copy()
        neg = negative.dropna().copy()
        if pos.empty or neg.empty:
            rows.append({"control": index, "roc_auc": np.nan, "positive_test_rows": 0, "negative_test_rows": 0})
            continue

        complexes = sorted(pos["Complex"].dropna().astype(str).unique())
        rng = np.random.default_rng(random_state + index)
        selected = rng.choice(
            complexes,
            size=min(heldout_complexes, len(complexes)),
            replace=False,
        )
        train_pos = pos[~pos["Complex"].isin(selected)]
        test_pos = pos[pos["Complex"].isin(selected)]
        feature_columns = _feature_columns(pos)

        neg = neg.sample(frac=1, random_state=random_state + index).reset_index(drop=True)
        train_neg = neg.iloc[: len(train_pos)]
        test_neg = neg.iloc[len(train_pos) : len(train_pos) + len(test_pos)]
        if train_pos.empty or test_pos.empty or train_neg.empty or test_neg.empty:
            rows.append({"control": index, "roc_auc": np.nan, "positive_test_rows": len(test_pos), "negative_test_rows": len(test_neg)})
            continue

        X_train_pos = train_pos[feature_columns].to_numpy(dtype=float)
        X_train_neg = train_neg[feature_columns].to_numpy(dtype=float)
        X_test = pd.concat([test_pos[feature_columns], test_neg[feature_columns]], axis=0).to_numpy(dtype=float)
        y_test = np.concatenate([np.ones(len(test_pos)), np.zeros(len(test_neg))])

        if features_left is not None:
            keep = max(1, min(features_left, X_train_pos.shape[1]))
            p_values = [
                stats.ks_2samp(X_train_pos[:, feature_index], X_train_neg[:, feature_index], method="asymp").pvalue
                for feature_index in range(X_train_pos.shape[1])
            ]
            selected_features = np.sort(np.argpartition(p_values, keep - 1)[:keep])
        else:
            selected_features = np.arange(X_train_pos.shape[1])

        X_train = np.vstack([X_train_pos[:, selected_features], X_train_neg[:, selected_features]])
        y_train = np.concatenate([np.ones(len(train_pos)), np.zeros(len(train_neg))])
        predictor = _build_predictor(model).fit(X_train, y_train)
        scores = _predict_scores(predictor, X_test[:, selected_features])
        rows.append(
            {
                "control": index,
                "roc_auc": roc_auc_score(y_test, scores),
                "positive_test_rows": len(test_pos),
                "negative_test_rows": len(test_neg),
                "features_used": len(selected_features),
                "heldout_complexes": ";".join(selected),
            }
        )
    return pd.DataFrame(rows)


def reverse_feature_source_columns(feature_columns: Sequence[str]) -> list[str]:
    blocks: dict[str, dict[int, str]] = {}
    for column in feature_columns:
        prefix, index = column.rsplit(".", 1)
        blocks.setdefault(prefix, {})[int(index)] = column

    dataset_sizes = {}
    for prefix, columns in blocks.items():
        dataset_a, dataset_b = prefix.split(".", 1)
        if dataset_a == dataset_b:
            dataset_sizes[dataset_a] = int(np.sqrt(len(columns)))

    source_columns = []
    for column in feature_columns:
        prefix, index = column.rsplit(".", 1)
        dataset_a, dataset_b = prefix.split(".", 1)
        rows = dataset_sizes[dataset_a]
        columns = dataset_sizes[dataset_b]
        row, col = divmod(int(index), columns)
        source_prefix = f"{dataset_b}.{dataset_a}"
        source_columns.append(blocks[source_prefix][col * rows + row])
    return source_columns


def reverse_pair_table(data: pd.DataFrame, feature_columns: Sequence[str]) -> pd.DataFrame:
    reversed_data = data.copy()
    reversed_data.loc[:, feature_columns] = data[reverse_feature_source_columns(feature_columns)].to_numpy()
    reversed_data["Gene1"] = data["Gene2"].to_numpy()
    reversed_data["Gene2"] = data["Gene1"].to_numpy()
    reversed_data["pair"] = [
        f"{gene_a}.{gene_b}"
        for gene_a, gene_b in zip(reversed_data["Gene1"], reversed_data["Gene2"])
    ]
    return reversed_data


def _choose_oriented_rows(
    original_values: np.ndarray,
    reversed_values: np.ndarray,
    row_indices: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    use_reversed = rng.integers(0, 2, size=len(row_indices)).astype(bool)
    values = original_values[row_indices].copy()
    values[use_reversed] = reversed_values[row_indices][use_reversed]
    return values


def _ks_feature_indices(
    positive_values: np.ndarray,
    negative_values: np.ndarray,
    features_left: int | None,
) -> np.ndarray:
    if features_left is None:
        features_left = len(positive_values)
    features_left = max(1, min(features_left, positive_values.shape[1]))
    p_values = [
        stats.ks_2samp(positive_values[:, index], negative_values[:, index], method="asymp").pvalue
        for index in range(positive_values.shape[1])
    ]
    return np.sort(np.argpartition(p_values, features_left - 1)[:features_left])


def sampled_cancer_complex_pair_set(
    cancer_complexes: pd.DataFrame,
    *,
    max_pairs_per_complex: int = 5,
    available_pairs: set[str] | None = None,
    rng: np.random.Generator | None = None,
    genes_column: str = GENES_COLUMN,
) -> set[str]:
    generator = np.random.default_rng() if rng is None else rng
    sampled_pairs = set()

    for genes in cancer_complexes[genes_column].map(parse_gene_list):
        possible_pairs = list(combinations(dict.fromkeys(genes), 2))
        if available_pairs is not None:
            possible_pairs = [
                pair
                for pair in possible_pairs
                if canonical_pair(pair[0], pair[1]) in available_pairs
            ]
        generator.shuffle(possible_pairs)

        used_genes = set()
        selected_count = 0
        for gene_a, gene_b in possible_pairs:
            if gene_a in used_genes or gene_b in used_genes:
                continue
            sampled_pairs.add(canonical_pair(gene_a, gene_b))
            used_genes.update((gene_a, gene_b))
            selected_count += 1
            if selected_count == max_pairs_per_complex:
                break

    return sampled_pairs


def rank_ppi_pairs_cancer(
    biogrid_feature_table: pd.DataFrame,
    cancer_complexes: pd.DataFrame,
    *,
    trials: int = 100,
    model: str = "GBR",
    ks_test: bool = True,
    features_left: int | None = None,
    max_pairs_per_complex: int = 5,
    random_state: int = 1,
    progress: bool = False,
) -> pd.DataFrame:
    data = biogrid_feature_table.copy()
    data["canonical_pair"] = [canonical_pair(a, b) for a, b in zip(data["Gene1"], data["Gene2"])]
    data["cancer_complex_pair"] = data["canonical_pair"].isin(cancer_complex_pair_set(cancer_complexes))

    feature_columns = [
        column
        for column in data.columns
        if column not in PPI_METADATA_COLUMNS | {"canonical_pair", "cancer_complex_pair"}
    ]
    model_data = data.dropna(subset=feature_columns).reset_index(drop=True)
    reversed_data = reverse_pair_table(model_data, feature_columns)
    original_values = model_data[feature_columns].to_numpy(dtype=float)
    reversed_values = reversed_data[feature_columns].to_numpy(dtype=float)
    available_pairs = set(model_data["canonical_pair"])
    scores = np.zeros(len(model_data))
    score_iterations = np.zeros(len(model_data))
    rng = np.random.default_rng(random_state)

    for _ in tqdm(range(trials), desc="PPI ranking trials", disable=not progress):
        sampled_pairs = sampled_cancer_complex_pair_set(
            cancer_complexes,
            max_pairs_per_complex=max_pairs_per_complex,
            available_pairs=available_pairs,
            rng=rng,
        )
        positive_indices = model_data.index[model_data["canonical_pair"].isin(sampled_pairs)].to_numpy()
        candidate_indices = model_data.index[~model_data["cancer_complex_pair"]].to_numpy()
        if len(positive_indices) == 0:
            raise ValueError("No sampled cancer-complex pairs were found in the Biogrid feature table.")
        negative_indices = rng.choice(candidate_indices, size=len(positive_indices), replace=False)
        test_indices = model_data.index.to_numpy()

        positive_values = _choose_oriented_rows(original_values, reversed_values, positive_indices, rng)
        negative_values = _choose_oriented_rows(original_values, reversed_values, negative_indices, rng)
        selected_features = (
            _ks_feature_indices(positive_values, negative_values, features_left)
            if ks_test
            else np.arange(len(feature_columns))
        )

        X = np.concatenate([positive_values[:, selected_features], negative_values[:, selected_features]])
        y = np.concatenate([np.ones(len(positive_values)), np.zeros(len(negative_values))])
        predictor = _build_predictor(model).fit(X, y)

        forward_scores = _predict_scores(predictor, original_values[test_indices][:, selected_features])
        reverse_scores = _predict_scores(predictor, reversed_values[test_indices][:, selected_features])
        scores[test_indices] += np.maximum(forward_scores, reverse_scores)
        score_iterations[test_indices] += 1

    average_scores = np.divide(
        scores,
        score_iterations,
        out=np.full_like(scores, np.nan),
        where=score_iterations > 0,
    )
    output_columns = ["Gene1", "Gene2", "pair", "canonical_pair", "cancer_complex_pair"]
    output = model_data[output_columns].assign(
        score=average_scores,
        score_iterations=score_iterations.astype(int),
    )
    return output.sort_values("score", ascending=False).reset_index(drop=True)
