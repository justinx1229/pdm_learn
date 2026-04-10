from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mutual_info_score

from pdm_learn.modeling import KFold_PR, LOOCV
from pdm_learn.preprocessing import densitymap, drop_nan


COPY_NUMBER_LEVELS = np.array([0, 1, 2, 3, 4, 6, 8], dtype=float)
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


def _standardize_gene_names(dataframe: pd.DataFrame) -> pd.DataFrame:
    output = dataframe.copy()
    output.iloc[:, 0] = output.iloc[:, 0].astype(str).str.strip()
    return output


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
    output.iloc[:, 1:] = numeric.sub(numeric.mean(axis=1), axis=0)
    return output


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

    mutation_pairs = mutation.iloc[:, [0, -1]].copy()
    mutation_pairs.columns = [first_column, "cell_line"]
    mutation_pairs[first_column] = mutation_pairs[first_column].astype(str).str.strip()
    mutation_pairs["cell_line"] = mutation_pairs["cell_line"].astype(str).str.strip()
    mutation_pairs = mutation_pairs[
        mutation_pairs[first_column].isin(row_names) & mutation_pairs["cell_line"].isin(data_columns)
    ]

    mutation_matrix = (
        pd.crosstab(mutation_pairs[first_column], mutation_pairs["cell_line"]).gt(0).astype(float)
    )
    mutation_matrix = mutation_matrix.reindex(index=row_names, columns=data_columns, fill_value=0.0)
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


def load_oncogene_inputs(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    root = Path(data_dir)

    gene_exp = _standardize_gene_names(
        pd.read_csv(root / "DepMap_data" / "CCLE_gene_expression_trimmed_Wei.csv")
    )
    gene_exp.name = "gene_exp"

    copy_num = snap_copy_number_levels(
        pd.read_csv(root / "DepMap_data" / "CCLE_gene_cn_trimmed_Wei.csv")
    )
    copy_num.name = "copy_num"

    shrna = _standardize_gene_names(
        pd.read_csv(root / "DepMap_data" / "shRNA_Broad_Trimmed_Wei.csv")
    )
    shrna.name = "shRNA"

    gene_mut = pd.read_csv(
        root / "DepMap_data" / "CCLE_gene_mutation_trimmed_Wei.csv",
        usecols=["Hugo_Symbol", "Cell line"],
        low_memory=False,
    )
    gene_mut.iloc[:, 0] = gene_mut.iloc[:, 0].astype(str).str.strip()
    gene_mut.name = "gene_mut"

    crispr = _standardize_gene_names(
        pd.read_csv(root / "DepMap_data" / "Avana_gene_effect_20Q3_Trimmed_Wei.csv")
    )
    crispr.name = "CRISPR"

    return {
        "gene_exp": gene_exp,
        "copy_num": copy_num,
        "shRNA": shrna,
        "gene_mut": gene_mut,
        "CRISPR": crispr,
    }


def oncogene_gene_list(datasets: Mapping[str, pd.DataFrame]) -> list[str]:
    return sorted(
        set(datasets["gene_exp"].iloc[:, 0])
        | set(datasets["copy_num"].iloc[:, 0])
        | set(datasets["shRNA"].iloc[:, 0])
        | set(datasets["CRISPR"].iloc[:, 0])
    )

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


def bicor(x: Sequence[float], y: Sequence[float], c: float = 9.0) -> float:
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
    if x_mad == 0 or y_mad == 0:
        return float("nan")

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
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def build_oncogene_density_features(
    datasets: Mapping[str, pd.DataFrame],
    *,
    boxes: int = 7,
    copy_number_levels: Sequence[float] = COPY_NUMBER_LEVELS,
    log_offset: float | None = None,
) -> pd.DataFrame:
    genes = oncogene_gene_list(datasets)
    dataset = pd.DataFrame({"gene name": genes})
    gene_index = {gene: index for index, gene in enumerate(genes)}

    discrete_levels = np.asarray(copy_number_levels, dtype=float)

    for left_name, right_name, flags in ONCOGENE_PAIRS:
        is_mut = "mut" in flags
        is_cn = "cn" in flags

        if is_mut and is_cn:
            column_count = 2 * len(discrete_levels)
        elif is_mut:
            column_count = 2 * boxes
        else:
            column_count = boxes * boxes

        columns = [f"{left_name}.{right_name}.{i}" for i in range(column_count)]
        temp = pd.DataFrame(np.nan, index=range(len(genes)), columns=columns)

        if is_mut:
            df1_t, df2_t = _mutation_to_reference(datasets[left_name], datasets[right_name])
        else:
            df1_t, df2_t = _trim_pair_dataframes(datasets[left_name], datasets[right_name])

        left_rows = _row_lookup(df1_t)
        right_rows = _row_lookup(df2_t)

        if is_mut and is_cn:
            for gene_name in df2_t.iloc[:, 0]:
                x = _extract_density_values(left_rows[gene_name], cutoff=False)
                y = _extract_density_values(right_rows[gene_name], cutoff=False)
                x, y = drop_nan(x, y)
                matrix = densitymap(
                    x,
                    y,
                    [0.0, 1.0],
                    discrete_levels,
                    xdiscrete=True,
                    ydiscrete=True,
                )
                temp.iloc[gene_index[gene_name]] = matrix.flatten()
        elif is_mut:
            df2_t = _normalize_rows(df2_t)
            sigma = float(np.nanstd(df2_t.iloc[:, 1:].to_numpy(dtype=float)))
            if not np.isfinite(sigma) or sigma == 0:
                sigma = 1.0
            for gene_name in df2_t.iloc[:, 0]:
                x = _extract_density_values(left_rows[gene_name], cutoff=False)
                y, y_centers = _extract_density_values(
                    right_rows[gene_name],
                    cutoff=True,
                    std=sigma,
                    max_value=boxes,
                    boxes=boxes,
                )
                x, y = drop_nan(x, y)
                matrix = densitymap(
                    x,
                    y,
                    [0.0, 1.0],
                    y_centers,
                    xdiscrete=True,
                    sigma=sigma,
                )
                temp.iloc[gene_index[gene_name]] = matrix.flatten()
        elif is_cn:
            df1_t = _normalize_rows(df1_t)
            sigma = float(np.nanstd(df1_t.iloc[:, 1:].to_numpy(dtype=float)))
            if not np.isfinite(sigma) or sigma == 0:
                sigma = 1.0
            for gene_name in df1_t.iloc[:, 0]:
                x, x_centers = _extract_density_values(
                    left_rows[gene_name],
                    cutoff=True,
                    std=sigma,
                    max_value=boxes,
                    boxes=boxes,
                )
                y = _extract_density_values(right_rows[gene_name], cutoff=False)
                x, y = drop_nan(x, y)
                matrix = densitymap(
                    x,
                    y,
                    x_centers,
                    discrete_levels,
                    ydiscrete=True,
                    sigma=sigma,
                )
                temp.iloc[gene_index[gene_name]] = matrix.flatten()
        else:
            df1_t = _normalize_rows(df1_t)
            df2_t = _normalize_rows(df2_t)
            x_std = float(np.nanstd(df1_t.iloc[:, 1:].to_numpy(dtype=float)))
            y_std = float(np.nanstd(df2_t.iloc[:, 1:].to_numpy(dtype=float)))
            sigma = float(
                np.sqrt(
                    (
                        np.nanvar(df1_t.iloc[:, 1:].to_numpy(dtype=float))
                        + np.nanvar(df2_t.iloc[:, 1:].to_numpy(dtype=float))
                    )
                    / 2
                )
            )
            if not np.isfinite(x_std) or x_std == 0:
                x_std = 1.0
            if not np.isfinite(y_std) or y_std == 0:
                y_std = 1.0
            if not np.isfinite(sigma) or sigma == 0:
                sigma = 1.0
            for gene_name in df1_t.iloc[:, 0]:
                x, x_centers = _extract_density_values(
                    left_rows[gene_name],
                    cutoff=True,
                    std=x_std,
                    max_value=boxes,
                    boxes=boxes,
                )
                y, y_centers = _extract_density_values(
                    right_rows[gene_name],
                    cutoff=True,
                    std=y_std,
                    max_value=boxes,
                    boxes=boxes,
                )
                x, y = drop_nan(x, y)
                matrix = densitymap(x, y, x_centers, y_centers, sigma=sigma)
                temp.iloc[gene_index[gene_name]] = matrix.flatten()

        offset = log_offset if log_offset is not None else 1 / len(df1_t.columns)
        temp = temp.astype(float) + offset
        dataset = pd.concat([dataset, temp.map(np.log)], axis=1)

    return dataset.dropna().reset_index(drop=True)


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
) -> pd.DataFrame:
    if method not in {"pearson", "spearman", "mi", "bicor"}:
        raise ValueError("method must be 'pearson', 'spearman', 'mi', or 'bicor'")

    genes = oncogene_gene_list(datasets)
    dataset = pd.DataFrame({"gene name": genes})
    gene_index = {gene: index for index, gene in enumerate(genes)}

    for left_name, right_name, _ in ONCOGENE_PAIRS:
        temp = pd.DataFrame(np.nan, index=range(len(genes)), columns=[f"{left_name}.{right_name}"])

        if left_name == "gene_mut":
            left_df, right_df = _mutation_to_reference(datasets[left_name], datasets[right_name])
        else:
            left_df, right_df = _trim_pair_dataframes(datasets[left_name], datasets[right_name])

        left_rows = _row_lookup(left_df)
        right_rows = _row_lookup(right_df)
        x_discrete = left_name == "gene_mut"
        y_discrete = right_name == "copy_num"

        for gene_name in left_df.iloc[:, 0]:
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
    density_name: str = "dataset_trimmed_v3.csv",
    pearson_name: str = "pearson.csv",
    spearman_name: str = "spearman.csv",
    mi_name: str = "mi.csv",
    bicor_name: str = "bicor.csv",
) -> dict[str, Path]:
    root = Path(data_dir)
    output_dir = root / "Trimmed data"
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = load_oncogene_inputs(root)
    density_path = output_dir / density_name
    pearson_path = output_dir / pearson_name
    spearman_path = output_dir / spearman_name
    mi_path = output_dir / mi_name
    bicor_path = output_dir / bicor_name

    build_oncogene_density_features(datasets).to_csv(density_path, index=False)
    build_oncogene_statistic_features(datasets, method="pearson").to_csv(pearson_path, index=False)
    build_oncogene_statistic_features(datasets, method="spearman").to_csv(spearman_path, index=False)
    build_oncogene_statistic_features(datasets, method="mi").to_csv(mi_path, index=False)
    build_oncogene_statistic_features(datasets, method="bicor").to_csv(bicor_path, index=False)

    return {
        "density": density_path,
        "pearson": pearson_path,
        "spearman": spearman_path,
        "mi": mi_path,
        "bicor": bicor_path,
    }


def load_oncogene_feature_sets(
    data_dir: str | Path,
    *,
    density_name: str = "dataset_trimmed_v3.csv",
    pearson_name: str = "pearson.csv",
    spearman_name: str = "spearman.csv",
    mi_name: str = "mi.csv",
    bicor_name: str = "bicor.csv",
    oncogene_name: str = "oncogene.txt",
) -> tuple[dict[str, tuple[pd.DataFrame, pd.DataFrame]], pd.DataFrame]:
    root = Path(data_dir)
    dataset_paths = {
        "PDM": root / "Trimmed data" / density_name,
        "Pearson": root / "Trimmed data" / pearson_name,
        "Spearman": root / "Trimmed data" / spearman_name,
        "MutualInfo": root / "Trimmed data" / mi_name,
        "Bicor": root / "Trimmed data" / bicor_name,
    }
    oncogene_set = set(pd.read_csv(root / oncogene_name).iloc[:, 0].astype(str).str.strip())
    oncogene_set.add("MYCL")

    data_dict: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    combined = pd.DataFrame()

    for name, path in dataset_paths.items():
        if not path.exists():
            continue
        dataset = pd.read_csv(path).dropna().reset_index(drop=True)
        label = dataset.iloc[:, 0].astype(str).str.strip().isin(oncogene_set)
        positive = dataset.loc[label].reset_index(drop=True)
        negative = dataset.loc[~label].reset_index(drop=True)
        data_dict[name] = (positive, negative)
        if combined.empty and name == "PDM":
            combined = dataset.assign(oncogene=label.to_numpy())

    return data_dict, combined


def rank_candidate_oncogenes(
    positive: pd.DataFrame,
    negative: pd.DataFrame,
    *,
    trials: int = 100,
    model: str = "GBR",
    ks_test: bool = True,
    features_left: int | None = None,
) -> pd.DataFrame:
    from pdm_learn.modeling import core_predict

    scores = core_predict(
        positive.iloc[:, 1:].to_numpy(),
        negative.iloc[:, 1:].to_numpy(),
        trials,
        model=model,
        ks_test=ks_test,
        features_left=features_left,
    )
    output = pd.DataFrame(
        {
            "gene name": negative.iloc[:, 0].to_numpy(),
            "score": scores,
        }
    )
    return output.sort_values("score", ascending=False).reset_index(drop=True)


def evaluate_method_curves(
    data_dict: Mapping[str, tuple[pd.DataFrame, pd.DataFrame]],
    *,
    trials: int,
    model: str = "XGB",
    ks_test: bool = False,
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
        pos_values = positive.iloc[:, 1:].to_numpy()
        neg_values = negative.iloc[:, 1:].to_numpy()
        if metric == "pr":
            area, x_values, y_values = KFold_PR(
                pos_values,
                neg_values,
                trials,
                model=model,
                ks_test=ks_test,
                features_left=feature_limit,
            )
        elif metric == "loocv":
            area, x_values, y_values = LOOCV(
                pos_values,
                neg_values,
                trials,
                model=model,
                ks_test=ks_test,
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
    style_cycle = ["-", "--", ":", "-."]
    plt.figure(figsize=(8, 5))
    for index, (name, (area, x_values, y_values)) in enumerate(results.items()):
        linestyle = style_cycle[index % len(style_cycle)]
        plt.plot(x_values, y_values, label=f"{name}: ({area:.3f})", linestyle=linestyle)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.show()
