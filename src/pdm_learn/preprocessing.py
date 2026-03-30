from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd


def trim(dataframe: pd.DataFrame, genes: Sequence[str]) -> pd.DataFrame:
    def process(df: pd.DataFrame) -> pd.DataFrame:
        output = df.copy()
        output.sort_index(inplace=True)
        first_column = output.iloc[:, 0].astype(str).str.strip()
        output.drop(columns=output.columns[0], inplace=True)
        output.sort_index(axis=1, inplace=True)
        output.insert(0, first_column.name, first_column)
        return output

    output = process(dataframe)
    mask = output.iloc[:, 0].isin(genes)
    return output[mask]


def mut_trim(mutation: pd.DataFrame, genes: Sequence[str]) -> pd.DataFrame:
    output = pd.DataFrame(
        columns=list(sorted(set(mutation.iloc[:, -1]))),
        index=range(len(genes) + 1),
    )
    output.insert(0, "gene name", list(genes) + ["throwaway"])
    output.insert(len(output.columns), "throwaway", np.zeros(len(genes) + 1))
    output.iloc[:, 1:] = 0

    row_index = {key: index for index, key in enumerate(output.iloc[:, 0])}
    column_index = {key: index for index, key in enumerate(output.columns)}
    for i in range(len(mutation)):
        output.iloc[
            row_index.get(mutation.iloc[i, 0], -1),
            column_index.get(mutation.iloc[i, -1], -1),
        ] = 1

    return output.iloc[:-1, :-1]


def normalize(matrix: np.ndarray) -> np.ndarray:
    output = np.copy(matrix)
    mean = np.nanmean(matrix, axis=1)[:, np.newaxis]
    return output - mean


def density_centers(df: pd.DataFrame, num: int) -> np.ndarray:
    std = np.nanstd(df.iloc[:, 1:])
    return np.linspace(-std * num, std * num, num=num * 2, endpoint=False)[1::2]


def extract(df: pd.DataFrame, name: str) -> np.ndarray | int:
    output = df[df.iloc[:, 0] == name].iloc[:, 1:].to_numpy()
    if len(output) == 0:
        return -1
    return output[0]


def drop_nan(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    combined = np.array(np.concatenate(([x], [y]), axis=0))
    combined = combined[:, ~pd.isna(combined).any(axis=0)]
    return combined[0], combined[1]


def densitymap(
    x: Sequence[float],
    y: Sequence[float],
    xDensityCenters: Sequence[float],
    yDensityCenters: Sequence[float],
    xdiscrete: bool = False,
    ydiscrete: bool = False,
    sigma: float = 1,
) -> np.ndarray | str:
    if len(x) != len(y):
        return "inconsistent size of x and y vectors"

    sigma_sq_inv = (1 / sigma) ** 2
    matrix = np.zeros((len(yDensityCenters), len(xDensityCenters)))

    x_lookup = {value: index for index, value in enumerate(xDensityCenters)}
    y_lookup = {value: index for index, value in enumerate(yDensityCenters)}

    if not xdiscrete and not ydiscrete:
        for pt in range(len(x)):
            temp = np.zeros((len(yDensityCenters), len(xDensityCenters)))
            for i, center_x in enumerate(xDensityCenters):
                for j, center_y in enumerate(yDensityCenters):
                    dist_sq = (x[pt] - center_x) ** 2 + (y[pt] - center_y) ** 2
                    temp[j, i] = np.exp(-0.5 * sigma_sq_inv * dist_sq)

            temp /= np.sum(temp)
            matrix += temp

    elif xdiscrete and ydiscrete:
        for i, center_x in enumerate(xDensityCenters):
            for j, center_y in enumerate(yDensityCenters):
                matrix[j, i] += np.sum(x[np.asarray(y) == center_y] == center_x)

    elif xdiscrete:
        for pt in range(len(x)):
            temp = np.zeros(len(yDensityCenters))
            for i, center_y in enumerate(yDensityCenters):
                dist_sq = (y[pt] - center_y) ** 2
                temp[i] = np.exp(-0.5 * sigma_sq_inv * dist_sq)

            temp /= np.sum(temp)
            matrix[:, x_lookup[x[pt]]] += temp
    else:
        for pt in range(len(y)):
            temp = np.zeros(len(xDensityCenters))
            for i, center_x in enumerate(xDensityCenters):
                dist_sq = (x[pt] - center_x) ** 2
                temp[i] = np.exp(-0.5 * sigma_sq_inv * dist_sq)

            temp /= np.sum(temp)
            matrix[y_lookup[y[pt]]] += temp

    matrix /= len(x)
    return matrix


def _is_boolean_sequence(values: object) -> bool:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return False
    return all(isinstance(value, (bool, np.bool_)) for value in values)


def _resolve_density_map_arguments(
    arg3: Sequence[object],
    arg4: Sequence[object],
) -> tuple[Sequence[Sequence[float]], Sequence[bool]]:
    if _is_boolean_sequence(arg3) and not _is_boolean_sequence(arg4):
        return arg4, arg3
    return arg3, arg4


def build_density_map(
    datasets: Sequence[pd.DataFrame],
    pairs: Sequence[Sequence[str]],
    arg3: Sequence[object],
    arg4: Sequence[object],
) -> pd.DataFrame:
    density_points, continuous = _resolve_density_map_arguments(arg3, arg4)
    output = pd.DataFrame({"pair": [f"{p1}.{p2}" for p1, p2 in pairs]})

    for i in range(len(datasets)):
        for j in range(len(datasets)):
            df1 = datasets[i]
            df2 = datasets[j]

            mask = df1.columns.str.strip().isin(df2.columns.str.strip())
            mask[0] = True
            df1 = df1.loc[:, mask]

            mask = df2.columns.str.strip().isin(df1.columns.str.strip())
            mask[0] = True
            df2 = df2.loc[:, mask]

            df1_pts = density_points[i]
            df2_pts = density_points[j]
            df1_cont = continuous[i]
            df2_cont = continuous[j]

            feature_prefix_1 = getattr(datasets[i], "name", f"dataset_{i}")
            feature_prefix_2 = getattr(datasets[j], "name", f"dataset_{j}")
            temp = pd.DataFrame(
                index=range(len(output)),
                columns=[
                    f"{feature_prefix_1}.{feature_prefix_2}.{value}"
                    for value in range(len(df1_pts) * len(df2_pts))
                ],
            )

            if df1_cont:
                if df2_cont:
                    std = math.sqrt(
                        (
                            np.nanstd(df1.iloc[:, 1:].to_numpy()) ** 2
                            + np.nanstd(df2.iloc[:, 1:].to_numpy()) ** 2
                        )
                        / 2
                    )
                else:
                    std = np.nanstd(df1.iloc[:, 1:].to_numpy())
            else:
                std = np.nanstd(df2.iloc[:, 1:].to_numpy())

            for index, (p1, p2) in enumerate(pairs):
                x = extract(df1, p1)
                y = extract(df2, p2)
                if isinstance(x, int) or isinstance(y, int):
                    continue
                x, y = drop_nan(x, y)
                matrix = densitymap(
                    x,
                    y,
                    df1_pts,
                    df2_pts,
                    xdiscrete=not df1_cont,
                    ydiscrete=not df2_cont,
                    sigma=std,
                )
                temp.iloc[index] = matrix.flatten()

            temp = temp.astype(float)
            temp += 1 / len(df1.columns)
            temp = temp.map(np.log)
            output = pd.concat([output, temp], axis=1)

    return output


def trim_pairs(pairs: Sequence[Sequence[str]], lim: Sequence[str]) -> np.ndarray:
    output = np.array([[None, None]])
    allowed = set(lim)
    for x, y in pairs:
        if x in allowed and y in allowed:
            output = np.append(output, [[x, y]], axis=0)

    return np.delete(output, 0, 0)
