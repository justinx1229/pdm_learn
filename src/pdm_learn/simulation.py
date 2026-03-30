from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd
from scipy import integrate as scipy_int
from sympy import diff, sqrt, utilities


def eps(size: int, std: float = 1) -> np.ndarray:
    return np.array([np.random.normal(scale=std) for _ in range(size)])


def standardize_pair(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_out = np.asarray(x, dtype=float).copy()
    y_out = np.asarray(y, dtype=float).copy()
    x_std = np.std(x_out)
    y_std = np.std(y_out)
    if x_std != 0:
        x_out /= x_std
    if y_std != 0:
        y_out /= y_std
    return x_out, y_out


def clip_pair_to_centers(
    x: np.ndarray,
    y: np.ndarray,
    centers: Iterable[float],
) -> tuple[np.ndarray, np.ndarray]:
    centers_array = np.asarray(list(centers), dtype=float)
    lower = float(np.min(centers_array))
    upper = float(np.max(centers_array))
    return np.clip(x, lower, upper), np.clip(y, lower, upper)


def perturb_pair(
    x: np.ndarray,
    y: np.ndarray,
    epsilon_std: float,
    *,
    shuffle_y: bool = False,
    centers: Iterable[float] | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    generator = rng or np.random.default_rng()
    x_out = np.asarray(x, dtype=float).copy()
    y_out = np.asarray(y, dtype=float).copy()

    x_out += generator.normal(scale=epsilon_std, size=len(x_out))
    y_out += generator.normal(scale=epsilon_std, size=len(y_out))

    if shuffle_y:
        y_out = generator.permutation(y_out)

    x_out, y_out = standardize_pair(x_out, y_out)
    if centers is not None:
        x_out, y_out = clip_pair_to_centers(x_out, y_out, centers)
    return x_out, y_out


def iter_simulated_pairs(
    positive: pd.DataFrame,
    *,
    repeats: int,
    epsilon_std: float,
    shuffle_y: bool = False,
    centers: Iterable[float] | None = None,
    rng: np.random.Generator | None = None,
):
    generator = rng or np.random.default_rng()
    pair_count = len(positive) // 2
    for _ in range(repeats):
        for index in range(pair_count):
            x = positive.iloc[2 * index].to_numpy()
            y = positive.iloc[2 * index + 1].to_numpy()
            yield perturb_pair(
                x,
                y,
                epsilon_std,
                shuffle_y=shuffle_y,
                centers=centers,
                rng=generator,
            )


def build_metric_dataset(
    positive: pd.DataFrame,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    *,
    repeats: int,
    epsilon_std: float,
    shuffle_y: bool = False,
    column_name: str | None = None,
    centers: Iterable[float] | None = None,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    values = [
        metric_fn(x, y)
        for x, y in iter_simulated_pairs(
            positive,
            repeats=repeats,
            epsilon_std=epsilon_std,
            shuffle_y=shuffle_y,
            centers=centers,
            rng=rng,
        )
    ]
    if column_name is None:
        return pd.DataFrame(values)
    return pd.DataFrame(values, columns=[column_name])


def build_heatmap_dataset(
    positive: pd.DataFrame,
    densitymap_fn: Callable[..., np.ndarray],
    *,
    centers: Iterable[float],
    repeats: int,
    epsilon_std: float,
    sigma: float,
    shuffle_y: bool = False,
    log_offset: float | None = None,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    centers_array = np.asarray(list(centers), dtype=float)
    rows = [
        densitymap_fn(x, y, centers_array, centers_array, sigma=sigma).flatten()
        for x, y in iter_simulated_pairs(
            positive,
            repeats=repeats,
            epsilon_std=epsilon_std,
            shuffle_y=shuffle_y,
            centers=centers_array,
            rng=rng,
        )
    ]
    output = pd.DataFrame(rows)
    offset = log_offset if log_offset is not None else 1 / len(output.columns)
    output += offset
    return output.map(np.log)


def partition(
    u,
    eq,
    bounds: tuple[float, float] = (-1, 1),
    num: int = -1,
    spacing: float = 0.1,
    iterations: int = 10,
    max_err: float = 0,
    endpoint: bool = True,
) -> np.ndarray:
    curr_x = bounds[0]
    out = np.array([curr_x])
    breakout = False

    if num != -1:
        t_func = utilities.lambdify(u, sqrt(1 + (diff(eq, u)) ** 2))
        spacing = scipy_int.quad(t_func, bounds[0], bounds[1])[0] / (num - 1)
        delta = spacing
        for _ in range(num - 1):
            curr_y = eq.subs(u, curr_x).evalf()
            for _ in range(iterations):
                next_y = eq.subs(u, curr_x + delta).evalf()
                if not next_y.is_real:
                    breakout = True
                    break
                dist = sqrt(delta**2 + (curr_y - next_y) ** 2)
                if abs(dist - spacing) < max_err:
                    break
                delta *= spacing / dist
                if curr_x + delta > bounds[1]:
                    breakout = True
                    break

            if breakout:
                break
            curr_x += delta
            out = np.append(out, curr_x)

        if len(out) < num:
            curr_x = bounds[1]
            temp = np.array([curr_x])
            delta *= -1
            while len(out) + len(temp) < num:
                curr_y = eq.subs(u, curr_x).evalf()
                for _ in range(iterations):
                    next_y = eq.subs(u, curr_x + delta).evalf()
                    dist = sqrt(delta**2 + (curr_y - next_y) ** 2)
                    if abs(dist - spacing) < max_err:
                        break
                    delta *= spacing / dist
                curr_x += delta
                temp = np.append(curr_x, temp)
            out = np.append(out, temp)

    else:
        delta = spacing
        while curr_x + delta <= bounds[1]:
            curr_y = eq.subs(u, curr_x).evalf()
            for _ in range(iterations):
                next_y = eq.subs(u, curr_x + delta).evalf()
                dist = sqrt(delta**2 + (curr_y - next_y) ** 2)
                if abs(dist - spacing) < max_err:
                    break
                delta *= spacing / dist
                if curr_x + delta > bounds[1]:
                    breakout = True
                    break
            if breakout:
                if endpoint:
                    out = np.append(out, bounds[1])
                break
            curr_x += delta
            out = np.append(out, curr_x)

    return out
