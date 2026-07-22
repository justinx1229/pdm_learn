"""Small, reproducible utilities for simulated paired observations."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator

import numpy as np
import pandas as pd

from .features import density_map


def standardize_pair(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Center and scale each vector independently when it has nonzero spread."""

    def standardize(values: np.ndarray) -> np.ndarray:
        output = np.asarray(values, dtype=float).copy()
        scale = float(np.nanstd(output))
        if scale > 0 and np.isfinite(scale):
            output = (output - np.nanmean(output)) / scale
        return output

    return standardize(x), standardize(y)


def clip_pair_to_centers(
    x: np.ndarray,
    y: np.ndarray,
    centers: Iterable[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Clip two vectors to the inclusive range spanned by density centers."""
    axis = np.asarray(tuple(centers), dtype=float)
    if axis.ndim != 1 or not len(axis):
        raise ValueError("centers must be a non-empty one-dimensional sequence.")
    return np.clip(x, axis.min(), axis.max()), np.clip(y, axis.min(), axis.max())


def perturb_pair(
    x: np.ndarray,
    y: np.ndarray,
    noise_scale: float,
    *,
    shuffle_y: bool = False,
    centers: Iterable[float] | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Add independent noise to a pair, optionally break association, and standardize."""
    if not np.isfinite(noise_scale) or noise_scale < 0:
        raise ValueError("noise_scale must be a finite non-negative number.")
    generator = rng or np.random.default_rng()
    x_values = np.asarray(x, dtype=float) + generator.normal(scale=noise_scale, size=len(x))
    y_values = np.asarray(y, dtype=float) + generator.normal(scale=noise_scale, size=len(y))
    if x_values.shape != y_values.shape:
        raise ValueError("x and y must have matching shapes.")
    if shuffle_y:
        y_values = generator.permutation(y_values)
    x_values, y_values = standardize_pair(x_values, y_values)
    return (
        clip_pair_to_centers(x_values, y_values, centers)
        if centers is not None
        else (x_values, y_values)
    )


def iter_simulated_pairs(
    seed_pairs: pd.DataFrame,
    *,
    repeats: int,
    noise_scale: float,
    shuffle_y: bool = False,
    centers: Iterable[float] | None = None,
    rng: np.random.Generator | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield perturbed pairs from consecutive rows of a seed-pair table."""
    if len(seed_pairs) % 2:
        raise ValueError("seed_pairs must contain an even number of rows.")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer.")
    generator = rng or np.random.default_rng()
    for _ in range(repeats):
        for index in range(0, len(seed_pairs), 2):
            yield perturb_pair(
                seed_pairs.iloc[index].to_numpy(dtype=float),
                seed_pairs.iloc[index + 1].to_numpy(dtype=float),
                noise_scale,
                shuffle_y=shuffle_y,
                centers=centers,
                rng=generator,
            )


def collect_simulated_pairs(
    seed_pairs: pd.DataFrame,
    *,
    repeats: int,
    noise_scale: float,
    shuffle_y: bool = False,
    centers: Iterable[float] | None = None,
    rng: np.random.Generator | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Materialize :func:`iter_simulated_pairs` for reuse across feature families."""
    return list(
        iter_simulated_pairs(
            seed_pairs,
            repeats=repeats,
            noise_scale=noise_scale,
            shuffle_y=shuffle_y,
            centers=centers,
            rng=rng,
        )
    )


def build_metric_dataset(
    seed_pairs: pd.DataFrame,
    metric: Callable[[np.ndarray, np.ndarray], float],
    *,
    repeats: int,
    noise_scale: float,
    shuffle_y: bool = False,
    centers: Iterable[float] | None = None,
    column_name: str = "value",
    rng: np.random.Generator | None = None,
    simulated_pairs: Iterable[tuple[np.ndarray, np.ndarray]] | None = None,
) -> pd.DataFrame:
    """Apply a scalar metric to simulated pairs and return a one-column table."""
    pairs = (
        simulated_pairs
        if simulated_pairs is not None
        else iter_simulated_pairs(
            seed_pairs,
            repeats=repeats,
            noise_scale=noise_scale,
            shuffle_y=shuffle_y,
            centers=centers,
            rng=rng,
        )
    )
    return pd.DataFrame({column_name: [metric(x, y) for x, y in pairs]})


def build_density_dataset(
    seed_pairs: pd.DataFrame,
    *,
    centers: Iterable[float],
    repeats: int,
    noise_scale: float,
    sigma: float,
    shuffle_y: bool = False,
    log_offset: float | None = None,
    rng: np.random.Generator | None = None,
    simulated_pairs: Iterable[tuple[np.ndarray, np.ndarray]] | None = None,
) -> pd.DataFrame:
    """Build log-density-map feature rows from simulated pairs."""
    axis = np.asarray(tuple(centers), dtype=float)
    pairs = (
        simulated_pairs
        if simulated_pairs is not None
        else iter_simulated_pairs(
            seed_pairs,
            repeats=repeats,
            noise_scale=noise_scale,
            shuffle_y=shuffle_y,
            centers=axis,
            rng=rng,
        )
    )
    rows = [density_map(x, y, axis, axis, sigma=sigma).ravel() for x, y in pairs]
    if not rows:
        return pd.DataFrame()
    output = pd.DataFrame(rows)
    offset = 1 / output.shape[1] if log_offset is None else float(log_offset)
    if not np.isfinite(offset) or offset <= 0:
        raise ValueError("log_offset must be a positive finite number.")
    return np.log(output + offset)


__all__ = [
    "build_density_dataset",
    "build_metric_dataset",
    "clip_pair_to_centers",
    "collect_simulated_pairs",
    "iter_simulated_pairs",
    "perturb_pair",
    "standardize_pair",
]
