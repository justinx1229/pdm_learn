"""Plot simulation patterns and saved benchmarks as PNG figures; no model fitting."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from functools import partial
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pdm_learn_matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from pdm_learn.simulation import build_positive_base, collect_simulated_pairs

PROJECT_ROOT = Path(__file__).resolve().parents[2]
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "0.75", "grid.color": "0.9", "grid.linewidth": 0.6,
    "axes.axisbelow": True,
})
NAMES = {
    "PDM": "PDM", "PEARSON": "Pearson", "SPEARMAN": "Spearman", "MI": "Mutual information",
    "BICOR": "Bicor", "logistic_regression": "Logistic regression",
    "random_forest": "Random forest", "gradient_boosting": "Gradient boosting", "svm": "SVM",
    "xgboost": "XGBoost",
}
PALETTE = ("#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7")
COLORS = dict(zip(NAMES, (*PALETTE, *PALETTE), strict=True))
LINESTYLES = ("-", "--", "-.", ":", (0, (3, 1, 1, 1)))


def _score_limits(values, *, start_at_zero=False):
    lower, upper = float(np.min(values)), float(np.max(values))
    span = max(upper - lower, abs(upper) * 0.1, 0.001)
    step = 10 ** (np.floor(np.log10(span)) - 1)
    lower = max(0, np.floor((lower - span * 0.15) / step) * step)
    upper = min(1.02, np.ceil((upper + span * 0.15) / step) * step)
    return (0 if start_at_zero else lower), upper


def _save(figure, output_path):
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(f"{output_path}.png", dpi=300, bbox_inches="tight")
    finally:
        plt.close(figure)


def save_patterns(epsilon, seed, output_path):
    """Show the first realization of each pattern, standardized but not clipped."""
    pairs = collect_simulated_pairs(
        build_positive_base(), repeats=1, noise_scale=epsilon, rng=np.random.default_rng(seed)
    )
    figure, axes = plt.subplots(2, 2, figsize=(7, 7), layout="constrained")
    names = ("Linear", "Quadratic", "Sinusoidal", "L-shaped")
    for axis, (x, y), name in zip(axes.flat, pairs, names, strict=True):
        axis.scatter(x, y, s=12, color="#1f77b4", alpha=0.8, linewidths=0)
        axis.set(title=name, xlabel="Standardized X", ylabel="Standardized Y")
        axis.set_box_aspect(1)
        axis.grid(False)
        axis.spines[["top", "right"]].set_visible(True)
    figure.suptitle(f"Epsilon std: {epsilon:g}", fontsize=16)
    figure.supxlabel("Before clipping to the PDM grid", fontsize=9, color="0.4")
    _save(figure, output_path)


def save_summary(table, output_path, *, group="method"):
    metrics = ("loocv_area", "average_precision")
    if table.empty or not np.isfinite(table[list(metrics)].to_numpy()).all():
        raise ValueError("Simulation summaries require nonempty, finite metrics.")
    sweep = "epsilon_std" in table and table.epsilon_std.nunique() > 1
    figure, axes = plt.subplots(1, 2, figsize=(8, 4.6 if sweep else 4), layout="constrained")
    labels = [NAMES.get(name, name) for name in table[group]]
    for axis, metric, label in zip(
        axes, metrics, ("Normalized LOOCV recovery area", "Average precision (stratified CV)"),
        strict=True,
    ):
        if sweep:
            for index, (name, rows) in enumerate(table.groupby(group, sort=False)):
                rows = rows.sort_values("epsilon_std")
                axis.plot(rows.epsilon_std, rows[metric], marker="o", linewidth=2,
                          color=COLORS.get(name), linestyle=LINESTYLES[index % len(LINESTYLES)],
                          label=NAMES.get(name, name))
            axis.set(xlabel="Noise standard deviation (ε std)", ylabel=label,
                     xticks=sorted(table.epsilon_std.unique()),
                     ylim=_score_limits(table[metric], start_at_zero=metric == "average_precision"))
            axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
            continue
        axis.scatter(np.arange(len(table)), table[metric], s=55, zorder=3,
                     c=[COLORS.get(name, "0.4") for name in table[group]])
        axis.set(xticks=np.arange(len(table)), xticklabels=labels, ylabel=label,
                 ylim=_score_limits(table[metric], start_at_zero=metric == "average_precision"))
        for index, value in enumerate(table[metric]):
            axis.annotate(f"{value:.3f}", (index, value), xytext=(0, 7),
                          textcoords="offset points", ha="center", fontsize=9)
        axis.tick_params(axis="x", labelrotation=30)
        axis.grid(axis="x", visible=False)
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
    comparison = "PDM classifiers" if group == "model" else "feature methods"
    figure.suptitle(f"Simulation · {comparison}")
    if sweep:
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="outside lower center",
                      ncol=3, frameon=False, fontsize=9)
    _save(figure, output_path)


def _draw_curve(table, axis, *, kind, group, show_metrics):
    if kind == "pr":
        table = table.loc[table.threshold.notna()]
    for index, (name, curve) in enumerate(table.groupby(group, sort=False)):
        options = {"color": COLORS.get(name), "linewidth": 2,
                   "linestyle": LINESTYLES[index % len(LINESTYLES)]}
        label = NAMES.get(name, name)
        if kind == "pr":
            if show_metrics:
                label += f" (AP = {curve.average_precision.iloc[0]:.3f})"
            axis.step(curve.recall, curve.precision, where="post", **options,
                      label=label)
        else:
            if show_metrics:
                label += f" (area = {curve.loocv_area.iloc[0]:.3f})"
            curve = curve.sort_values("rank_threshold")
            axis.step(curve.rank_threshold, curve.recovered_positives, where="post",
                      **options, label=label)
    if kind == "pr":
        prevalence = table.prevalence.iloc[0]
        if not np.allclose(table.prevalence, prevalence):
            raise ValueError("PR comparison requires the same class prevalence for every curve.")
        axis.axhline(prevalence, color="0.6", linestyle="--", linewidth=1,
                     label=f"Positive prevalence ({prevalence:.3g})")
        axis.set(xlabel="Recall", ylabel="Precision",
                 xlim=(0, 1), ylim=_score_limits(table.precision, start_at_zero=True))
    else:
        axis.set(xlabel="Rank threshold (zero-based; lower is better)",
                 ylabel="Held-out positives recovered",
                 xlim=(0, table.rank_threshold.max()), ylim=(0, table.positive_rows.max() * 1.02))
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.yaxis.set_major_locator(MaxNLocator(integer=True))


def save_curve(table, output_path, *, kind, group="method"):
    levels = (list(table.groupby("epsilon_std", sort=True))
              if "epsilon_std" in table else [(None, table)])
    count = len(levels)
    columns = min(3, int(np.ceil(np.sqrt(count))))
    rows = (count + columns - 1) // columns
    size = (6.4, 4.8) if count == 1 else (4.4 * columns, 3.8 * rows + 0.6)
    figure, axes = plt.subplots(rows, columns, squeeze=False, layout="constrained", figsize=size)
    for axis, (epsilon, subset) in zip(axes.flat, levels, strict=False):
        _draw_curve(subset, axis, kind=kind, group=group, show_metrics=count == 1)
        if epsilon is not None:
            axis.set_title(f"ε std = {epsilon:g}")
    for axis in list(axes.flat)[count:]:
        axis.set_axis_off()
    comparison = "PDM classifiers" if group == "model" else "feature methods"
    title = "Precision–recall (stratified CV)" if kind == "pr" else "LOOCV rank recovery"
    figure.suptitle(f"{title} · simulation {comparison}")
    note = "Conventional (recall=0, precision=1) endpoint omitted."
    if count == 1:
        axes[0, 0].legend(frameon=False, fontsize=9, loc="best")
    else:
        handles, labels = axes[0, 0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="outside lower center",
                      ncol=3, frameon=False, fontsize=9,
                      title=note if kind == "pr" else None, title_fontsize=8)
    if kind == "pr" and count == 1:
        figure.supxlabel(note, fontsize=8, color="0.4")
    _save(figure, output_path)


# Each plot reads one saved CSV; classifier plots reuse the same two functions.
PLOTS = {
    "summary": ("benchmarks", save_summary),
    "loocv": ("loocv_curve", partial(save_curve, kind="loocv")),
    "pr": ("pr_curve", partial(save_curve, kind="pr")),
    "models": ("model_benchmarks", partial(save_summary, group="model")),
    "model-loocv": ("model_loocv_curve", partial(save_curve, kind="loocv", group="model")),
    "model-pr": ("model_pr_curve", partial(save_curve, kind="pr", group="model")),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path,
                        default=PROJECT_ROOT / "runs" / "simulation" / "results")
    parser.add_argument("--output-dir", type=Path, help="Defaults to RESULTS_DIR/figures.")
    parser.add_argument("--feature-dir", type=Path,
                        help="Manifest directory; defaults to RESULTS_DIR/../features.")
    parser.add_argument("--epsilon-std", type=float, nargs="+",
                        help="Noise levels for pattern figures; defaults to the manifest levels.")
    parser.add_argument("--seed", type=int,
                        help="Pattern seed; defaults to the manifest seed, or 42.")
    parser.add_argument("--plots", nargs="+", choices=("all", "patterns", *PLOTS), default=["all"])
    args = parser.parse_args()
    if "all" in args.plots and len(args.plots) > 1:
        parser.error("Use 'all' alone or select individual plots.")
    if args.epsilon_std is not None and any(
        not np.isfinite(value) or value < 0 for value in args.epsilon_std
    ):
        parser.error("epsilon-std values must be finite and non-negative.")
    output_dir = args.output_dir or args.results_dir / "figures"
    selected = ("patterns", *PLOTS) if args.plots == ["all"] else args.plots
    written = 0
    for name in selected:
        if name == "patterns":
            feature_dir = args.feature_dir or args.results_dir.parent / "features"
            manifest_path = feature_dir / "simulation_manifest.json"
            manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
            epsilons = args.epsilon_std or manifest.get("epsilon_stds")
            if not epsilons:
                message = f"missing {manifest_path}; supply --feature-dir or --epsilon-std"
                if args.plots != ["all"]:
                    raise ValueError(message)
                print(f"Skipping patterns: {message}")
                continue
            seed = args.seed if args.seed is not None else manifest.get("seed", 42)
            for epsilon in sorted(set(epsilons)):
                output = output_dir / f"simulation_patterns_epsilon_{epsilon:g}"
                save_patterns(epsilon, seed, output)
                print(f"patterns: {output}.png")
                written += 1
            continue
        suffix, plot = PLOTS[name]
        source = args.results_dir / f"simulation_{suffix}.csv"
        if not source.is_file():
            if args.plots != ["all"]:
                raise FileNotFoundError(source)
            print(f"Skipping {name}: missing {source}")
            continue
        # Old classifier results can remain after rerunning only the method comparison.
        # Skip their curves too, rather than mixing old and current evaluations.
        model_plot = name.startswith("model")
        summary = args.results_dir / (
            "simulation_model_benchmarks.csv" if model_plot else "simulation_benchmarks.csv"
        )
        if summary.is_file():
            columns = pd.read_csv(summary, nrows=0).columns
            if "roc_auc" in columns and "loocv_area" not in columns:
                instruction = "Rerun scripts/simulation/run_benchmarks.py"
                if model_plot:
                    instruction += " with --models and your chosen classifiers"
                message = (
                    f"{summary} uses obsolete ROC results; loocv_area is missing. {instruction}."
                )
                if args.plots != ["all"]:
                    raise ValueError(message)
                print(f"Skipping {name}: {message}")
                continue
        output = output_dir / source.stem.removesuffix("_curve")
        plot(pd.read_csv(source), output)
        print(f"{name}: {output}.png")
        written += 1
    if not written:
        raise FileNotFoundError(f"No compatible plot inputs found in {args.results_dir}.")


if __name__ == "__main__":
    main()
