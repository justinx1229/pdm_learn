"""Plot saved oncogene KS statistics and benchmark results; no model fitting."""

from __future__ import annotations

import argparse
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
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Matplotlib includes this seaborn theme, so no extra plotting dependency is needed.
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "0.75", "axes.linewidth": 0.7,
    "grid.color": "0.9", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
COLORS = {"PDM": "#0072B2", "Pearson": "#E69F00", "Spearman": "#009E73",
          "MutualInfo": "#D55E00", "Bicor": "#CC79A7"}
MODEL_NAMES = {"logistic_regression": "Logistic regression", "random_forest": "Random forest",
               "gradient_boosting": "Gradient boosting", "svm": "SVM"}
LINESTYLES = ("-", "--", "-.", ":", (0, (3, 1, 1, 1)))


def _score_limits(values, *, start_at_zero=False):
    """Pad the observed score range, rounded outward to readable limits."""
    lower, upper = float(np.min(values)), float(np.max(values))
    span = max(upper - lower, abs(upper) * 0.1, 0.001)
    step = 10 ** (np.floor(np.log10(span)) - 1)
    lower = max(0, np.floor((lower - span * 0.15) / step) * step)
    upper = min(1.02, np.ceil((upper + span * 0.15) / step) * step)
    return (0 if start_at_zero else lower), upper


def _save_plot(figure, output_path: Path, pdf_output_path: Path | None = None) -> None:
    """Save paper figures as PNG and optional PDF, then release the figure."""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=300, bbox_inches="tight")
        if pdf_output_path is not None:
            pdf_output_path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(pdf_output_path, bbox_inches="tight")
    finally:
        plt.close(figure)


PAIR_ORDER = (
    ("gene_exp", "copy_num"),
    ("gene_exp", "shRNA"),
    ("gene_mut", "gene_exp"),
    ("gene_exp", "CRISPR"),
    ("shRNA", "copy_num"),
    ("gene_mut", "copy_num"),
    ("CRISPR", "copy_num"),
    ("gene_mut", "shRNA"),
    ("shRNA", "CRISPR"),
    ("gene_mut", "CRISPR"),
)
AXIS_WIDTHS = {"gene_exp": 7, "copy_num": 7, "shRNA": 7, "gene_mut": 2, "CRISPR": 7}
ROW_ORDER = ("gene_mut", "CRISPR", "shRNA", "copy_num", "gene_exp")
COLUMN_ORDER = ("gene_exp", "copy_num", "shRNA", "CRISPR", "gene_mut")
DISPLAY_NAMES = {
    "gene_exp": "Gene\nExpression",
    "copy_num": "Copy\nNumber",
    "shRNA": "shRNA",
    "gene_mut": "Gene\nMutation",
    "CRISPR": "CRISPR",
}


def _feature_values(table: pd.DataFrame, value_column: str) -> dict[str, np.ndarray]:
    required = {"feature", value_column}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"KS table is missing required columns: {', '.join(sorted(missing))}.")

    values: dict[str, np.ndarray] = {}
    for left, right in PAIR_ORDER:
        prefix = f"{left}.{right}."
        block = table.loc[table["feature"].astype(str).str.startswith(prefix)].copy()
        expected = AXIS_WIDTHS[left] * AXIS_WIDTHS[right]
        if len(block) != expected:
            raise ValueError(
                f"Expected {expected} KS values for {left} vs {right}, found {len(block)}."
            )
        indices = pd.to_numeric(
            block["feature"].astype(str).str.removeprefix(prefix), errors="raise"
        ).astype(int)
        if sorted(indices.tolist()) != list(range(expected)):
            raise ValueError(
                f"KS feature indices for {left} vs {right} must run from 0 to {expected - 1}."
            )
        ordered = block.assign(_index=indices).sort_values("_index")
        matrix = (
            ordered[value_column]
            .to_numpy(dtype=float)
            .reshape(AXIS_WIDTHS[right], AXIS_WIDTHS[left])
        )
        row, column = sorted((left, right), key=ROW_ORDER.index)
        if row == left:
            matrix = matrix.T
        values[f"{row}.{column}"] = matrix
    return values


def save_ks_heatmap(
    table: pd.DataFrame,
    output_path: Path,
    *,
    value_column: str = "log10_ks_pvalue",
    vmin: float | None = None,
    vmax: float = 0.0,
    cmap: str = "gist_heat_r",
    title: str = "Oncogene PDM feature differences",
    pdf_output_path: Path | None = None,
) -> None:
    """Write a shared-scale PDM KS block heatmap as PNG and optional PDF."""
    blocks = _feature_values(table, value_column)
    finite_values = np.concatenate([matrix[np.isfinite(matrix)] for matrix in blocks.values()])
    if not len(finite_values):
        raise ValueError("KS table does not contain finite values to plot.")
    lower = float(np.floor(finite_values.min())) if vmin is None else vmin
    if lower >= vmax:
        lower = float(finite_values.min())
    norm = Normalize(vmin=lower, vmax=vmax, clip=True)

    # The diagonal is unused: omit its empty final row and column.
    rows, columns = ROW_ORDER[:-1], COLUMN_ORDER[:-1]
    figure = plt.figure(figsize=(7.2, 6))
    grid = figure.add_gridspec(
        len(rows),
        len(columns),
        width_ratios=[AXIS_WIDTHS[name] for name in columns],
        height_ratios=[AXIS_WIDTHS[name] for name in rows],
        left=0.19,
        right=0.83,
        bottom=0.06,
        top=0.79,
        wspace=0.07,
        hspace=0.07,
    )
    axes = [[figure.add_subplot(grid[row, column]) for column in range(4)] for row in range(4)]
    image = None
    for row, left in enumerate(rows):
        for column, right in enumerate(columns):
            axis = axes[row][column]
            matrix = blocks.get(f"{left}.{right}")
            if matrix is None:
                axis.set_axis_off()
                continue
            image = axis.imshow(np.flipud(matrix), cmap=cmap, norm=norm, aspect="auto")
            axis.grid(False)
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)

    figure.canvas.draw()
    for row, name in enumerate(rows):
        position = axes[row][0].get_position()
        figure.text(
            0.17,
            (position.y0 + position.y1) / 2,
            DISPLAY_NAMES[name],
            ha="right",
            va="center",
            fontsize=10,
        )
    for column, name in enumerate(columns):
        position = axes[0][column].get_position()
        figure.text(
            (position.x0 + position.x1) / 2,
            0.81,
            DISPLAY_NAMES[name],
            ha="center",
            va="bottom",
            fontsize=10,
        )

    if image is None:
        raise RuntimeError("No KS blocks were plotted.")
    colorbar_axis = figure.add_axes([0.87, 0.08, 0.025, 0.65])
    colorbar = figure.colorbar(image, cax=colorbar_axis)
    colorbar.set_label(r"$\log_{10}$ KS p-value")
    figure.suptitle(title, y=0.95)
    _save_plot(figure, output_path, pdf_output_path)


def save_loocv_plot(
    curves: pd.DataFrame,
    output_path: Path,
    *,
    pdf_output_path: Path | None = None,
    group: str = "method",
    recovery_scale: str = "count",
) -> None:
    """Plot saved rank-recovery curves; their areas are not ROC AUC."""
    figure, axis = plt.subplots(figsize=(7.2, 4.8), layout="constrained")
    column = "recovered_positives" if recovery_scale == "count" else "recovered_fraction"
    for index, (method, curve) in enumerate(curves.groupby(group, sort=False)):
        curve = curve.sort_values("rank_threshold")
        area = float(curve["loocv_area"].iloc[0])
        axis.step(
            curve["rank_threshold"],
            curve[column],
            where="post",
            linewidth=2,
            color=COLORS.get(method, list(COLORS.values())[index % len(COLORS)]),
            linestyle=LINESTYLES[index % len(LINESTYLES)],
            label=f"{MODEL_NAMES.get(method, method)} (area = {area:.3f})",
        )
    axis.set_xlabel("Rank threshold (zero-based; lower is better)")
    axis.set_ylabel("Held-out oncogenes recovered" if recovery_scale == "count"
                    else "Fraction of held-out oncogenes recovered")
    axis.set_title("Classifier comparison" if group == "model" else "Feature-method comparison")
    maximum = curves.positive_rows.max() if recovery_scale == "count" else 1
    axis.set_ylim(0, maximum * 1.03)
    axis.set_xlim(0, curves.rank_threshold.max())
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    if recovery_scale == "count":
        axis.yaxis.set_major_locator(MaxNLocator(integer=True))
    axis.legend(title="LOOCV rank recovery", frameon=False, loc="lower right", fontsize=9)
    _save_plot(figure, output_path, pdf_output_path)


def save_labeled_plot(
    table: pd.DataFrame,
    output_path: Path,
    *,
    pdf_output_path: Path | None = None,
) -> None:
    """Plot saved ROC AUC and AP summaries, without inventing fold error bars."""
    metrics = ("roc_auc", "average_precision")
    values = table[list(metrics)].to_numpy(dtype=float)
    if table.empty or not np.isfinite(values).all():
        raise ValueError("Labeled benchmarks require nonempty, finite metric values.")
    figure, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
    for axis, metric, label in zip(axes, metrics, ("ROC AUC", "Average precision"), strict=True):
        axis.scatter(np.arange(len(table)), table[metric], s=55,
                     c=[COLORS.get(method, "0.4") for method in table.method], zorder=3)
        axis.set(
            xticks=np.arange(len(table)), xticklabels=table["method"], ylabel=label,
            ylim=_score_limits(table[metric], start_at_zero=metric == "average_precision"),
        )
        for index, value in enumerate(table[metric]):
            axis.annotate(f"{value:.3f}", (index, value), xytext=(0, 7),
                          textcoords="offset points", ha="center", fontsize=9)
        axis.grid(axis="x", visible=False)
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
        axis.tick_params(axis="x", labelrotation=30)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(_comparison_title(table))
    _save_plot(figure, output_path, pdf_output_path)


def _comparison_title(table):
    if "comparison" in table and (table.comparison == "unlabeled_candidates").all():
        return "Known oncogenes vs. unlabeled genes"
    return "Oncogenes vs. explicit negative controls"


def save_classification_curve(table, output_path, *, kind="pr", pdf_output_path=None):
    """Display pooled out-of-fold curves with a prevalence/chance reference."""
    figure, axis = plt.subplots(figsize=(6.4, 4.8), layout="constrained")
    if kind == "pr":
        # The extra (0, 1) point has no threshold; it is not a measured precision.
        table = table.loc[table.threshold.notna()]
    for index, (method, curve) in enumerate(table.groupby("method", sort=False)):
        options = {"color": COLORS.get(method), "linewidth": 2,
                   "linestyle": LINESTYLES[index % len(LINESTYLES)]}
        if kind == "pr":
            axis.step(curve.recall, curve.precision, where="post", **options,
                      label=f"{method} (AP = {curve.average_precision.iloc[0]:.3f})")
        else:
            axis.plot(curve.false_positive_rate, curve.true_positive_rate, **options,
                      label=f"{method} (AUC = {curve.roc_auc.iloc[0]:.3f})")
    if kind == "pr":
        prevalence = table.prevalence.iloc[0]
        if not np.allclose(table.prevalence, prevalence):
            raise ValueError("PR comparison requires the same class prevalence for all methods.")
        axis.axhline(prevalence, color="0.6", linestyle="--", linewidth=1,
                     label=f"Label prevalence ({prevalence:.3g})")
        axis.set(xlabel="Recall", ylabel="Precision",
                 title="Precision–recall\n" + _comparison_title(table))
        axis.set_ylim(*_score_limits(table.precision, start_at_zero=True))
        axis.text(1, -0.19, "Conventional (recall=0, precision=1) endpoint omitted.",
                  transform=axis.transAxes, ha="right", va="top", fontsize=8, color="0.4")
    else:
        axis.plot([0, 1], [0, 1], color="0.6", linestyle="--", linewidth=1, label="Chance")
        axis.set(xlabel="False positive rate", ylabel="True positive rate",
                 title="ROC\n" + _comparison_title(table), ylim=(0, 1.02))
    axis.set_xlim(0, 1)
    axis.legend(frameon=False, fontsize=9, loc="best")
    _save_plot(figure, output_path, pdf_output_path)


def save_ablation_plot(table, output_path, *, pdf_output_path=None):
    """Pair assay inclusion cells with a zoomed recovery-area dot plot."""
    assays = ("gene_exp", "copy_num", "shRNA", "gene_mut", "CRISPR")
    figure, (matrix, scores) = plt.subplots(
        1, 2, figsize=(8, 7), sharey=True, layout="constrained",
        gridspec_kw={"width_ratios": [2.4, 3.6], "wspace": 0.04},
    )
    used_color, unused_color = "#0072B2", "#E6E6E6"
    matrix.imshow(table[list(assays)].to_numpy(), aspect="auto", vmin=0, vmax=1,
                  cmap=ListedColormap([unused_color, used_color]))
    matrix.grid(False)
    matrix.set_xticks(range(5), [DISPLAY_NAMES[name].replace("\n", " ") for name in assays],
                      rotation=40, ha="left")
    matrix.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False, length=0)
    matrix.set_yticks([])
    matrix.set_xticks(np.arange(-0.5, 5), minor=True)
    matrix.set_yticks(np.arange(-0.5, len(table)), minor=True)
    matrix.grid(which="minor", color="white", linewidth=1.5)
    matrix.tick_params(which="minor", length=0)
    matrix.spines[:].set_visible(False)
    matrix.legend(handles=[Patch(color=used_color, label="Used"),
                           Patch(color=unused_color, label="Omitted")],
                  loc="upper left", bbox_to_anchor=(0, -0.02), ncol=2, frameon=False)
    scores.scatter(table.loocv_area, np.arange(len(table)), s=45, color=used_color, zorder=3)
    for row, area in enumerate(table.loocv_area):
        scores.annotate(f"{area:.3f}", (area, row), xytext=(7, 0),
                        textcoords="offset points", va="center", fontsize=9,
                        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5})
    baseline = table.loc[table[list(assays)].eq(1).all(axis=1), "loocv_area"]
    if len(baseline):
        scores.axvline(baseline.iloc[0], color="0.5", linestyle="--", linewidth=1,
                       label=f"All assays ({baseline.iloc[0]:.3f})")
        scores.legend(loc="lower right", bbox_to_anchor=(1, 1.01), frameon=False, fontsize=9)
    scores.set(xlim=_score_limits(table.loocv_area),
               xlabel="Normalized LOOCV recovery area (zoomed)")
    scores.xaxis.set_major_locator(MaxNLocator(nbins=5))
    scores.grid(axis="y", visible=False)
    scores.tick_params(axis="y", which="both", left=False)
    scores.spines["left"].set_visible(False)
    figure.suptitle("Assay ablation · PDM", fontsize=12)
    _save_plot(figure, output_path, pdf_output_path)


# Add a plot function above, then register (input CSV suffix, output suffix, function).
# Plot functions consume saved results and never rerun benchmark calculations.
PLOTS = {
    "ks": ("ks_pvalues", "ks_heatmap", save_ks_heatmap),
    "loocv": ("loocv_curve", "loocv", save_loocv_plot),
    "labeled": ("method_benchmarks", "method_benchmarks", save_labeled_plot),
    "models": ("model_loocv_curve", "model_loocv", partial(save_loocv_plot, group="model")),
    "ablation": ("ablation", "ablation", save_ablation_plot),
    "pr": ("pr_curve", "pr", save_classification_curve),
    "roc": ("roc_curve", "roc", partial(save_classification_curve, kind="roc")),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir", type=Path, default=PROJECT_ROOT / "runs" / "oncogene" / "results"
    )
    parser.add_argument("--output-dir", type=Path, help="Defaults to RESULTS_DIR/figures.")
    parser.add_argument(
        "--label-name", default="oncogene", help="Prefix of input and output files."
    )
    parser.add_argument("--plots", nargs="+", choices=("all", *PLOTS), default=["all"])
    parser.add_argument("--vmin", type=float, default=None, help="KS heatmap scale minimum.")
    parser.add_argument("--vmax", type=float, default=0.0, help="KS heatmap scale maximum.")
    parser.add_argument("--cmap", default="gist_heat_r", help="KS heatmap colormap.")
    parser.add_argument("--recovery-scale", choices=("count", "fraction"), default="count",
                        help="Y-axis for method and classifier recovery curves.")
    args = parser.parse_args()
    if "all" in args.plots and len(args.plots) > 1:
        parser.error("Use 'all' alone or select individual plots.")
    return args


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.results_dir / "figures"
    selected = PLOTS if args.plots == ["all"] else dict.fromkeys(args.plots)
    written = 0
    for name in selected:
        source_suffix, output_suffix, plot = PLOTS[name]
        source = args.results_dir / f"{args.label_name}_{source_suffix}.csv"
        if not source.is_file():
            if args.plots != ["all"]:
                raise FileNotFoundError(source)
            print(f"Skipping {name}: missing {source}")
            continue
        output = output_dir / f"{args.label_name}_{output_suffix}.png"
        options = {"vmin": args.vmin, "vmax": args.vmax, "cmap": args.cmap} if name == "ks" else {}
        if name in ("loocv", "models"):
            options["recovery_scale"] = args.recovery_scale
        plot(pd.read_csv(source), output, pdf_output_path=output.with_suffix(".pdf"), **options)
        print(f"{name}: {output} and {output.with_suffix('.pdf')}")
        written += 1
    if not written:
        raise FileNotFoundError(f"No plot inputs found in {args.results_dir}.")


if __name__ == "__main__":
    main()
