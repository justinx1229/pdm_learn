"""Plot saved oncogene KS statistics and benchmark results; no model fitting."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pdm_learn_matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

PROJECT_ROOT = Path(__file__).resolve().parents[2]


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

    figure = plt.figure(figsize=(12.5, 10.5))
    grid = figure.add_gridspec(
        len(ROW_ORDER),
        len(COLUMN_ORDER),
        width_ratios=[AXIS_WIDTHS[name] for name in COLUMN_ORDER],
        height_ratios=[AXIS_WIDTHS[name] for name in ROW_ORDER],
        left=0.19,
        right=0.83,
        bottom=0.12,
        top=0.88,
        wspace=0.07,
        hspace=0.07,
    )
    axes = [[figure.add_subplot(grid[row, column]) for column in range(5)] for row in range(5)]
    image = None
    for row, left in enumerate(ROW_ORDER):
        for column, right in enumerate(COLUMN_ORDER):
            axis = axes[row][column]
            matrix = blocks.get(f"{left}.{right}")
            if matrix is None:
                axis.set_axis_off()
                continue
            image = axis.imshow(np.flipud(matrix), cmap=cmap, norm=norm, aspect="auto")
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)

    figure.canvas.draw()
    for row, name in enumerate(ROW_ORDER):
        position = axes[row][0].get_position()
        figure.text(
            0.17,
            (position.y0 + position.y1) / 2,
            DISPLAY_NAMES[name],
            ha="right",
            va="center",
            fontsize=15,
            fontweight="bold",
            color="0.35",
        )
    for column, name in enumerate(COLUMN_ORDER):
        position = axes[0][column].get_position()
        figure.text(
            (position.x0 + position.x1) / 2,
            0.895,
            DISPLAY_NAMES[name],
            ha="center",
            va="bottom",
            rotation=-90,
            fontsize=14,
            fontweight="bold",
            color="0.35",
        )

    if image is None:
        raise RuntimeError("No KS blocks were plotted.")
    colorbar_axis = figure.add_axes([0.87, 0.16, 0.025, 0.68])
    colorbar = figure.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("log10 KS p-value", fontsize=13)
    figure.suptitle(title, fontsize=18, fontweight="bold", y=1.08)
    _save_plot(figure, output_path, pdf_output_path)


def save_loocv_plot(
    curves: pd.DataFrame,
    output_path: Path,
    *,
    pdf_output_path: Path | None = None,
) -> None:
    """Plot saved rank-recovery curves; their areas are not ROC AUC."""
    figure, axis = plt.subplots(figsize=(7.5, 5))
    for method, curve in curves.groupby("method", sort=False):
        curve = curve.sort_values("rank_threshold")
        area = float(curve["loocv_area"].iloc[0])
        candidates = int(curve["unlabeled_candidate_rows"].iloc[0])
        positives = int(curve["positive_rows"].iloc[0])
        axis.step(
            curve["rank_threshold"],
            curve["recovered_fraction"],
            where="post",
            linewidth=2,
            label=(
                f"{method} (area = {area:.3f})\n"
                f"{candidates:,} unlabeled candidates, {positives:,} positives"
            ),
        )
    axis.set_xlabel("Rank threshold (zero-based; lower is better)")
    axis.set_ylabel("Fraction of held-out positives recovered")
    axis.set_title("Leave-one-positive-out rank recovery")
    axis.set_ylim(0, 1.02)
    axis.legend(title="Feature table", frameon=False)
    figure.tight_layout()
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
        axis.scatter(np.arange(len(table)), table[metric])
        axis.set(
            xticks=np.arange(len(table)), xticklabels=table["method"], ylabel=label, ylim=(0, 1.02)
        )
        axis.tick_params(axis="x", labelrotation=30)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Classification against explicit negative controls")
    _save_plot(figure, output_path, pdf_output_path)


# Add a plot function above, then register (input CSV suffix, output suffix, function).
# Plot functions consume saved results and never rerun benchmark calculations.
PLOTS = {
    "ks": ("ks_pvalues", "ks_heatmap", save_ks_heatmap),
    "loocv": ("loocv_curve", "loocv", save_loocv_plot),
    "labeled": ("method_benchmarks", "method_benchmarks", save_labeled_plot),
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
        plot(pd.read_csv(source), output, pdf_output_path=output.with_suffix(".pdf"), **options)
        print(f"{name}: {output} and {output.with_suffix('.pdf')}")
        written += 1
    if not written:
        raise FileNotFoundError(f"No plot inputs found in {args.results_dir}.")


if __name__ == "__main__":
    main()
