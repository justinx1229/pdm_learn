"""Render a block heatmap from oncogene PDM KS-test statistics."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a block heatmap from oncogene PDM KS-test results."
    )
    parser.add_argument(
        "--ks-file",
        type=Path,
        default=PROJECT_ROOT / "runs" / "oncogene" / "results" / "oncogene_ks_pvalues.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "runs" / "oncogene" / "results" / "oncogene_ks_heatmap.png",
    )
    parser.add_argument(
        "--pdf-output",
        type=Path,
        default=None,
        help="Optional PDF output path. Defaults to OUTPUT with a .pdf suffix.",
    )
    parser.add_argument("--value-column", default="log10_ks_pvalue")
    parser.add_argument("--vmin", type=float, default=None)
    parser.add_argument("--vmax", type=float, default=0.0)
    parser.add_argument("--cmap", default="gist_heat_r")
    parser.add_argument("--title", default="Oncogene PDM feature differences")
    return parser.parse_args()


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
        matrix = ordered[value_column].to_numpy(dtype=float).reshape(
            AXIS_WIDTHS[right], AXIS_WIDTHS[left]
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    if pdf_output_path is not None:
        pdf_output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(pdf_output_path, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    table = pd.read_csv(args.ks_file)
    pdf_path = args.pdf_output or args.output.with_suffix(".pdf")
    save_ks_heatmap(
        table,
        args.output,
        value_column=args.value_column,
        vmin=args.vmin,
        vmax=args.vmax,
        cmap=args.cmap,
        title=args.title,
        pdf_output_path=pdf_path,
    )
    print(f"KS heatmap: {args.output}")
    print(f"KS heatmap PDF: {pdf_path}")


if __name__ == "__main__":
    main()
