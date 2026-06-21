from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pdm_learn.oncogene import trim_oncogene_input_tables
from pdm_learn.ppi import derive_shared_trimmed_inputs


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the shared DepMap_Trimmed inputs used by both oncogene and "
            "PPI workflows."
        )
    )
    parser.add_argument("--source-dir", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument(
        "--intermediate-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "DepMap_data",
        help="Directory for intermediate *_trimmed_Wei.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "DepMap_Trimmed",
        help="Directory for required shared *_Trimmed.csv analysis inputs.",
    )
    parser.add_argument("--expression", type=Path, default=None, help="Raw gene-expression matrix.")
    parser.add_argument("--copy-number", type=Path, default=None, help="Raw copy-number matrix.")
    parser.add_argument("--shrna", type=Path, default=None, help="Raw shRNA matrix.")
    parser.add_argument("--mutation", type=Path, default=None, help="Raw mutation table.")
    parser.add_argument("--crispr", type=Path, default=None, help="Raw CRISPR dependency/effect matrix.")
    parser.add_argument(
        "--matrix-orientation",
        choices=("genes-as-rows", "samples-as-rows"),
        default="genes-as-rows",
        help="Orientation of expression/copy-number/shRNA/CRISPR matrices.",
    )
    parser.add_argument(
        "--gene-list",
        type=Path,
        default=None,
        help="Optional one-column file limiting output to a gene set.",
    )
    parser.add_argument("--gene-column", default=None, help="Gene column for genes-as-rows matrices.")
    parser.add_argument("--sample-column", default=None, help="Cell-line/sample column for samples-as-rows matrices.")
    parser.add_argument("--mutation-gene-column", default=None, help="Gene column in the mutation table.")
    parser.add_argument("--mutation-cell-line-column", default=None, help="Cell-line column in the mutation table.")
    parser.add_argument(
        "--no-align-cell-lines",
        action="store_true",
        help="Do not restrict matrix columns to shared cell lines.",
    )
    parser.add_argument(
        "--from-intermediate",
        action="store_true",
        help="Start from existing intermediate *_trimmed_Wei.csv files instead of raw CCLE/DepMap files.",
    )
    return parser.parse_args()


def _default_path(source_dir: Path, filename: str) -> Path:
    return source_dir / filename


def _load_gene_list(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    return pd.read_csv(path).iloc[:, 0].astype(str).str.strip().tolist()


def main() -> None:
    args = parse_args()
    if args.from_intermediate:
        shared_paths = derive_shared_trimmed_inputs(args.intermediate_dir, args.output_dir)
        for name, path in shared_paths.items():
            print(f"{name}: {path}")
        return

    raw_paths = {
        "expression": args.expression or _default_path(args.source_dir, "CCLE_gene_expression.csv"),
        "copy_number": args.copy_number or _default_path(args.source_dir, "CCLE_gene_cn.csv"),
        "shrna": args.shrna or _default_path(args.source_dir, "shRNA_Broad.csv"),
        "mutation": args.mutation or _default_path(args.source_dir, "CCLE_gene_mutation.csv"),
        "crispr": args.crispr or _default_path(args.source_dir, "Avana_gene_effect_20Q3.csv"),
    }

    for name, path in raw_paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {name} raw file: {path}. Pass --{name.replace('_', '-')} "
                "or place the file under --source-dir with the default name."
            )

    output_paths = trim_oncogene_input_tables(
        raw_paths,
        args.intermediate_dir,
        matrix_orientation=args.matrix_orientation,
        gene_list=_load_gene_list(args.gene_list),
        align_cell_lines=not args.no_align_cell_lines,
        gene_column=args.gene_column,
        sample_column=args.sample_column,
        mutation_gene_column=args.mutation_gene_column,
        mutation_cell_line_column=args.mutation_cell_line_column,
    )

    for name, path in output_paths.items():
        print(f"{name}: {path}")

    shared_paths = derive_shared_trimmed_inputs(args.intermediate_dir, args.output_dir)
    for name, path in shared_paths.items():
        print(f"required_{name}: {path}")


if __name__ == "__main__":
    main()
