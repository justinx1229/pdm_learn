"""Build oncogene PDM and association feature tables from supplied DepMap inputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from pdm_learn.workflows.oncogene import build_features, load_inputs

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build gene-level feature tables from trimmed DepMap inputs."
    )
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "runs" / "oncogene" / "features",
    )
    parser.add_argument("--density-name", default="dataset_trimmed_package.csv")
    parser.add_argument("--expression", type=Path, default=None)
    parser.add_argument("--copy-number", type=Path, default=None)
    parser.add_argument("--shrna", type=Path, default=None)
    parser.add_argument("--mutation", type=Path, default=None)
    parser.add_argument("--crispr", type=Path, default=None)
    parser.add_argument("--methods", default="pearson,spearman,mi,bicor")
    return parser.parse_args()


def input_overrides(args: argparse.Namespace) -> dict[str, Path]:
    return {
        key: path
        for key, path in {
            "expression": args.expression,
            "copy_number": args.copy_number,
            "shrna": args.shrna,
            "mutation": args.mutation,
            "crispr": args.crispr,
        }.items()
        if path is not None
    }


def main() -> None:
    args = parse_args()
    requested_methods = [name.strip() for name in args.methods.split(",") if name.strip()]
    methods = tuple(dict.fromkeys(("pdm", *requested_methods)))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = load_inputs(args.data_dir, input_paths=input_overrides(args))
    tables = build_features(
        datasets,
        methods=methods,
        output_id_column="gene name",
        progress=True,
    )
    filenames = {
        "pdm": args.density_name,
        "pearson": "pearson.csv",
        "spearman": "spearman.csv",
        "mi": "mi.csv",
        "bicor": "bicor.csv",
    }
    for name, table in tables.items():
        output_path = args.output_dir / filenames[name]
        table.to_csv(output_path, index=False)
        print(f"{name}: {output_path}")


if __name__ == "__main__":
    main()
