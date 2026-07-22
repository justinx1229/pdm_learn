"""Build PDM feature rows for the supplied BioGRID interaction table."""

from __future__ import annotations

import argparse
from pathlib import Path

from pdm_learn.workflows.ppi import build_features, load_biogrid_pairs, load_inputs

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build PDM features for BioGRID pairs using the shared trimmed DepMap inputs."
    )
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--biogrid", type=Path, default=None)
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "runs" / "ppi" / "features"
    )
    parser.add_argument("--output-name", default="biogrid_pdm.csv")
    parser.add_argument("--expression", type=Path, default=None)
    parser.add_argument("--copy-number", type=Path, default=None)
    parser.add_argument("--shrna", type=Path, default=None)
    parser.add_argument("--mutation", type=Path, default=None)
    parser.add_argument("--crispr", type=Path, default=None)
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
    biogrid_path = args.biogrid or args.data_dir / "clean_biogrid_interactions.csv"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    datasets = load_inputs(args.data_dir, input_paths=input_overrides(args))
    pairs = load_biogrid_pairs(biogrid_path)
    features = build_features(datasets, pairs, methods=("pdm",), progress=True)["pdm"]
    output_path = args.output_dir / args.output_name
    features.to_csv(output_path, index=False)
    print(f"PPI feature table: {output_path}")


if __name__ == "__main__":
    main()
