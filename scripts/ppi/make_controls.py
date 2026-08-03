"""Build reproducible PPI control feature tables from the frozen DepMap inputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pdm_learn.workflows.ppi import (
    build_control_feature_tables,
    load_biogrid_pairs,
    load_cancer_complexes,
    load_inputs,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build PDM positive and exclusion-based negative PPI controls."
    )
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument(
        "--complexes", type=Path, default=PROJECT_ROOT / "data" / "cancer_complexes_clean.xlsx"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "runs" / "ppi" / "controls"
    )
    parser.add_argument("--controls", type=int, default=10)
    parser.add_argument("--max-pairs-per-complex", type=int, default=5)
    parser.add_argument("--negative-pairs", type=int, default=5_000)
    parser.add_argument(
        "--negative-pair-candidates",
        type=Path,
        required=True,
        help=(
            "CSV defining the negative-control sampling population. It must contain "
            "Gene1/Gene2 or Gene_A/Gene_B columns (for example, BioGRID pairs)."
        ),
    )
    parser.add_argument("--random-state", type=int, default=42)
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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = load_inputs(args.data_dir, input_paths=input_overrides(args))
    positives, negatives = build_control_feature_tables(
        datasets,
        load_cancer_complexes(args.complexes),
        negative_candidate_pairs=load_biogrid_pairs(args.negative_pair_candidates),
        controls=args.controls,
        max_pairs_per_complex=args.max_pairs_per_complex,
        negative_pairs=args.negative_pairs,
        random_state=args.random_state,
        progress=True,
    )
    positive_path = args.output_dir / "positive_controls.pkl"
    negative_path = args.output_dir / "negative_controls.pkl"
    pd.to_pickle(positives, positive_path)
    pd.to_pickle(negatives, negative_path)
    print(f"positive controls: {positive_path}")
    print(f"negative controls: {negative_path}")


if __name__ == "__main__":
    main()
