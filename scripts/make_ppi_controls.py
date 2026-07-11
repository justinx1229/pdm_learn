from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pdm_learn.ppi import build_control_feature_tables, load_cancer_complexes, load_ppi_inputs


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate PPI positive/negative controls and PDM features from required DepMap_Trimmed inputs."
    )
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--artifacts-dir", type=Path, default=PROJECT_ROOT / "artifacts")
    parser.add_argument("--complexes", type=Path, default=PROJECT_ROOT / "data" / "cancer_complexes_clean.xlsx")
    parser.add_argument("--controls", type=int, default=10)
    parser.add_argument("--max-pairs-per-complex", type=int, default=5)
    parser.add_argument("--negative-pairs", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expression", type=Path, default=None, help="Shared trimmed expression CSV.")
    parser.add_argument("--copy-number", type=Path, default=None, help="Shared trimmed copy-number CSV.")
    parser.add_argument("--shrna", type=Path, default=None, help="Shared trimmed shRNA CSV.")
    parser.add_argument("--mutation", type=Path, default=None, help="Shared trimmed binary mutation CSV.")
    parser.add_argument("--crispr", type=Path, default=None, help="Shared trimmed CRISPR CSV.")
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
    pair_dir = args.data_dir / "PPI_Pairs"
    control_dir = args.artifacts_dir / "controls"
    pair_dir.mkdir(parents=True, exist_ok=True)
    control_dir.mkdir(parents=True, exist_ok=True)

    print("Loading shared DepMap input tables...")
    datasets = load_ppi_inputs(args.data_dir, input_paths=input_overrides(args))
    print(f"Loading cancer-complex workbook from {args.complexes}...")
    complexes = load_cancer_complexes(args.complexes)
    print(
        "Sampling controls and building PDM features "
        f"({args.controls} controls, {args.negative_pairs} negative pairs each)..."
    )
    positive_pairs, negative_pairs, positive_features, negative_features = build_control_feature_tables(
        datasets,
        complexes,
        controls=args.controls,
        max_pairs_per_complex=args.max_pairs_per_complex,
        negative_pairs=args.negative_pairs,
        random_state=args.seed,
        progress=True,
    )

    print(f"Writing pair controls to {pair_dir}...")
    ordinals = ["First", "Second", "Third", "Fourth", "Fifth"]
    for index, (pos_pairs, neg_pairs) in enumerate(zip(positive_pairs, negative_pairs), start=1):
        label = ordinals[index - 1] if index <= len(ordinals) else f"Control{index}"
        pos_pairs.to_csv(pair_dir / f"{label}PositiveControl.csv", index=False)
        neg_pairs.to_csv(pair_dir / f"{label}NegativeControl.csv", index=False)

    print(f"Writing feature controls to {control_dir}...")
    pd.to_pickle(positive_features, control_dir / "positive_controls.pkl")
    pd.to_pickle(negative_features, control_dir / "negative_controls.pkl")
    print(f"pair controls: {pair_dir}")
    print(f"feature controls: {control_dir}")


if __name__ == "__main__":
    main()
