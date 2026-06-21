from __future__ import annotations

import argparse
from pathlib import Path

from pdm_learn.ppi import build_ppi_feature_table, load_biogrid_pairs, load_ppi_inputs


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a PDM feature table for BioGRID PPI pairs from required DepMap_Trimmed inputs."
    )
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--artifacts-dir", type=Path, default=PROJECT_ROOT / "artifacts")
    parser.add_argument("--biogrid", type=Path, default=PROJECT_ROOT / "data" / "clean_biogrid_interactions.csv")
    parser.add_argument("--output-name", default="clean_biogrid_interactions_pdm.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.artifacts_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = load_ppi_inputs(args.data_dir)
    pairs = load_biogrid_pairs(args.biogrid)
    features = build_ppi_feature_table(datasets, pairs)
    output_path = output_dir / args.output_name
    features.to_csv(output_path, index=False)
    print(f"PPI feature table: {output_path}")


if __name__ == "__main__":
    main()
