from __future__ import annotations

import argparse
from pathlib import Path

from pdm_learn.oncogene import save_oncogene_feature_tables


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build oncogene PDM and statistic feature tables from required DepMap_Trimmed inputs."
    )
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--density-name", default="dataset_trimmed_v3.csv")
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
    output_paths = save_oncogene_feature_tables(
        args.data_dir,
        input_paths=input_overrides(args),
        density_name=args.density_name,
        verbose=True,
        progress=True,
    )
    for name, path in output_paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
