from __future__ import annotations

import argparse
from pathlib import Path

from pdm_learn.ppi import derive_ppi_trimmed_inputs


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create PPI-ready DepMap_Trimmed files from the oncogene "
            "DepMap_data/*_trimmed_Wei.csv files."
        )
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "DepMap_data",
        help="Directory containing the *_trimmed_Wei.csv source files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "DepMap_Trimmed",
        help="Directory where *_Trimmed.csv files will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_paths = derive_ppi_trimmed_inputs(args.source_dir, args.output_dir)
    for name, path in output_paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
