from __future__ import annotations

import argparse
from pathlib import Path

from pdm_learn.oncogene import save_oncogene_feature_tables


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build oncogene PDM and statistic feature tables.")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--density-name", default="dataset_trimmed_v3.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_paths = save_oncogene_feature_tables(
        args.data_dir,
        density_name=args.density_name,
    )
    for name, path in output_paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
