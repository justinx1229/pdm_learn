from __future__ import annotations

import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build gene-level PDM and statistic feature tables from gene-by-sample matrices."
    )
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for generated feature tables.")
    parser.add_argument("--density-name", default="dataset_trimmed_v3.csv")
    parser.add_argument("--expression", type=Path, default=None, help="Shared trimmed expression CSV.")
    parser.add_argument("--copy-number", type=Path, default=None, help="Shared trimmed copy-number CSV.")
    parser.add_argument("--shrna", type=Path, default=None, help="Shared trimmed shRNA CSV.")
    parser.add_argument("--mutation", type=Path, default=None, help="Shared trimmed binary mutation CSV.")
    parser.add_argument("--crispr", type=Path, default=None, help="Shared trimmed CRISPR CSV.")
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        metavar="NAME=CSV",
        help="Generic input matrix. Repeat for arbitrary datasets, e.g. --input methylation=data/methylation.csv.",
    )
    parser.add_argument(
        "--dataset-kind",
        action="append",
        default=[],
        metavar="NAME=KIND",
        help="Dataset kind for generic inputs: continuous, binary, or discrete.",
    )
    parser.add_argument(
        "--dataset-levels",
        action="append",
        default=[],
        metavar="NAME=V1,V2",
        help="Comma-separated discrete levels for a binary/discrete dataset.",
    )
    parser.add_argument(
        "--pair",
        action="append",
        default=[],
        metavar="LEFT:RIGHT",
        help="Dataset pair to featurize. Repeat to control pair order. Defaults to legacy DepMap pairs or all generic combinations.",
    )
    parser.add_argument(
        "--methods",
        default="pearson,spearman,mi,bicor",
        help="Comma-separated statistic feature methods to build.",
    )
    return parser.parse_args()


def _parse_name_value(value: str, separator: str = "=") -> tuple[str, str]:
    if separator not in value:
        raise ValueError(f"Expected NAME{separator}VALUE, got {value!r}")
    name, text = value.split(separator, maxsplit=1)
    name = name.strip()
    text = text.strip()
    if not name or not text:
        raise ValueError(f"Expected NAME{separator}VALUE, got {value!r}")
    return name, text


def input_overrides(args: argparse.Namespace) -> dict[str, Path]:
    paths = {
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
    for value in args.input:
        name, path = _parse_name_value(value)
        paths[name] = Path(path)
    return paths


def dataset_specs(args: argparse.Namespace) -> dict[str, dict[str, object]]:
    kinds = dict(_parse_name_value(value) for value in args.dataset_kind)
    levels = {}
    for value in args.dataset_levels:
        name, text = _parse_name_value(value)
        levels[name] = tuple(float(part.strip()) for part in text.split(",") if part.strip())
    return {
        name: {"kind": kind, "levels": levels.get(name)}
        for name, kind in kinds.items()
    }


def pair_specs(args: argparse.Namespace) -> list[str] | None:
    if not args.pair:
        return None
    for value in args.pair:
        _parse_name_value(value, separator=":")
    return args.pair


def method_names(args: argparse.Namespace) -> list[str]:
    return [method.strip() for method in args.methods.split(",") if method.strip()]


def main() -> None:
    args = parse_args()
    from pdm_learn.oncogene import save_oncogene_feature_tables

    output_paths = save_oncogene_feature_tables(
        args.data_dir,
        input_paths=input_overrides(args),
        output_dir=args.output_dir,
        pairs=pair_specs(args),
        dataset_specs=dataset_specs(args),
        feature_methods=method_names(args),
        density_name=args.density_name,
        verbose=True,
        progress=True,
    )
    for name, path in output_paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
