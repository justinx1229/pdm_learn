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


def generic_dataset_specs(args: argparse.Namespace):
    """Create generic specs only for explicitly configured custom inputs."""
    from pdm_learn.datasets import DatasetSpec

    return [DatasetSpec(name, **options) for name, options in dataset_specs(args).items()]


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
    paths = input_overrides(args)
    methods = ("pdm", *method_names(args))
    output_dir = args.output_dir or args.data_dir / "Trimmed data"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.input:
        # Arbitrary matrices use only the generic core.  The oncogene-named
        # script remains a convenient historical entry point, not a dependency
        # of the reusable library.
        from pdm_learn.datasets import load_matrices
        from pdm_learn.features import build_features

        print("Loading custom matrices...")
        datasets = load_matrices(paths, specs=generic_dataset_specs(args))
        tables = build_features(
            datasets,
            specs=generic_dataset_specs(args),
            pairs=pair_specs(args),
            methods=methods,
            output_id_column="gene name",
            progress=True,
        )
    else:
        from pdm_learn.workflows.oncogene import build_features, load_inputs

        print("Loading DepMap oncogene preset inputs...")
        datasets = load_inputs(args.data_dir, input_paths=paths)
        tables = build_features(
            datasets,
            specs=dataset_specs(args) or None,
            pairs=pair_specs(args),
            methods=methods,
            id_column="gene name",
            progress=True,
        )

    filenames = {
        "pdm": args.density_name,
        "pearson": "pearson.csv",
        "spearman": "spearman.csv",
        "mi": "mi.csv",
        "bicor": "bicor.csv",
    }
    output_paths = {}
    for name, table in tables.items():
        path = output_dir / filenames[name]
        table.to_csv(path, index=False)
        output_paths[name] = path
    for name, path in output_paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
