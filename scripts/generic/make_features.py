from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
METHOD_LABELS = {
    "pearson": "Pearson",
    "spearman": "Spearman",
    "mi": "MutualInfo",
    "bicor": "Bicor",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build generic gene-level PDM and statistic feature tables from gene-by-sample matrices."
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        metavar="NAME=CSV",
        help="Input matrix with one gene column followed by sample columns. Repeat for each dataset.",
    )
    parser.add_argument(
        "--dataset-kind",
        action="append",
        default=[],
        metavar="NAME=KIND",
        help="Dataset kind: continuous, binary, or discrete. Defaults to continuous.",
    )
    parser.add_argument(
        "--dataset-levels",
        action="append",
        default=[],
        metavar="NAME=V1,V2",
        help="Comma-separated levels for binary/discrete datasets. Inferred if omitted.",
    )
    parser.add_argument(
        "--dataset-normalize",
        action="append",
        default=[],
        metavar="NAME=true|false",
        help="Override row-centering for a dataset. Continuous defaults to true; binary/discrete defaults to false.",
    )
    parser.add_argument(
        "--pair",
        action="append",
        default=[],
        metavar="LEFT:RIGHT",
        help="Dataset pair to featurize. Repeat to control pair order. Defaults to all unique dataset combinations.",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "generic" / "features")
    parser.add_argument("--pdm-name", default="pdm.csv")
    parser.add_argument("--methods", default="pearson,spearman,mi,bicor")
    parser.add_argument("--boxes", type=int, default=7)
    parser.add_argument("--gene-universe", choices=("union", "intersection"), default="union")
    parser.add_argument("--gene-column-name", default="gene name")
    parser.add_argument("--log-offset", type=float, default=None)
    parser.add_argument("--keep-missing", action="store_true", help="Keep genes with missing feature blocks instead of dropping rows with NaNs.")
    parser.add_argument("--manifest-name", default="feature_manifest.json")
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


def _parse_bool(text: str) -> bool:
    normalized = text.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Expected a boolean value, got {text!r}")


def input_paths(args: argparse.Namespace) -> dict[str, Path]:
    paths = {}
    for value in args.input:
        name, path_text = _parse_name_value(value)
        paths[name] = Path(path_text)
    return paths


def dataset_specs(args: argparse.Namespace, names: list[str]) -> dict[str, dict[str, Any]]:
    kinds = dict(_parse_name_value(value) for value in args.dataset_kind)
    normalize = {
        name: _parse_bool(text)
        for name, text in (_parse_name_value(value) for value in args.dataset_normalize)
    }
    levels = {}
    for value in args.dataset_levels:
        name, text = _parse_name_value(value)
        levels[name] = tuple(float(part.strip()) for part in text.split(",") if part.strip())

    unknown = (set(kinds) | set(normalize) | set(levels)) - set(names)
    if unknown:
        raise ValueError(f"Metadata provided for unknown datasets: {', '.join(sorted(unknown))}")

    specs = {}
    for name in names:
        kind = kinds.get(name, "continuous")
        specs[name] = {
            "kind": kind,
            "levels": levels.get(name),
            "normalize": normalize.get(name),
        }
    return specs


def pair_specs(args: argparse.Namespace) -> list[str] | None:
    if not args.pair:
        return None
    for value in args.pair:
        _parse_name_value(value, separator=":")
    return args.pair


def method_names(args: argparse.Namespace) -> list[str]:
    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    invalid = sorted(set(methods) - set(METHOD_LABELS))
    if invalid:
        raise ValueError(f"Unsupported statistic methods: {', '.join(invalid)}")
    return methods


def _path_for_manifest(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def load_gene_matrices(paths: dict[str, Path]):
    import pandas as pd

    datasets = {}
    for name, path in paths.items():
        dataframe = pd.read_csv(path, low_memory=False)
        dataframe = dataframe.copy()
        dataframe.iloc[:, 0] = dataframe.iloc[:, 0].astype(str).str.strip()
        dataframe.name = name
        datasets[name] = dataframe
    return datasets


def main() -> None:
    args = parse_args()

    from pdm_learn import build_gene_density_features, build_gene_statistic_features

    paths = input_paths(args)
    names = list(paths)
    specs = dataset_specs(args, names)
    pairs = pair_specs(args)
    methods = method_names(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = load_gene_matrices(paths)

    feature_tables = {}
    pdm_path = args.output_dir / args.pdm_name
    print(f"Building PDM features -> {pdm_path}")
    build_gene_density_features(
        datasets,
        pairs=pairs,
        dataset_specs=specs,
        gene_universe=args.gene_universe,
        gene_column_name=args.gene_column_name,
        boxes=args.boxes,
        log_offset=args.log_offset,
        drop_missing=not args.keep_missing,
        progress=True,
    ).to_csv(pdm_path, index=False)
    feature_tables["PDM"] = pdm_path

    for method in methods:
        label = METHOD_LABELS[method]
        path = args.output_dir / f"{method}.csv"
        print(f"Building {label} features -> {path}")
        build_gene_statistic_features(
            datasets,
            method=method,
            pairs=pairs,
            dataset_specs=specs,
            gene_universe=args.gene_universe,
            gene_column_name=args.gene_column_name,
            progress=True,
        ).to_csv(path, index=False)
        feature_tables[label] = path

    manifest = {
        "inputs": {name: _path_for_manifest(path) for name, path in paths.items()},
        "dataset_specs": specs,
        "pairs": pairs or "all_unique_combinations",
        "feature_tables": {name: _path_for_manifest(path) for name, path in feature_tables.items()},
        "settings": {
            "boxes": args.boxes,
            "gene_universe": args.gene_universe,
            "gene_column_name": args.gene_column_name,
            "drop_missing": not args.keep_missing,
            "log_offset": args.log_offset,
        },
    }
    manifest_path = args.output_dir / args.manifest_name
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
