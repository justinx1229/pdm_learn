"""Run oncogene, PPI, and simulation workflows with the supplied data and settings."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "reproduction" / "workflows.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run deterministic oncogene, PPI, and simulation reproduction workflows."
    )
    parser.add_argument(
        "workflow", choices=("oncogene", "ppi", "simulation", "all")
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Directory containing input files directly. Defaults to data/.",
    )
    for flag in ("expression", "copy-number", "shrna", "mutation", "crispr"):
        parser.add_argument(
            f"--{flag}", type=Path,
            help="Matrix file path (overrides the default in --data-dir).",
        )
    parser.add_argument(
        "--label-file", type=Path,
        help="Oncogene labels (default: <data-dir>/oncogene.txt).",
    )
    parser.add_argument(
        "--biogrid", type=Path,
        help="PPI candidate pairs (default: <data-dir>/clean_biogrid_interactions.csv).",
    )
    parser.add_argument(
        "--complexes", type=Path,
        help=(
            "PPI complex table (default: cancer_complexes_clean.xlsx "
            "or cancer_complexes.csv in data-dir)."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "runs")
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="Workflow settings JSON. Defaults to reproduction/workflows.json.",
    )
    return parser.parse_args()


def _run(command: list[str], *, stage: str) -> None:
    print(f"\n[{stage}] Starting", flush=True)
    print("+", " ".join(command), flush=True)
    started = perf_counter()
    subprocess.run(command, check=True)
    print(f"[{stage}] Completed in {perf_counter() - started:.1f}s", flush=True)


def run_oncogene(
    data_dir: Path,
    output_dir: Path,
    settings: dict[str, object],
    *,
    matrix_args: tuple[str, ...] = (),
    label_file: Path | None = None,
) -> None:
    features = output_dir / "oncogene" / "features"
    results = output_dir / "oncogene" / "results"
    _run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "oncogene" / "make_features.py"),
            "--data-dir",
            str(data_dir),
            *matrix_args,
            "--output-dir",
            str(features),
        ],
        stage="Oncogene 1/2: Building features",
    )
    _run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "oncogene" / "run_predictions.py"),
            "--label-file",
            str(label_file or data_dir / "oncogene.txt"),
            "--feature-dir",
            str(features),
            "--results-dir",
            str(results),
            "--model",
            str(settings["model"]),
            "--ranking-trials",
            str(settings["ranking_trials"]),
            "--random-state",
            str(settings["random_state"]),
        ],
        stage="Oncogene 2/2: Ranking genes",
    )


def run_ppi(
    data_dir: Path,
    output_dir: Path,
    settings: dict[str, object],
    *,
    matrix_args: tuple[str, ...] = (),
    biogrid: Path | None = None,
    complexes: Path | None = None,
) -> None:
    features = output_dir / "ppi" / "features"
    results = output_dir / "ppi" / "results"
    controls = output_dir / "ppi" / "controls"
    biogrid = biogrid or data_dir / "clean_biogrid_interactions.csv"
    if complexes is None:
        workbook = data_dir / "cancer_complexes_clean.xlsx"
        complexes = workbook if workbook.exists() else data_dir / "cancer_complexes.csv"
    _run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "ppi" / "make_features.py"),
            "--data-dir",
            str(data_dir),
            *matrix_args,
            "--biogrid",
            str(biogrid),
            "--output-dir",
            str(features),
        ],
        stage="PPI 1/3: Building pair features",
    )
    _run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "ppi" / "make_controls.py"),
            "--data-dir",
            str(data_dir),
            *matrix_args,
            "--complexes",
            str(complexes),
            "--output-dir",
            str(controls),
            "--controls",
            str(settings["controls"]),
            "--max-pairs-per-complex",
            str(settings["max_pairs_per_complex"]),
            "--negative-pairs",
            str(settings["negative_pairs"]),
            "--negative-pair-candidates",
            str(biogrid),
            "--random-state",
            str(settings["random_state"]),
        ],
        stage="PPI 2/3: Generating control sets",
    )
    _run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "ppi" / "run_predictions.py"),
            "--features",
            str(features / "biogrid_pdm.csv"),
            "--complexes",
            str(complexes),
            "--results-dir",
            str(results),
            "--model",
            str(settings["model"]),
            "--trials",
            str(settings["trials"]),
            "--max-pairs-per-complex",
            str(settings.get("ranking_max_pairs_per_complex", 10)),
            "--n-splits",
            str(settings["n_splits"]),
            "--random-state",
            str(settings["random_state"]),
            "--positive-controls",
            str(controls / "positive_controls.pkl"),
            "--negative-controls",
            str(controls / "negative_controls.pkl"),
        ],
        stage="PPI 3/3: Ranking pairs and benchmarking controls",
    )


def run_simulation(output_dir: Path, settings: dict[str, object]) -> None:
    features = output_dir / "simulation" / "features"
    results = output_dir / "simulation" / "results"
    _run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "simulation" / "make_features.py"),
            "--output-dir",
            str(features),
            "--positive-repeats",
            str(settings["positive_repeats"]),
            "--negative-repeats",
            str(settings["negative_repeats"]),
            "--seed",
            str(settings["random_state"]),
        ],
        stage="Simulation 1/2: Building features",
    )
    _run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "simulation" / "run_benchmarks.py"),
            "--feature-dir",
            str(features),
            "--results-dir",
            str(results),
            "--model",
            str(settings["model"]),
            "--n-splits",
            str(settings["n_splits"]),
            "--random-state",
            str(settings["random_state"]),
        ],
        stage="Simulation 2/2: Running benchmarks",
    )


def main() -> None:
    args = parse_args()
    settings = json.loads(args.config.read_text())
    matrix_args = tuple(
        value
        for name in ("expression", "copy_number", "shrna", "mutation", "crispr")
        if (path := getattr(args, name)) is not None
        for value in (f"--{name.replace('_', '-')}", str(path))
    )
    workflows = ("oncogene", "ppi", "simulation") if args.workflow == "all" else (args.workflow,)
    for workflow in workflows:
        if workflow == "oncogene":
            run_oncogene(
                args.data_dir, args.output_dir, settings[workflow],
                matrix_args=matrix_args, label_file=args.label_file,
            )
        elif workflow == "ppi":
            run_ppi(
                args.data_dir, args.output_dir, settings[workflow],
                matrix_args=matrix_args, biogrid=args.biogrid, complexes=args.complexes,
            )
        else:
            run_simulation(args.output_dir, settings[workflow])


if __name__ == "__main__":
    main()
