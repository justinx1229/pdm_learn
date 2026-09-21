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
        help="Input directory for study, demo, or custom datasets. Defaults to data/.",
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


def run_oncogene(data_dir: Path, output_dir: Path, settings: dict[str, object]) -> None:
    features = output_dir / "oncogene" / "features"
    results = output_dir / "oncogene" / "results"
    _run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "oncogene" / "make_features.py"),
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(features),
        ],
        stage="Oncogene 1/2: Building features",
    )
    _run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "oncogene" / "run_predictions.py"),
            "--data-dir",
            str(data_dir),
            "--feature-dir",
            str(features),
            "--results-dir",
            str(results),
            "--model",
            str(settings["model"]),
            "--ranking-trials",
            str(settings["ranking_trials"]),
            "--n-splits",
            str(settings["n_splits"]),
            "--random-state",
            str(settings["random_state"]),
        ],
        stage="Oncogene 2/2: Ranking genes and calculating KS statistics",
    )


def run_ppi(data_dir: Path, output_dir: Path, settings: dict[str, object]) -> None:
    features = output_dir / "ppi" / "features"
    results = output_dir / "ppi" / "results"
    controls = output_dir / "ppi" / "controls"
    biogrid = data_dir / "clean_biogrid_interactions.csv"
    _run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "ppi" / "make_features.py"),
            "--data-dir",
            str(data_dir),
            "--biogrid",
            str(biogrid),
            "--output-dir",
            str(features),
        ],
        stage="PPI 1/3: Building pair features",
    )
    complex_name = (
        "cancer_complexes_clean.xlsx"
        if (data_dir / "cancer_complexes_clean.xlsx").exists()
        else "cancer_complexes.csv"
    )
    complexes = data_dir / complex_name
    _run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "ppi" / "make_controls.py"),
            "--data-dir",
            str(data_dir),
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
    workflows = ("oncogene", "ppi", "simulation") if args.workflow == "all" else (args.workflow,)
    for workflow in workflows:
        if workflow == "oncogene":
            run_oncogene(args.data_dir, args.output_dir, settings[workflow])
        elif workflow == "ppi":
            run_ppi(args.data_dir, args.output_dir, settings[workflow])
        else:
            run_simulation(args.output_dir, settings[workflow])


if __name__ == "__main__":
    main()
