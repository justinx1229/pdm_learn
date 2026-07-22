"""Run the reviewer-facing workflows with one frozen configuration."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "reproduction" / "workflows.json"
MANIFEST_PATH = PROJECT_ROOT / "reproduction" / "input_manifest.json"
SMOKE_DATA_DIR = PROJECT_ROOT / "tests" / "fixtures" / "reproduction"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run deterministic oncogene, PPI, and simulation reproduction workflows."
    )
    parser.add_argument(
        "workflow", choices=("oncogene", "ppi", "simulation", "all", "verify-inputs")
    )
    parser.add_argument("--profile", choices=("full", "smoke"), default="full")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--skip-input-verification",
        action="store_true",
        help="Do not validate the full study-data bundle against reproduction/input_manifest.json.",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_inputs(data_dir: Path) -> None:
    manifest = json.loads(MANIFEST_PATH.read_text())
    missing = []
    mismatches = []
    for relative_path, expected in manifest["files"].items():
        path = data_dir / relative_path
        if not path.is_file():
            missing.append(relative_path)
        elif _sha256(path) != expected:
            mismatches.append(relative_path)
    if missing or mismatches:
        details = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if mismatches:
            details.append(f"checksum mismatch: {', '.join(mismatches)}")
        raise SystemExit("Study-data verification failed (" + "; ".join(details) + ").")
    print(f"verified frozen study inputs: {data_dir}")


def _run(command: list[str]) -> None:
    print("+", " ".join(command))
    subprocess.run(command, check=True)


def _settings(profile: str) -> dict[str, dict[str, object]]:
    settings = json.loads(CONFIG_PATH.read_text())
    if profile == "smoke":
        settings["oncogene"].update({"ranking_trials": 3, "n_splits": 2})
        settings["ppi"].update({"trials": 3, "n_splits": 2})
        settings["simulation"].update({"negative_repeats": 3, "n_splits": 2})
    return settings


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
        ]
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
        ]
    )


def run_ppi(data_dir: Path, output_dir: Path, settings: dict[str, object]) -> None:
    features = output_dir / "ppi" / "features"
    results = output_dir / "ppi" / "results"
    _run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "ppi" / "make_features.py"),
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(features),
        ]
    )
    complex_name = (
        "cancer_complexes.csv"
        if (data_dir / "cancer_complexes.csv").exists()
        else "cancer_complexes_clean.xlsx"
    )
    complexes = data_dir / complex_name
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
            "--n-splits",
            str(settings["n_splits"]),
            "--random-state",
            str(settings["random_state"]),
        ]
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
            "--negative-repeats",
            str(settings["negative_repeats"]),
            "--seed",
            str(settings["random_state"]),
        ]
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
        ]
    )


def main() -> None:
    args = parse_args()
    if args.workflow == "verify-inputs":
        verify_inputs(args.data_dir)
        return

    data_dir = SMOKE_DATA_DIR if args.profile == "smoke" else args.data_dir
    if args.profile == "full" and not args.skip_input_verification:
        verify_inputs(data_dir)
    output_dir = args.output_dir or PROJECT_ROOT / "runs" / args.profile
    settings = _settings(args.profile)
    workflows = ("oncogene", "ppi", "simulation") if args.workflow == "all" else (args.workflow,)
    for workflow in workflows:
        if workflow == "oncogene":
            run_oncogene(data_dir, output_dir, settings[workflow])
        elif workflow == "ppi":
            run_ppi(data_dir, output_dir, settings[workflow])
        else:
            run_simulation(output_dir, settings[workflow])


if __name__ == "__main__":
    main()
