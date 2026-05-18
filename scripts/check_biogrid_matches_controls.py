from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


CONTROL_METADATA_COLUMNS = {"Complex", "pair"}
BIOGRID_METADATA_COLUMNS = {
    "Gene1",
    "Gene2",
    "Interaction_Type",
    "Confidence_Score",
    "pair",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check that positive control PDM rows match the corresponding rows in "
            "the BIogrid interaction PDM matrix."
        )
    )
    parser.add_argument(
        "--controls",
        type=Path,
        default=Path("artifacts/controls/positive_controls.pkl"),
        help="Path to positive_controls.pkl",
    )
    parser.add_argument(
        "--biogrid",
        type=Path,
        default=Path("artifacts/results/clean_biogrid_interactions_pdm_combined.csv"),
        help="Path to the BIogrid PDM csv",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-9,
        help="Absolute tolerance for floating point comparison",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-7,
        help="Relative tolerance for floating point comparison",
    )
    parser.add_argument(
        "--max-report",
        type=int,
        default=10,
        help="Maximum number of mismatches or missing pairs to print",
    )
    return parser.parse_args()


def reverse_pair(pair: str) -> str:
    left, right = pair.split(".", maxsplit=1)
    return f"{right}.{left}"


def load_controls(path: Path) -> pd.DataFrame:
    control_tables = pd.read_pickle(path)
    if isinstance(control_tables, pd.DataFrame):
        controls = control_tables.copy()
    else:
        controls = pd.concat(control_tables, ignore_index=True)
    if "pair" not in controls.columns:
        raise ValueError("Controls file does not contain a 'pair' column.")
    return controls


def get_feature_columns(df: pd.DataFrame, metadata_columns: set[str]) -> list[str]:
    return [column for column in df.columns if column not in metadata_columns]


def build_biogrid_lookup(df: pd.DataFrame) -> dict[str, pd.Series]:
    lookup: dict[str, pd.Series] = {}
    for _, row in df.iterrows():
        pair = row["pair"]
        if pair not in lookup:
            lookup[pair] = row
    return lookup


def main() -> None:
    args = parse_args()

    controls = load_controls(args.controls)
    biogrid = pd.read_csv(args.biogrid)

    control_feature_columns = get_feature_columns(controls, CONTROL_METADATA_COLUMNS)
    biogrid_feature_columns = get_feature_columns(biogrid, BIOGRID_METADATA_COLUMNS)

    if control_feature_columns != biogrid_feature_columns:
        raise ValueError(
            "Feature columns do not align between controls and BIogrid matrix."
        )

    controls = controls.drop_duplicates(subset="pair", keep="first").reset_index(drop=True)
    biogrid_lookup = build_biogrid_lookup(biogrid)

    matched = 0
    missing: list[str] = []
    mismatched: list[tuple[str, float, str]] = []
    reversed_matches = 0

    for _, control_row in controls.iterrows():
        pair = control_row["pair"]
        matched_row = biogrid_lookup.get(pair)
        match_source = "direct"

        if matched_row is None:
            reversed_pair = reverse_pair(pair)
            matched_row = biogrid_lookup.get(reversed_pair)
            if matched_row is not None:
                match_source = "reversed"
                reversed_matches += 1

        if matched_row is None:
            missing.append(pair)
            continue

        control_values = control_row[control_feature_columns].to_numpy(dtype=float)
        biogrid_values = matched_row[biogrid_feature_columns].to_numpy(dtype=float)

        if np.allclose(
            control_values,
            biogrid_values,
            equal_nan=True,
            atol=args.atol,
            rtol=args.rtol,
        ):
            matched += 1
            continue

        diff = np.abs(control_values - biogrid_values)
        max_abs_diff = float(np.nanmax(diff))
        mismatched.append((pair, max_abs_diff, match_source))

    total = len(controls)
    print(f"Controls checked: {total}")
    print(f"Exact/tolerant matches: {matched}")
    print(f"Missing from BIogrid PDM: {len(missing)}")
    print(f"Mismatched rows: {len(mismatched)}")
    print(f"Matched via reversed pair key: {reversed_matches}")

    if missing:
        print("\nMissing pairs:")
        for pair in missing[: args.max_report]:
            print(f"  {pair}")

    if mismatched:
        print("\nMismatched pairs:")
        for pair, max_abs_diff, match_source in mismatched[: args.max_report]:
            print(f"  {pair} ({match_source}, max abs diff={max_abs_diff:.3e})")


if __name__ == "__main__":
    main()
