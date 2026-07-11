from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


CONTROL_METADATA_COLUMNS = {"Complex", "pair"}
BIOGRID_METADATA_COLUMNS = {"Gene1", "Gene2", "Interaction_Type", "Confidence_Score", "pair"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check that positive control pairs exist in the Biogrid table and that "
            "the control rows are not all NaNs."
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
        default=Path("artifacts/results/clean_biogrid_interactions_pdm.csv"),
        help="Path to the Biogrid PDM csv",
    )
    parser.add_argument(
        "--max-report",
        type=int,
        default=10,
        help="Maximum number of missing or invalid pairs to print",
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


def row_is_all_nan(row: pd.Series, feature_columns: list[str]) -> bool:
    return row[feature_columns].isna().all()


def main() -> None:
    args = parse_args()

    controls = load_controls(args.controls)
    biogrid = pd.read_csv(args.biogrid)

    control_feature_columns = get_feature_columns(controls, CONTROL_METADATA_COLUMNS)

    controls = controls.drop_duplicates(subset="pair", keep="first").reset_index(drop=True)
    biogrid_lookup = build_biogrid_lookup(biogrid)

    missing: list[str] = []
    invalid_nan_rows: list[str] = []
    reversed_matches = 0

    for _, control_row in controls.iterrows():
        pair = control_row["pair"]
        if pd.isna(pair) or str(pair).strip() == "":
            invalid_nan_rows.append("<missing pair>")
            continue

        pair = str(pair)
        if "." not in pair:
            invalid_nan_rows.append(pair)
            continue

        if row_is_all_nan(control_row, control_feature_columns):
            invalid_nan_rows.append(pair)
            continue

        matched_row = biogrid_lookup.get(pair)

        if matched_row is None:
            reversed_pair = reverse_pair(pair)
            matched_row = biogrid_lookup.get(reversed_pair)
            if matched_row is not None:
                reversed_matches += 1

        if matched_row is None:
            missing.append(pair)

    total = len(controls)
    print(f"Controls checked: {total}")
    print(f"Missing from Biogrid PDM: {len(missing)}")
    print(f"Invalid/NaN control rows: {len(invalid_nan_rows)}")
    print(f"Matched via reversed pair key: {reversed_matches}")

    if missing:
        print("\nMissing pairs:")
        for pair in missing[: args.max_report]:
            print(f"  {pair}")

    if invalid_nan_rows:
        print("\nInvalid/NaN control pairs:")
        for pair in invalid_nan_rows[: args.max_report]:
            print(f"  {pair}")


if __name__ == "__main__":
    main()
