from __future__ import annotations

import argparse
import ast
from itertools import combinations
from pathlib import Path

import pandas as pd


DEFAULT_COMPLEXES_PATH = Path(
    "notebooks/PPI/Wanyi's scripts and data/cancer_complexes_clean.xlsx"
)
DEFAULT_BIOGRID_PATH = Path("artifacts/results/clean_biogrid_interactions_pdm.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check whether every pair of genes within each cancer complex appears "
            "in the clean Biogrid interaction table."
        )
    )
    parser.add_argument(
        "--complexes",
        type=Path,
        default=DEFAULT_COMPLEXES_PATH,
        help="Path to cancer_complexes_clean.xlsx",
    )
    parser.add_argument(
        "--biogrid",
        type=Path,
        default=DEFAULT_BIOGRID_PATH,
        help="Path to clean_biogrid_interactions_pdm.csv",
    )
    parser.add_argument(
        "--complex-name-column",
        default="Complex Name",
        help="Column name containing the complex name",
    )
    parser.add_argument(
        "--genes-column",
        default="Representative Genes (Core Members)",
        help="Column name containing the gene list for each complex",
    )
    parser.add_argument(
        "--max-report",
        type=int,
        default=25,
        help="Maximum number of missing complexes to print in the summary",
    )
    return parser.parse_args()


def parse_gene_list(value: object) -> list[str]:
    if isinstance(value, list):
        genes = value
    elif isinstance(value, tuple):
        genes = list(value)
    elif value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    elif isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            parsed = [part.strip() for part in value.split(",")]
        if isinstance(parsed, (list, tuple)):
            genes = list(parsed)
        else:
            genes = [str(parsed)]
    else:
        genes = [str(value)]

    cleaned = []
    for gene in genes:
        gene_text = str(gene).strip()
        if gene_text:
            cleaned.append(gene_text)
    return cleaned


def canonical_pair(gene_a: str, gene_b: str) -> str:
    left, right = sorted((gene_a.strip(), gene_b.strip()))
    return f"{left}.{right}"


def build_biogrid_pair_lookup(df: pd.DataFrame) -> set[str]:
    if "pair" in df.columns:
        pairs = df["pair"].dropna().astype(str)
        return {canonical_pair(*pair.split(".", maxsplit=1)) for pair in pairs if "." in pair}

    if {"Gene1", "Gene2"}.issubset(df.columns):
        return {
            canonical_pair(str(row["Gene1"]), str(row["Gene2"]))
            for _, row in df.iterrows()
            if pd.notna(row["Gene1"]) and pd.notna(row["Gene2"])
        }

    raise ValueError(
        "Biogrid file must contain either a 'pair' column or both 'Gene1' and 'Gene2' columns."
    )


def main() -> None:
    args = parse_args()

    complexes = pd.read_excel(args.complexes, index_col=0)
    biogrid = pd.read_csv(args.biogrid)

    if args.complex_name_column not in complexes.columns:
        raise ValueError(
            f"Complex name column '{args.complex_name_column}' was not found in the workbook."
        )
    if args.genes_column not in complexes.columns:
        raise ValueError(
            f"Genes column '{args.genes_column}' was not found in the workbook."
        )

    biogrid_lookup = build_biogrid_pair_lookup(biogrid)

    missing_complexes: list[dict[str, object]] = []
    all_missing_pairs: list[str] = []
    complete_count = 0
    total_pairs_checked = 0

    for _, row in complexes.iterrows():
        complex_name = str(row[args.complex_name_column]).strip()
        genes = parse_gene_list(row[args.genes_column])
        unique_genes = list(dict.fromkeys(genes))

        if len(unique_genes) < 2:
            continue

        missing_pairs = []
        for gene_a, gene_b in combinations(unique_genes, 2):
            total_pairs_checked += 1
            pair_key = canonical_pair(gene_a, gene_b)
            if pair_key not in biogrid_lookup:
                missing_pairs.append(pair_key)

        if missing_pairs:
            all_missing_pairs.extend(missing_pairs)
            missing_complexes.append(
                {
                    "complex_name": complex_name,
                    "gene_count": len(unique_genes),
                    "total_pairs": len(unique_genes) * (len(unique_genes) - 1) // 2,
                    "missing_pairs": missing_pairs,
                }
            )
        else:
            complete_count += 1

    print(f"Complexes checked: {len(complexes)}")
    print(f"Complexes with all pairs found: {complete_count}")
    print(f"Complexes with at least one missing pair: {len(missing_complexes)}")
    print(f"Total gene pairs checked: {total_pairs_checked}")
    print(f"Total missing pairs: {len(all_missing_pairs)}")

    print("\nAll missing pairs:")
    print(all_missing_pairs)

    if not missing_complexes:
        print("\nNo missing complexes were found.")
        return

    print("\nMissing complexes:")
    for item in missing_complexes[: args.max_report]:
        missing_pairs = item["missing_pairs"]
        print(
            f"  {item['complex_name']} "
            f"({len(missing_pairs)} missing of {item['total_pairs']} pairs)"
        )
        for pair in missing_pairs[:5]:
            print(f"    - {pair}")
        if len(missing_pairs) > 5:
            print(f"    - ... and {len(missing_pairs) - 5} more")


if __name__ == "__main__":
    main()
