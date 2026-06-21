# PDM Learn

This repository contains the code used to build and evaluate pairwise density map
(PDM) features for DepMap/CCLE molecular profiles, protein-protein interaction
(PPI) controls, simulated associations, and oncogene-ranking experiments.

The reviewer-facing code is organized around a small reusable Python package in
`src/pdm_learn/`. Command-line scripts under `scripts/` are the canonical
reproduction path; notebooks are companion walkthroughs for inspection and
exploration.

## Reproducibility Quick Start

`uv` is the recommended environment manager because it uses `uv.lock` as the
dependency lockfile and installs `pdm_learn` in editable mode.

```bash
uv sync
uv run python -m unittest discover -s tests
uv run jupyter lab
```

If you prefer conda, use it only to create Python and install `uv`, then let
`uv` install the locked project dependencies:

```bash
conda env create -f environment.yml
conda activate pdm_learn_env
uv sync
uv run python -m unittest discover -s tests
uv run jupyter lab
```

The test suite currently covers the shared preprocessing, simulation, modeling,
and PPI helpers:

```text
Ran 25 tests
OK
```

## Publication-Ready Repository Layout

Keep the repository organized as follows for submission:

```text
.
├── README.md                    # reviewer entry point and reproduction guide
├── pyproject.toml               # package metadata and dependency ranges
├── uv.lock                      # locked Python dependency graph
├── environment.yml              # optional conda bootstrap for Python + uv
├── src/pdm_learn/               # reusable analysis code
├── tests/                       # regression tests for shared code
├── notebooks/
│   ├── oncogene/                # oncogene feature generation and evaluation
│   ├── PPI/                     # PPI controls, feature generation, ranking
│   └── simulation/              # simulated associations and benchmarking
├── scripts/                     # canonical reproduction and validation commands
├── archive/                     # historical/provenance notebooks, not reviewer workflow
├── data/                        # local raw/intermediate data, not committed
├── artifacts/                   # generated controls/results, not committed
└── figures/                     # exported publication figures
```

Reviewer-facing notebooks should live directly under `notebooks/oncogene/`,
`notebooks/PPI/`, and `notebooks/simulation/`. Historical notebooks with legacy
absolute paths live under `archive/notebooks/` and are provenance only; they are
not part of the main reproduction path.

## Data Manifest

Large data files are intentionally ignored by git. To reproduce the analyses,
place the required files under `data/` with the paths below.

### Oncogene / DepMap Inputs

The trimming script can start from raw CCLE/DepMap exports. By default it looks
for these files under `data/raw/`, but each path can be overridden with a command
line flag:

```text
data/raw/CCLE_gene_expression.csv
data/raw/CCLE_gene_cn.csv
data/raw/shRNA_Broad.csv
data/raw/CCLE_gene_mutation.csv
data/raw/Avana_gene_effect_20Q3.csv
```

It writes intermediate `trimmed_Wei` files under `data/DepMap_data/` and then
derives the shared analysis-ready files under `data/DepMap_Trimmed/`. Oncogene
and PPI analysis both require the shared `data/DepMap_Trimmed/*_Trimmed.csv`
files.

Intermediate files:

```text
data/DepMap_data/CCLE_gene_expression_trimmed_Wei.csv
data/DepMap_data/CCLE_gene_cn_trimmed_Wei.csv
data/DepMap_data/shRNA_Broad_Trimmed_Wei.csv
data/DepMap_data/CCLE_gene_mutation_trimmed_Wei.csv
data/DepMap_data/Avana_gene_effect_20Q3_Trimmed_Wei.csv
```

Required shared trimmed files:

```text
data/DepMap_Trimmed/Gene_Expression_Trimmed.csv
data/DepMap_Trimmed/Copy_Number_Trimmed.csv
data/DepMap_Trimmed/shRNA_Trimmed.csv
data/DepMap_Trimmed/Gene_Mutation_Trimmed.csv
data/DepMap_Trimmed/CRISPR_Trimmed.csv
data/oncogene.txt
```

Optional related inputs present in the local workflow:

```text
data/DepMap_data/GeCKO_gene_effect_19Q1_Trimmed_Wei.csv
data/DepMap_data/Sanger_gene_effect_Trimmed_Wei.csv
data/DepMap_data/CCLE_mutation_Wei.csv
```

### PPI Inputs

The shared `data/DepMap_Trimmed/*_Trimmed.csv` derivation keeps the 15,305 genes
shared across expression, copy-number, shRNA, and CRISPR; row-centers
expression/shRNA/CRISPR; doubles and snaps copy number to `0, 1, 2, 3, 4, 6, 8`;
and converts the long mutation table into a binary gene-by-cell-line matrix.

Generated PPI-ready inputs:

```text
data/DepMap_Trimmed/Gene_Expression_Trimmed.csv
data/DepMap_Trimmed/Copy_Number_Trimmed.csv
data/DepMap_Trimmed/shRNA_Trimmed.csv
data/DepMap_Trimmed/Gene_Mutation_Trimmed.csv
data/DepMap_Trimmed/CRISPR_Trimmed.csv
data/clean_biogrid_interactions.csv
```

PPI controls and ranking also use the curated cancer-complex workbook at:

```text
data/cancer_complexes_clean.xlsx
```

### Generated Data And Artifacts

These are outputs, not source files. Reviewers can regenerate them by running
the scripts in the order below.

```text
data/Trimmed data/*.csv
data/simulated/*.csv
data/PPI_Pairs/*.csv
artifacts/controls/*.pkl
artifacts/results/*.csv
figures/*
```

If any generated tables are too expensive to regenerate during review, deposit
them with the raw data archive and document the archive DOI or download link in
the final manuscript README.

## Reproduction Workflow

Run scripts from the repository root. The scripts write generated tables to
`data/`, `artifacts/`, and `figures/`.

### 1. Prepare Shared Trimmed Data

Generate the shared `data/DepMap_Trimmed/*_Trimmed.csv` inputs. These are the
required inputs for both oncogene and PPI analyses.

```bash
uv run python scripts/make_trimmed_inputs.py
```

Main outputs:

```text
data/DepMap_Trimmed/Gene_Expression_Trimmed.csv
data/DepMap_Trimmed/Copy_Number_Trimmed.csv
data/DepMap_Trimmed/shRNA_Trimmed.csv
data/DepMap_Trimmed/Gene_Mutation_Trimmed.csv
data/DepMap_Trimmed/CRISPR_Trimmed.csv
```

If you already have the intermediate `data/DepMap_data/*_trimmed_Wei.csv` files
and only need to recreate the required shared trimmed files, run:

```bash
uv run python scripts/make_trimmed_inputs.py --from-intermediate
```

### 2. Oncogene Analysis

Build feature tables, run rankings/benchmarks, and export heatmap-ready
statistics:

```bash
uv run python scripts/make_oncogene_features.py
uv run python scripts/run_oncogene_predictions.py
```

Main outputs:

```text
data/Trimmed data/dataset_trimmed_v3.csv
data/Trimmed data/pearson.csv
data/Trimmed data/spearman.csv
data/Trimmed data/mi.csv
data/Trimmed data/bicor.csv
artifacts/results/oncogene_ranking.csv
artifacts/results/oncogene_method_benchmarks.csv
artifacts/results/oncogene_feature_sweep.csv
artifacts/results/oncogene_ks_pvalues.csv
```

### 3. Simulation Analysis

Generate simulated associations/features and benchmark feature representations:

```bash
uv run python scripts/make_simulation_features.py
uv run python scripts/run_simulation_benchmarks.py
```

Main outputs:

```text
data/simulated/positive.csv
data/simulated/positive_heatmap.csv
data/simulated/negative_heatmap.csv
data/simulated/positive_pearson.csv
data/simulated/negative_pearson.csv
data/simulated/positive_spearman.csv
data/simulated/negative_spearman.csv
data/simulated/positive_mi.csv
data/simulated/negative_mi.csv
data/simulated/positive_bicor.csv
data/simulated/negative_bicor.csv
artifacts/results/simulation_benchmarks.csv
```

### 4. PPI Analysis

Generate PPI controls, build BioGRID PDM features, and rank candidate
cancer-complex pairs:

```bash
uv run python scripts/make_ppi_controls.py
uv run python scripts/make_ppi_features.py
uv run python scripts/run_ppi_predictions.py
```

Main outputs:

```text
data/PPI_Pairs/*.csv
artifacts/controls/positive_controls.pkl
artifacts/controls/negative_controls.pkl
artifacts/results/clean_biogrid_interactions_pdm.csv
artifacts/results/ranked_PPI_pairs_cancer.csv
artifacts/results/ppi_control_benchmarks.csv
```

After generating PPI controls/results, run the validation utilities:

```bash
uv run python scripts/check_biogrid_matches_controls.py
uv run python scripts/check_complex_pairs_in_biogrid.py
```

### 5. Figures

After the analysis scripts have written their result tables, generate scripted
figure files with:

```bash
uv run python scripts/make_figures.py
```

Main outputs:

```text
figures/oncogene_ks_heatmaps.png
figures/simulation_positive_heatmap.png
```

## Notebook Policy

Notebooks are retained as readable companions, not as the canonical execution
engine. Use them to inspect intermediate tables, explore plots interactively, or
explain the analysis flow. If notebook logic becomes required for reproduction,
move that logic into `src/pdm_learn/` or one of the command-line scripts first.

Launch companion notebooks with:

```bash
uv run jupyter lab
```

## Shared Python Package

The reusable implementation lives in `src/pdm_learn/`:

- `preprocessing.py`: trimming, missing-value handling, density-map generation
- `simulation.py`: simulated pair generation and simulated feature tables
- `modeling.py`: rank recovery, cross-validation, PR/ROC evaluation, plotting
- `oncogene.py`: oncogene-specific input trimming, data loading, feature tables, ranking
- `ppi.py`: PPI control generation, BioGRID feature tables, PPI ranking

Prefer adding new analysis logic to this package and calling it from scripts.
That keeps workflows testable and easier for reviewers to audit.

## Generated Files And Version Control

The repository ignores local data, large generated artifacts, virtual
environments, cache directories, and binary outputs. This keeps the source repo
reviewable while avoiding accidental commits of large or licensed datasets.

Before submission:

1. Keep `uv.lock`, `pyproject.toml`, `environment.yml`, `src/`, `tests/`,
   `scripts/`, reviewer-facing notebooks, and this README under version control.
2. Do not commit `.venv/`, `.uv-cache/`, `.DS_Store`, `__pycache__/`,
   generated pickles, generated CSVs, or large raw datasets unless the journal
   explicitly requires them in the repository.
3. Strip or clear exploratory notebook outputs if they make diffs noisy. Keep
   final figures in `figures/` or in the external artifact archive.
4. Document the raw-data source, version/date, and any access restrictions in
   the manuscript or data-availability statement.

## Notes For Maintainers

If notebooks pick up duplicated helper functions or machine-specific paths
again, rerun:

```bash
uv run python scripts/refactor_notebooks.py
```

The setup cells define:

```text
PROJECT_ROOT
DATA_DIR
ARTIFACTS_DIR
FIGURES_DIR
```

Use those variables for every file path in reviewer-facing notebooks.
