# PDM Learn

This repository contains the code used to build and evaluate pairwise density map
(PDM) features for DepMap/CCLE molecular profiles, protein-protein interaction
(PPI) controls, simulated associations, and oncogene-ranking experiments.

The reviewer-facing code is organized around a small reusable Python package in
`src/pdm_learn/`. The notebooks call that package to reproduce the analyses and
write generated tables to `data/`, `artifacts/`, and `figures/`.

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

The test suite currently covers the shared preprocessing, simulation, and
modeling helpers:

```text
Ran 12 tests
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
├── scripts/                     # notebook refactoring and validation utilities
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

Required by `pdm_learn.oncogene.load_oncogene_inputs()` and the oncogene
notebooks:

```text
data/DepMap_data/CCLE_gene_expression_trimmed_Wei.csv
data/DepMap_data/CCLE_gene_cn_trimmed_Wei.csv
data/DepMap_data/shRNA_Broad_Trimmed_Wei.csv
data/DepMap_data/CCLE_gene_mutation_trimmed_Wei.csv
data/DepMap_data/Avana_gene_effect_20Q3_Trimmed_Wei.csv
data/oncogene.txt
```

Optional related inputs present in the local workflow:

```text
data/DepMap_data/GeCKO_gene_effect_19Q1_Trimmed_Wei.csv
data/DepMap_data/Sanger_gene_effect_Trimmed_Wei.csv
data/DepMap_data/CCLE_mutation_Wei.csv
```

### PPI Inputs

Required by the PPI notebooks and validation scripts:

```text
data/DepMap_Trimmed/Gene_Expression_Trimmed.csv
data/DepMap_Trimmed/Copy_Number_Trimmed.csv
data/DepMap_Trimmed/shRNA_Trimmed.csv
data/DepMap_Trimmed/Gene_Mutation_Trimmed.csv
data/DepMap_Trimmed/CRISPR_Trimmed.csv
data/clean_biogrid_interactions.csv
data/fcg.txt
data/humanComplexes.txt
data/positive_pairs.csv
```

The cancer-complex ranking notebook also expects a cancer-complex workbook at:

```text
data/cancer_complexes_clean.xlsx
```

### Generated Data And Artifacts

These are outputs, not source files. Reviewers can regenerate them by running
the notebooks in the order below.

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

Run notebooks from the repository root with `uv run jupyter lab` so the shared
setup cells can locate `src/`, `data/`, `artifacts/`, and `figures/`.

### 1. Oncogene Analysis

Run:

```text
notebooks/oncogene/data conversion.ipynb
notebooks/oncogene/Machine Learning.ipynb
```

Main outputs:

```text
data/Trimmed data/dataset_trimmed_v3.csv
data/Trimmed data/pearson.csv
data/Trimmed data/spearman.csv
data/Trimmed data/mi.csv
data/Trimmed data/bicor.csv
artifacts/results/oncogene_ranking.csv
```

### 2. Simulation Analysis

Run:

```text
notebooks/simulation/Associations.ipynb
notebooks/simulation/data conversion Sim.ipynb
notebooks/simulation/Machine Learning Sim.ipynb
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
```

### 3. PPI Analysis

Run:

```text
notebooks/PPI/Pos_Neg_control.ipynb
notebooks/PPI/data conversion PPI.ipynb
notebooks/PPI/Density Map PPI.ipynb
notebooks/PPI/MI_featureSignificance.ipynb
notebooks/PPI/Bicor_featureSignificance.ipynb
notebooks/PPI/Machine Learning PPI.ipynb
notebooks/PPI/rank_PPI_pairs_cancer.ipynb
```

Main outputs:

```text
data/PPI_Pairs/*.csv
artifacts/controls/positive_controls.pkl
artifacts/controls/negative_controls.pkl
artifacts/results/clean_biogrid_interactions_pdm.csv
artifacts/results/clean_biogrid_interactions_pdm_combined.csv
artifacts/results/ranked_PPI_pairs_cancer.csv
```

After generating PPI controls/results, run the validation utilities:

```bash
uv run python scripts/check_biogrid_matches_controls.py
uv run python scripts/check_complex_pairs_in_biogrid.py
```

## Shared Python Package

The reusable implementation lives in `src/pdm_learn/`:

- `preprocessing.py`: trimming, missing-value handling, density-map generation
- `simulation.py`: simulated pair generation and simulated feature tables
- `modeling.py`: rank recovery, cross-validation, PR/ROC evaluation, plotting
- `oncogene.py`: oncogene-specific data loading, feature tables, ranking

Prefer adding new analysis logic to this package and calling it from notebooks.
That keeps notebooks short, testable, and easier for reviewers to audit.

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
