# PDM Learn

PDM Learn is an installable Python library for building and evaluating pairwise
density-map (PDM) features from aligned entity-by-sample matrices. It includes
generic loading, alignment, density/statistic feature construction, labels,
ranking, and simulation utilities. DepMap/CCLE oncogene and protein-protein
interaction (PPI) analyses are maintained recipes built on that core.

The library lives in `src/pdm_learn/`. Workflow scripts under `scripts/` are
kept as reproducible examples; notebooks are optional walkthroughs.

## Architecture

The reusable package has a deliberately small, domain-neutral center:

- `datasets.py`: `DatasetSpec`, CSV loading, matrix preparation, and alignment.
- `features.py`: row-level PDM, Pearson, Spearman, MI, and bicor feature builders.
- `pair_features.py`: the equivalent builders for tables of ordered entity pairs.
- `labels.py`: label-file loading, row splitting, and class balancing.
- `modeling.py`: candidate ranking, cross-validation, scoring, and feature selection.
- `simulation.py`: general simulation utilities.

`presets.py` holds optional DepMap defaults. `workflows/oncogene.py` and
`workflows/ppi.py` are thin recipes that select those defaults and call the
generic APIs. Historical `pdm_learn.oncogene` and `pdm_learn.ppi` imports remain
available for compatibility, but new code should use the generic modules or
`pdm_learn.workflows`.

## Installation / Setup

`uv` is the recommended environment manager because it uses `uv.lock` as the
dependency lockfile and installs `pdm_learn` in editable mode.

```bash
uv sync
uv run python -m unittest discover -s tests
uv build
uv run jupyter lab
```

To use the library from another project, build a wheel with `uv build` and
install the generated file from `dist/`, or install this checkout directly with
`pip install .`.

If you prefer conda, use it only to create Python and install `uv`, then let
`uv` install the locked project dependencies:

```bash
conda env create -f environment.yml
conda activate pdm_learn_env
uv sync
uv run python -m unittest discover -s tests
uv run jupyter lab
```

The test suite covers the public feature API and the shared preprocessing,
simulation, modeling, oncogene, and PPI helpers.

Run all workflow commands from the repository root. The examples below prefix
script calls with `PYTHONPATH=src` so Python can find `pdm_learn` even if the
editable install has not been activated in the current shell.

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
├── scripts/                     # workflow-specific reproduction and validation commands
│   ├── data/
│   ├── oncogene/
│   ├── ppi/
│   ├── simulation/
│   ├── validation/
│   └── figures/
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
place the required files under `data/` with the paths below, or pass explicit
file paths to the scripts. Users do not need to rename their own local files if
they provide the corresponding command-line flag.

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
derives the shared analysis-ready files under `data/DepMap_Trimmed/`. The
intermediate files preserve assay-specific cleaned versions of the raw exports.
The required shared trimmed files are the harmonized matrices used by the main
oncogene and PPI scripts after intersecting genes/cell lines and applying the
shared preprocessing rules.

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

`data/oncogene.txt` is a user-provided one-column list of known oncogenes used
to label positives for the oncogene-ranking step. If the file has another name
or location, pass it with `--oncogene` when running
`scripts/oncogene/run_predictions.py`.

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

If your BioGRID pair table has a different filename or location, pass it with
`--biogrid /path/to/your_pairs.csv` when running `scripts/ppi/make_features.py`.
The table must contain either `Gene_A`/`Gene_B` or `Gene1`/`Gene2` columns.

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
PYTHONPATH=src uv run python scripts/data/prepare_trimmed_inputs.py
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
PYTHONPATH=src uv run python scripts/data/prepare_trimmed_inputs.py --from-intermediate
```

To use files with non-default names, pass the paths directly, for example:

```bash
PYTHONPATH=src uv run python scripts/data/prepare_trimmed_inputs.py \
  --expression /path/to/expression.csv \
  --copy-number /path/to/copy_number.csv \
  --shrna /path/to/shrna.csv \
  --mutation /path/to/mutation.csv \
  --crispr /path/to/crispr.csv
```

### 2. Oncogene Analysis

Build feature tables, run rankings/benchmarks, and export heatmap-ready
statistics:

```bash
PYTHONPATH=src uv run python scripts/oncogene/make_features.py
PYTHONPATH=src uv run python scripts/oncogene/run_predictions.py
```

`scripts/oncogene/make_features.py` creates PDM, Pearson, Spearman, mutual-information,
and bicor feature tables for each gene. `scripts/oncogene/run_predictions.py` uses
`oncogene.txt` to split known oncogenes from candidates, then writes ranking and
benchmark summaries. To use already-trimmed files with other names, pass
`--expression`, `--copy-number`, `--shrna`, `--mutation`, and `--crispr` to
`scripts/oncogene/make_features.py`; pass `--oncogene` to use a non-default oncogene
list.

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

### 3. Generic Matrix Analysis

For any aligned entity-by-sample CSVs, use the library API. An identifier can
have any column name (and does not have to be first); declare it in
`DatasetSpec`. Dataset names and feature pairs are arbitrary.

```python
from pdm_learn import (
    DatasetSpec,
    FeaturePair,
    build_features,
    load_label_set,
    load_matrices,
    rank_candidates,
)

specs = [
    DatasetSpec("expression", id_column="gene"),
    DatasetSpec("mutation", kind="binary", normalize=None, id_column="symbol"),
]
datasets = load_matrices({
    "expression": "expression.csv",
    "mutation": "mutation.csv",
}, specs=specs)
features = build_features(
    datasets,
    specs=specs,
    pairs=[FeaturePair("mutation", "expression")],
    methods=["pdm", "pearson", "spearman", "mi", "bicor"],
)
labels = load_label_set("positive_entity_labels.csv")
ranking = rank_candidates(features["pdm"], labels)
```

`features` contains one table per requested method. Use `FeatureConfig` for
shared options such as density bins, log smoothing, missing-row behavior, and
the union/intersection entity universe. For observations with two different
row IDs—such as protein interactions—use `build_pair_features` and
`PairFeatureConfig` instead.

### 4. Simulation Analysis

Generate simulated associations/features and benchmark feature representations:

```bash
PYTHONPATH=src uv run python scripts/simulation/make_features.py
PYTHONPATH=src uv run python scripts/simulation/run_benchmarks.py
```

The first command creates simulated positive/negative associations and their
feature representations. The second command checks whether the modeling
pipeline recovers the simulated positives from the negatives.

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

### 5. PPI Analysis

Generate PPI controls, build BioGRID PDM features, and rank candidate
cancer-complex pairs:

```bash
PYTHONPATH=src uv run python scripts/ppi/make_controls.py
PYTHONPATH=src uv run python scripts/ppi/make_features.py
PYTHONPATH=src uv run python scripts/ppi/run_predictions.py
```

`scripts/ppi/make_controls.py` samples positive cancer-complex controls and negative
control pairs, then builds PDM features for those controls. The default
`--negative-pairs 5000` gives a larger control set but can be slow on laptops;
for a faster smoke test, use a smaller value such as `--negative-pairs 1000`.
`scripts/ppi/make_features.py` builds PDM features for BioGRID pairs. Use `--biogrid`
if your pair list is not named `data/clean_biogrid_interactions.csv`, and use
the same trimmed-file override flags listed in the oncogene step if your shared
DepMap files have custom paths.

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
PYTHONPATH=src uv run python scripts/validation/check_biogrid_matches_controls.py
PYTHONPATH=src uv run python scripts/validation/check_complex_pairs_in_biogrid.py
```

These utilities are sanity checks. `scripts/validation/check_biogrid_matches_controls.py` reports
how many sampled positive-control pairs also appear in the BioGRID feature
table and whether any control rows are entirely missing feature values. Missing
pairs are useful diagnostics about BioGRID coverage; a short printed missing
list does not necessarily mean the previous scripts failed. The second utility
reports which curated cancer-complex pairs are present in BioGRID.

### 6. Figures

After the analysis scripts have written their result tables, generate scripted
figure files with:

```bash
PYTHONPATH=src uv run python scripts/figures/make_figures.py
```

`oncogene_ks_heatmaps.png` summarizes feature-wise KS enrichment across the
oncogene feature blocks. `simulation_positive_heatmap.png` shows the density
map for a simulated positive association.

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

## Library API

The package root re-exports the primary generic API:

- `DatasetSpec`, `FeaturePair`, and `FeatureConfig` describe input matrices and feature blocks.
- `load_matrices`, `prepare_matrix`, and `align_matrices` keep file I/O and matrix alignment separate from computation.
- `build_features` builds row-level PDM and scalar association features from in-memory tables.
- `build_pair_features` handles ordered pair tables without PPI-specific assumptions.
- `load_label_set`, `split_labeled_rows`, `rank_candidates`/`rank_unlabeled`, and `evaluate_labeled_features` provide generic positive-label workflows.

Use `from pdm_learn.workflows import oncogene, ppi` when the maintained DepMap
recipes are useful. The workflow modules provide `load_inputs`,
`build_features`, and `rank_candidates` conveniences while keeping file
layouts, assay names, and control sampling outside the core API.

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

Notebook setup cells should define:

```text
PROJECT_ROOT
DATA_DIR
ARTIFACTS_DIR
FIGURES_DIR
```

Use those variables for every file path in reviewer-facing notebooks.
