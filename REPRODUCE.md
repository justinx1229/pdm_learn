# Run the analysis workflows

Use the same pipeline for study data, demos, subsets, and your own datasets.
Choose the input directory, output directory, and optional workflow settings.

## Setup

```bash
uv sync --group dev --locked
uv run pytest
```

For the optional walkthrough notebook, also run
`uv sync --group review --locked`.

## Inputs

Set `--data-dir` to the directory containing the input files. Oncogene and
PPI workflows use these default filenames:

```text
<data-dir>/
├── Gene_Expression_Trimmed.csv
├── Copy_Number_Trimmed.csv
├── shRNA_Trimmed.csv
├── Gene_Mutation_Trimmed.csv
├── CRISPR_Trimmed.csv
├── oncogene.txt                  # Oncogene labels
├── cancer_complexes_clean.xlsx   # PPI complexes
└── clean_biogrid_interactions.csv # PPI candidate pairs
```

Matrices contain a gene identifier column followed by sample columns. Supply
only the files used by the workflow you run. Simulation generates its own data.

For other names or locations, use `--expression`, `--copy-number`, `--shrna`,
`--mutation`, and `--crispr` to override individual matrix paths. Use
`--label-file`, `--biogrid`, and `--complexes` for labels, candidate pairs, and
complexes. Relative file paths are resolved from your working directory.
These options are available on `reproduce.py` and the corresponding individual
scripts.

`--complexes` accepts XLSX or CSV files. When omitted, `reproduce.py` uses
`cancer_complexes_clean.xlsx` if present, otherwise `cancer_complexes.csv`.

To generate oncogene features:

```bash
uv run python scripts/oncogene/make_features.py \
  --data-dir data/DepMap_Trimmed \
  --output-dir runs/oncogene/features
```

These scripts use the DepMap assay recipe. For different assay types or
entity domains, use the generic `load_matrices`, `DatasetSpec`, `build_features`,
and `build_pair_features` APIs described in [README.md](README.md).

## Run

```bash
uv run python scripts/reproduce.py all --data-dir /path/to/data --output-dir runs/my-analysis
```

Replace `all` with `oncogene`, `ppi`, or `simulation` to run one workflow.
Defaults are `data/` for inputs and `runs/` for outputs. Each stage reports its
name, progress, and elapsed time.

Settings come from [`reproduction/workflows.json`](reproduction/workflows.json).
To change trial counts, folds, control counts, or simulation repeats, copy
that file, edit the values, and supply `--config /path/to/settings.json`.
For separate datasets, reuse the same command with a different `--data-dir`
and `--output-dir`.

The repository includes small sample datasets and reduced settings:

```bash
uv run python scripts/reproduce.py all \
  --data-dir tests/fixtures/reproduction/DepMap_Trimmed \
  --label-file tests/fixtures/reproduction/oncogene.txt \
  --biogrid tests/fixtures/reproduction/clean_biogrid_interactions.csv \
  --complexes tests/fixtures/reproduction/cancer_complexes.csv \
  --config tests/fixtures/reproduction/workflows.json \
  --output-dir runs/example
```

Use those reduced settings with your own small dataset as well. They request
three eligible negative PPI pairs and two folds; the default study settings
request 5,000 negative pairs and five folds. Choose counts appropriate for
your data. The sample run exercises the pipeline without reproducing study
results.

## Outputs

Under your chosen output directory:

```text
oncogene/results/oncogene_ranking.csv
oncogene/results/oncogene_ks_pvalues.csv
ppi/results/ranked_PPI_pairs_cancer.csv
ppi/results/ppi_control_benchmarks.csv
simulation/results/simulation_benchmarks.csv
```

Ranking fits one model per trial using known positives and randomly sampled
provisional negatives. Only candidates outside that training set are scored.
`score` is the average of a candidate's predictions, and `score_trials` is
the number contributing to that average. These are ranking scores, not
calibrated probabilities of being an oncogene or an interacting pair.
Short runs can leave candidates with `score_trials = 0` and a blank score;
increase the trial count for more coverage. LOOCV rank recovery requires a
score for every candidate and asks for more trials if coverage is incomplete.

PPI ranking redraws up to 10 eligible positive pairs per complex in every
trial. Set `ppi.ranking_max_pairs_per_complex` in the workflow JSON, or
`--max-pairs-per-complex` when running `scripts/ppi/run_predictions.py`
directly. The separate `ppi.max_pairs_per_complex` setting controls benchmark
sampling. All known positives stay outside the provisional-negative pool,
including positives not selected in a trial. Shared pairs count toward every
complex they belong to and occur only once in training; overlaps can leave
some complexes below the cap. The cap is a starting setting, not a tuned optimum.

Render the oncogene KS heatmap after running the pipeline:

```bash
uv run python scripts/oncogene/make_ks_heatmap.py \
  --ks-file runs/my-analysis/oncogene/results/oncogene_ks_pvalues.csv \
  --output runs/my-analysis/oncogene/results/oncogene_ks_heatmap.png
```

The oncogene workflow uses positive-unlabelled ranking. To generate
`oncogene_method_benchmarks.csv`, run `scripts/oncogene/run_predictions.py`
with a separate `--negative-label-file` containing explicit negative labels.
PPI benchmarks hold out cancer-complex groups, joining complexes that share
curated positive pairs into one group to prevent overlap across folds.
Their sampled negative pairs are exclusion-based controls, not confirmed
noninteractions.
These benchmarks evaluate a classifier on the supplied controls, not the
candidate-ranking procedure or the benefit of its sampling cap. They also
do not establish generalization to unseen proteins.

Open `notebooks/reviewer_walkthrough.ipynb` and set `RUN_DIR` to your output
directory to inspect the results.
