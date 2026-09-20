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

Oncogene and PPI workflows use this layout under `--data-dir`:

```text
<data-dir>/
├── DepMap_Trimmed/
│   ├── Gene_Expression_Trimmed.csv
│   ├── Copy_Number_Trimmed.csv
│   ├── shRNA_Trimmed.csv
│   ├── Gene_Mutation_Trimmed.csv
│   └── CRISPR_Trimmed.csv
├── oncogene.txt                  # Oncogene labels
├── cancer_complexes_clean.xlsx   # PPI complexes; cancer_complexes.csv also works
└── clean_biogrid_interactions.csv # PPI candidate pairs
```

Matrices contain a gene identifier column followed by sample columns. Supply
only the files used by the workflow you run. If both complex tables are
present, PPI uses the XLSX workbook. Simulation generates its own data.

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
  --data-dir tests/fixtures/reproduction \
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

Render the oncogene KS heatmap after running the pipeline:

```bash
uv run python scripts/oncogene/make_ks_heatmap.py \
  --ks-file runs/my-analysis/oncogene/results/oncogene_ks_pvalues.csv \
  --output runs/my-analysis/oncogene/results/oncogene_ks_heatmap.png
```

The oncogene workflow uses positive-unlabelled ranking. To generate
`oncogene_method_benchmarks.csv`, run `scripts/oncogene/run_predictions.py`
with a separate `--negative-label-file` containing explicit negative labels.
PPI benchmarks hold out cancer-complex groups; their sampled negative pairs
are exclusion-based controls, not confirmed noninteractions.

Open `notebooks/reviewer_walkthrough.ipynb` and set `RUN_DIR` to your output
directory to inspect the results.
