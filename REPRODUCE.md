# Reproduce the reviewer workflows

The command-line workflow is the canonical path.  It uses the locked Python
environment, records every output below `runs/`, and fixes the models and
random seeds in [`reproduction/workflows.json`](reproduction/workflows.json).
The accompanying notebook is explanatory only.

## Setup

```bash
uv sync --group dev --locked
uv run pytest
```

To execute the optional walkthrough notebook, also run:

```bash
uv sync --group review --locked
```

The full oncogene and PPI workflows need the frozen study-data bundle.  Place
its contents under a local directory with this layout:

```text
<data-dir>/
├── DepMap_Trimmed/
│   ├── Gene_Expression_Trimmed.csv
│   ├── Copy_Number_Trimmed.csv
│   ├── shRNA_Trimmed.csv
│   ├── Gene_Mutation_Trimmed.csv
│   └── CRISPR_Trimmed.csv
├── oncogene.txt
├── cancer_complexes_clean.xlsx
└── clean_biogrid_interactions.csv
```

The exact SHA-256 checksums are in
[`reproduction/input_manifest.json`](reproduction/input_manifest.json).  The
study-data deposit is intentionally separate from the code repository because
the local bundle is large.  Add its DOI or stable URL to that manifest before
submission.

```bash
uv run python scripts/reproduce.py verify-inputs --data-dir /path/to/study-data
```

## Full reproduction

```bash
uv run python scripts/reproduce.py oncogene --data-dir /path/to/study-data
uv run python scripts/reproduce.py ppi --data-dir /path/to/study-data
uv run python scripts/reproduce.py simulation --data-dir /path/to/study-data
```

Use `all` to run the three workflows in sequence.  Outputs are deterministic
for the packaged dependency lock and configuration:

```text
runs/full/oncogene/results/oncogene_ranking.csv
runs/full/oncogene/results/oncogene_ks_pvalues.csv
runs/full/ppi/results/ranked_PPI_pairs_cancer.csv
runs/full/ppi/results/ppi_control_benchmarks.csv
runs/full/simulation/results/simulation_benchmarks.csv
```

The oncogene workflow is positive-unlabelled ranking. It deliberately does
not report ROC AUC or average precision unless you provide a separate,
explicit negative-label file to `scripts/oncogene/run_predictions.py`.
PPI control results use held-out cancer-complex groups. Negative controls are
sampled from the supplied BioGRID candidate-pair table after removing pairs in
the curated complex catalogue; they are not claims of experimentally confirmed
noninteraction.

## Fast verification

The repository includes small synthetic matrices for a complete smoke run.
This does not reproduce the manuscript numbers; it verifies that every stage
works together without the study-data bundle.

```bash
uv run python scripts/reproduce.py all --profile smoke
```

The smoke run writes to `runs/smoke/` and is suitable for continuous
integration.  Open `notebooks/reviewer_walkthrough.ipynb` after either run to
inspect the generated tables.
