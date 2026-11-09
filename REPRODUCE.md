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

## Oncogene predictions, benchmarks, and figures

The four oncogene entry points live in `scripts/oncogene/`. Their default
feature and result directories remain `runs/oncogene/features/` and
`runs/oncogene/results/`.

```bash
uv run python scripts/oncogene/make_features.py --data-dir data/DepMap_Trimmed
uv run python scripts/oncogene/run_predictions.py --label-file data/oncogene.txt
uv run python scripts/oncogene/run_benchmarks.py --label-file data/oncogene.txt
uv run python scripts/oncogene/plot_benchmarks.py
```

`run_predictions.py` writes only the ranked gene list.
`run_benchmarks.py` defaults to the PDM KS diagnostic and leave-one-positive-out
rank recovery (LOOCV). It writes `oncogene_ks_pvalues.csv`, `oncogene_loocv_curve.csv`, and
`oncogene_loocv_held_out_ranks.csv`, without generating figures or rankings.
Use `--methods PDM,Pearson` to select feature methods and `--loocv-trials` to
change the ranking trial count per held-out positive. KS always uses PDM;
`--benchmarks ks` runs just that report, optionally with `--negative-label-file`
for explicit comparison labels instead of unlabeled candidates.

To evaluate against explicit negative labels instead of running LOOCV:

```bash
uv run python scripts/oncogene/run_benchmarks.py --benchmarks labeled \
  --label-file data/oncogene.txt --negative-label-file /path/to/negative_labels.txt
```

This writes `oncogene_method_benchmarks.csv` containing ROC AUC and average
precision. Use `--benchmarks loocv labeled` to run both. The labeled benchmark
requires explicit negative labels; unlabeled candidates are not used as
confirmed negatives. `--n-splits` controls its cross-validation folds.
Both benchmarks retain the existing model, feature selection, and random-seed
options. The old prediction flags `--loocv` and `--loocv-methods` are replaced
by `run_benchmarks.py --benchmarks loocv --methods ...`.

`plot_benchmarks.py` reads existing results and writes PNG and PDF files to
`RESULTS_DIR/figures/`. By default it plots all available benchmark
results, reporting missing inputs. Explicitly selected plots require their
source CSVs. To plot a chosen run:

```bash
uv run python scripts/oncogene/plot_benchmarks.py \
  --results-dir runs/my-analysis/oncogene/results \
  --output-dir figures/oncogene --plots ks loocv
```

Available plot names are `ks`, `loocv`, `models`, `ablation`, `pr`, `roc`, and `labeled`.
The heatmap retains
`--vmin`, `--vmax`, and `--cmap` options. Rank-recovery area is not ROC AUC;
KS p-values describe distribution differences, not enrichment direction.
The plotting script replaces `scripts/oncogene/make_ks_heatmap.py`.

### Paper figure suite

Generate the method comparison, classifier comparison, assay ablation, and
precision–recall/ROC curves from existing feature tables:

```bash
uv run python scripts/oncogene/run_benchmarks.py \
  --label-file data/oncogene.txt \
  --benchmarks ks loocv models ablation pr
uv run python scripts/oncogene/plot_benchmarks.py
```

These are new evaluations, so they need not match numbers from an older version.
All comparisons use the intersection of complete genes across the selected
feature methods, sorted identically. KS remains a standalone diagnostic on the
full PDM table. Feature selection stays inside each training trial or fold.

| Benchmark | Saved results (after the `oncogene_` prefix) | Figure |
| --- | --- | --- |
| `loocv` | `loocv_curve.csv`, `loocv_held_out_ranks.csv` | Feature-method recovery curves |
| `models` | `model_loocv_curve.csv`, `model_loocv_held_out_ranks.csv`, `model_benchmarks.csv` | PDM classifier recovery curves |
| `ablation` | `ablation.csv`, `ablation_held_out_ranks.csv` | Assay inclusion matrix with recovery-area dots |
| `pr` or `labeled` | `cv_predictions.csv`, `pr_curve.csv`, `roc_curve.csv`, `method_benchmarks.csv` | PR, ROC, and metric summary |

`models` uses the four supported classifiers: logistic regression, random forest,
gradient boosting, and SVM. Select a subset with `--models gradient_boosting svm`.
The other benchmarks use `--model` (default: gradient boosting). Model summaries
also record elapsed seconds. `ablation` uses PDM and evaluates all five assays,
each single omission, and each pair of omissions: 16 combinations total. Removing
an assay removes every feature pair involving it, while keeping the gene set fixed.

SVM uses sigmoid calibration through `CalibratedClassifierCV(ensemble=False)`
with up to five stratified folds inside each training set. It requires at least
two training rows per class (at least three positives for LOOCV). Scaling is fitted
inside the calibration folds. SVM scores can differ from older runs using
`SVC(probability=True)`.

Method and classifier curves show the number of recovered oncogenes by default;
use `plot_benchmarks.py --recovery-scale fraction` for fractions. Their legends
report normalized **rank-recovery area**, not ROC AUC. The ablation dots report
the same rank-recovery area on a zoomed axis, with the all-assay score marked
as a dashed reference. Metric-summary panels also use separate scales fitted
to their observed values. Plots use Matplotlib's bundled seaborn white-grid theme.

`pr` runs stratified cross-validation and saves one out-of-fold score per gene
and method, with its label and fold. PR/ROC and their AP/AUC summaries use these
pooled scores; they evaluate the classifier, separately from the repeated
positive–unlabeled ranking procedure. Without `--negative-label-file`, class zero
means **unlabeled**, not a confirmed non-oncogene. In that case precision and AP
measure known-label recovery, and ROC measures separation of known oncogenes from
unlabeled genes. They do not estimate biological precision or specificity.
The PR reference line is the known-positive prevalence in the evaluated set.
The displayed precision axis fits the values at evaluated thresholds; the
conventional extra `(recall=0, precision=1)` endpoint is omitted and noted on
the figure. This does not alter saved curves, predictions, or AP calculations.
Supply explicit negative labels for a control-based evaluation, or use `labeled`,
which requires them and writes the same outputs. Do not select both `pr` and `labeled`.

Model comparison and ablation are opt-in because they rerun LOOCV many times:
approximately `(number of methods + number of models + 16) × positive genes ×
loocv-trials` model fits for the full suite. Use small fixtures and fewer trials
for a pipeline check; use adequate trials for the paper results. Plotting alone
never fits models and exports both 300-dpi PNG and PDF with embedded TrueType fonts.

For a custom run, pass the same `--feature-dir`, `--results-dir`, and
`--label-name` to prediction and benchmarking, and the matching `--results-dir`
and `--label-name` to plotting. `reproduce.py oncogene` continues to run only
feature generation and prediction; run benchmarks and plotting separately.

To add a benchmark, write a function in `run_benchmarks.py` returning
`{filename_suffix: dataframe}` and add it to `BENCHMARKS`. To add its plot,
write a function in `plot_benchmarks.py` and add its input suffix, output suffix,
and function to `PLOTS`. The existing CLI and output loops handle the rest.

## Simulation benchmarks and figures

Simulation evaluates noisy associations against shuffled controls using stratified
cross-validation. It compares PDM, Pearson, Spearman, mutual information, and bicor:

```bash
uv run python scripts/simulation/make_features.py
uv run python scripts/simulation/run_benchmarks.py
uv run python scripts/simulation/plot_benchmarks.py
```

The default directories are `runs/simulation/features/` and
`runs/simulation/results/`. Benchmarking writes `simulation_benchmarks.csv`,
`simulation_cv_predictions.csv`, `simulation_roc_curve.csv`, and
`simulation_pr_curve.csv`. Plotting reads those CSVs and exports a metric summary,
ROC, and PR as PNG/PDF files under `RESULTS_DIR/figures/`. It never fits models.
`reproduce.py simulation` generates features and benchmarks; plotting is a separate step.

To also compare classifiers on PDM features:

```bash
uv run python scripts/simulation/run_benchmarks.py \
  --models logistic_regression random_forest gradient_boosting svm
uv run python scripts/simulation/plot_benchmarks.py
```

This adds corresponding `simulation_model_*.csv` files and three classifier
figures. `--model` selects the classifier for feature-method comparisons;
`--methods pdm pearson` selects feature methods. Existing fold count, feature
selection, and random-seed options remain available. Use `--feature-dir` and
`--results-dir` for another run; plotting accepts `--results-dir`, `--output-dir`,
and `--plots summary roc pr models model-roc model-pr` (default: all available).

Feature CSVs must come from the same generation run: rows correspond by position.
The benchmark requires matching positive/negative row counts and keeps only rows
finite across all selected methods. Every method and classifier uses the same
fold assignments. PDM feature selection uses training rows only. AUC and AP use
pooled out-of-fold scores, also saved with simulation IDs, targets, and fold IDs.

These evaluate supervised classification of new realizations of the four seed
patterns, not unseen pattern families or the transductive ranking procedure.
Shuffled controls supply the negative class; the PR baseline reflects the chosen
positive/control ratio. Metric and precision axes adapt to observed values;
the conventional PR `(0, 1)` endpoint is omitted from display, without changing AP.
Assay ablation and oncogene rank-recovery plots are not part of this workflow.

### Compare noise levels

Pass several standard deviations to feature generation; benchmarking and plotting
automatically read the resulting sweep:

```bash
uv run python scripts/simulation/make_features.py --epsilon-std 0.5 1 2 3
uv run python scripts/simulation/run_benchmarks.py
uv run python scripts/simulation/plot_benchmarks.py
```

`simulation_benchmarks.png/.pdf` now shows ROC AUC and average precision versus
noise, with one line per feature method. ROC and PR figures show one panel per
noise level. Optional `--models ...` comparisons get the same combined views.
The CSVs retain their filenames and include `epsilon_std` to distinguish levels;
predictions and ROC/PR curves are evaluated separately at each level.

Generation writes `epsilon_<value>/positive/` and `epsilon_<value>/negative/`
under the feature directory, plus `simulation_manifest.json` identifying the
levels and generation settings. The benchmark uses only levels in that manifest.
One standard deviation retains the original flat directory layout, and older
single-level feature directories without a manifest remain supported.

Random draws and control permutations are reused across levels. Benchmarking
retains the same complete simulation rows across all levels and methods, with
identical folds, so changes in noise can be compared directly. The lines summarize
one seeded experiment, not repeated-run uncertainty estimates.

The reproduction entry point also accepts the list:

```bash
uv run python scripts/reproduce.py simulation --epsilon-std 0.5 1 2 3
uv run python scripts/simulation/plot_benchmarks.py
```

Alternatively, set `simulation.epsilon_std` in the workflow JSON to a number or
list. The default remains `2.0`; values must be finite and non-negative.

## PPI benchmark interpretation

PPI benchmarks hold out cancer-complex groups, joining complexes that share
curated positive pairs into one group to prevent overlap across folds.
Their sampled negative pairs are exclusion-based controls, not confirmed
noninteractions.
These benchmarks evaluate a classifier on the supplied controls, not the
candidate-ranking procedure or the benefit of its sampling cap. They also
do not establish generalization to unseen proteins.

Open `notebooks/reviewer_walkthrough.ipynb` and set `RUN_DIR` to your output
directory to inspect the results.
