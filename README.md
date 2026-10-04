# PDM Learn

PDM Learn is a Python library for building features from entity-by-sample
matrices and ranking unlabeled candidates. It supports continuous, discrete,
and binary data, with a domain-independent API and optional DepMap workflows
for oncogene and protein–protein interaction (PPI) analyses.

A pairwise density map (PDM) represents the joint distribution of two vectors
across shared samples. PDM Learn flattens and log-transforms these maps into
features for classification. It also provides Pearson correlation, Spearman
correlation, mutual information, and biweight midcorrelation (`bicor`) for
comparison.

## Installation

Python **3.11 or newer** is required. From the repository root, install the
package and development tools using the committed lockfile:

```bash
uv sync --group dev --locked
```

Alternatively, install the library from this checkout with pip:

```bash
python -m pip install .
```

The analysis scripts, sample data, and notebook are repository resources;
use a checkout to run them. The installed Python package is `pdm_learn`.

## Quick start

This self-contained example builds features for the same entities in two
matrices, then ranks candidates outside the supplied positive set. The toy
data demonstrates the API, not a scientific result.

```python
import pandas as pd

from pdm_learn import DatasetSpec, FeatureConfig, FeaturePair, build_features, rank_candidates

entities = list("ABCDEF")
datasets = {
    "expression": pd.DataFrame({
        "entity": entities,
        "S1": [1, 2, 4, 3, 2, 5],
        "S2": [2, 4, 3, 1, 5, 2],
        "S3": [4, 3, 1, 2, 3, 4],
        "S4": [3, 1, 2, 4, 1, 3],
    }),
    "mutation": pd.DataFrame({
        "entity": entities,
        "S1": [0, 0, 1, 0, 1, 1],
        "S2": [0, 1, 0, 1, 0, 1],
        "S3": [1, 1, 0, 0, 1, 0],
        "S4": [1, 0, 1, 1, 0, 0],
    }),
}
specs = [
    DatasetSpec("expression", id_column="entity"),
    DatasetSpec("mutation", kind="binary", normalize=None, id_column="entity"),
]

feature_tables = build_features(
    datasets,
    specs=specs,
    pairs=[FeaturePair("expression", "mutation")],
    methods=("pdm", "pearson", "spearman", "mi", "bicor"),
    config=FeatureConfig(density_bins=7),
)

ranking = rank_candidates(feature_tables["pdm"], {"A", "B"}, trials=100, random_state=42)
print(ranking)  # entity, score, score_trials; candidates C–F sorted by score
```

`build_features` returns a dictionary of DataFrames keyed by method. Each
row represents one entity, with its identifier first and numeric features
after it. This example produces 14 PDM features (seven continuous centers ×
two binary states) and one feature per scalar method.

## Input format and preparation

Each matrix has one unique, nonblank entity identifier per row and sample
identifiers as column headers. Matrices are matched by identifiers and shared
sample names, so their row and column orders can differ.

`DatasetSpec` controls each matrix:

- `id_column` selects the identifier header, even when it is not the first
  column. The default is the first column.
- `kind` is `"continuous"` (default), `"discrete"`, or `"binary"`.
- `normalize="row_center"` subtracts each row's sample mean. Use
  `normalize=None` to preserve values, including discrete and binary states.
- `levels` declares discrete states; otherwise they are inferred. Binary
  states default to `(0, 1)`.

For CSV matrices with identifiers in the first column, replace the example's
`datasets` with:

```python
from pdm_learn import load_label_set, load_matrices

# Loading without specs cleans the tables without normalizing values.
datasets = load_matrices({"expression": "expression.csv", "mutation": "mutation.csv"})
# Supply the specs to build_features, as in the quick start.
positives = load_label_set("known_positives.csv", header=0, column="entity")
```

For identifiers in other positions, load raw DataFrames with `pd.read_csv`
and supply `id_column` through the feature builder's specs. Label files without
headers can use `load_label_set("known_positives.txt")`; multiple files are
combined by set union.

Preparation converts nonnumeric sample values to missing values. Feature
calculations use finite paired observations. Row-level feature construction
fills missing binary entries and absent binary entity rows with zero where
alignment includes them; use binary inputs only when this matches your data.
`FeatureConfig` defaults to the union of entities and drops rows with missing
features. Use `entity_universe="intersection"` to restrict eligible entities,
or `drop_missing=False` to inspect incomplete rows. Each method may retain a
different set of rows.

## Ordered-pair features

Use `build_pair_features` when each output row compares two entities, such as
an interaction candidate. Continuing with the quick-start data:

```python
from pdm_learn import PairFeatureConfig, build_pair_features

pair_table = pd.DataFrame({
    "pair_id": ["A:B", "C:D", "E:F"],
    "source": ["A", "C", "E"],
    "target": ["B", "D", "F"],
})
pair_features = build_pair_features(
    datasets,
    pair_table,
    left_id_column="source",
    right_id_column="target",
    specs=specs,
    pairs=[FeaturePair("expression", "mutation")],
    methods=("pdm", "pearson"),
    config=PairFeatureConfig(density_bins=7),
)
```

Here, each block compares the source's expression with the target's mutation
values across shared samples. Output tables preserve pair-table metadata and
append generated features. Unresolved entity IDs produce missing features;
`drop_missing=True` removes those rows rather than imputing their entities.

Without explicit `pairs`, the row-level builder uses each distinct matrix
combination once; the pair-table builder uses all ordered matrix combinations,
including a matrix paired with itself. For modeling pair tables, supply a
unique `id_column` and explicit `feature_columns` so numeric metadata is not
used as a feature.

## Ranking and evaluation

`rank_candidates` (also available as `rank_unlabeled`) repeatedly fits a
balanced classifier to known positives and randomly sampled provisional
negatives from the unlabeled pool. Each trial scores only candidates excluded
from that trial's training set. Ranking requires at least one positive and
two unlabeled candidates with complete numeric features.

The returned `score` averages those held-out predictions; `score_trials`
counts them. Scores are ranking signals, not calibrated probabilities of
true positive status. Candidates never scored have a missing score and zero
`score_trials`; increasing `trials` improves coverage.

Available models are `logistic_regression` (the core API default),
`random_forest`, `gradient_boosting`, and `svm`. Ranking selects features using
KS-test p-values within each trial by default. `positive_groups` together with
`max_positive_per_group` enables capped positive sampling by group.

`evaluate_labeled_features` reports cross-validated ROC AUC and average
precision, with optional group-aware folds and feature selection within each
training fold. Supply `negative_labels` for explicit negative controls;
otherwise it treats every non-positive row as negative.
`loocv_rank_recovery` instead measures recovery of held-out known positives
in candidate rankings.

## Analysis workflows

Run all three workflows on the included small fixtures:

```bash
uv run python scripts/reproduce.py all \
  --data-dir tests/fixtures/reproduction \
  --config tests/fixtures/reproduction/workflows.json \
  --output-dir runs/example
```

Replace `all` with `oncogene`, `ppi`, or `simulation` to run one workflow.
Simulation generates its own inputs. For study or custom data, supply your
input directory and settings; defaults are `data/`, `runs/`, and
[`reproduction/workflows.json`](reproduction/workflows.json).

| Workflow | Results directory | Main files |
| --- | --- | --- |
| Oncogene | `oncogene/results/` | `oncogene_ranking.csv`, `oncogene_ks_pvalues.csv` |
| PPI | `ppi/results/` | `ranked_PPI_pairs_cancer.csv`, `ppi_control_benchmarks.csv` |
| Simulation | `simulation/results/` | `simulation_benchmarks.csv` |

See [REPRODUCE.md](REPRODUCE.md) for input filenames, settings, control
sampling, benchmark interpretation, and the oncogene KS heatmap command.
The fixture run exercises the pipeline without reproducing study results.

The Python recipes live in `pdm_learn.workflows.oncogene` and
`pdm_learn.workflows.ppi`; their DepMap specifications live in
`pdm_learn.presets`. For other domains, use the core API directly. Version 1.0
removes the former root-level `pdm_learn.oncogene`, `pdm_learn.ppi`, and
`pdm_learn.preprocessing` modules.

The optional [reviewer walkthrough](notebooks/reviewer_walkthrough.ipynb)
reads workflow outputs. Install its environment with
`uv sync --group review --locked`, then open it with
`uv run jupyter lab notebooks/reviewer_walkthrough.ipynb` and set `RUN_DIR`
to your output directory. The command-line scripts contain the analysis logic.

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/pdm_learn/datasets.py` | Matrix loading, preparation, and alignment |
| `src/pdm_learn/features.py`, `pair_features.py` | Entity and ordered-pair feature builders |
| `src/pdm_learn/labels.py`, `modeling.py` | Label handling, ranking, and evaluation |
| `src/pdm_learn/simulation.py` | Reusable simulation and feature utilities |
| `src/pdm_learn/workflows/` | DepMap recipes and PPI control utilities |
| `scripts/`, `reproduction/` | Analysis entry points and default settings |
| `tests/` | Unit and workflow integration tests, including small input fixtures |

## Development

```bash
uv sync --group dev --locked
uv run ruff check src tests
uv run pytest
uv build
```

CI runs lint and tests on Python 3.11, 3.12, and 3.13, checks source-distribution
contents, and verifies that the wheel imports in a fresh environment.
