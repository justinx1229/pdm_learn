# PDM Learn

PDM Learn is a Python library for learning features from aligned
entity-by-sample matrices. It provides matrix preparation, pairwise density
maps (PDMs), scalar association features, labels, candidate ranking, and
simulation utilities without assuming a particular scientific domain.

## Install

For development in this checkout:

```bash
uv sync --group dev --locked
uv run pytest
```

To install a released package:

```bash
pip install pdm-learn
```

## Review and reproduction

The package is tested independently of the manuscript workflows.  For the
reviewer-facing oncogene, PPI, and simulation analyses, see
[REPRODUCE.md](REPRODUCE.md). It documents the input layout, commands,
expected outputs, and a small example using the same pipeline as your own data.

The command-line workflows are canonical.  The optional
[reviewer walkthrough](notebooks/reviewer_walkthrough.ipynb) explains their
outputs after they have run; it does not duplicate analysis logic.
To execute it locally, install the optional notebook environment with
`uv sync --group review --locked`.

## Core API

The public API lives in `src/pdm_learn/` and is imported as `pdm_learn`.
Matrices stay in memory after loading; feature builders never depend on a
fixed file layout or a biological identifier name.

```python
from pdm_learn import (
    DatasetSpec,
    FeatureConfig,
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

datasets = load_matrices(
    {
        "expression": "expression.csv",
        "mutation": "mutation.csv",
    }
)

feature_tables = build_features(
    datasets,
    specs=specs,
    pairs=[FeaturePair("expression", "mutation")],
    methods=("pdm", "pearson", "spearman", "mi", "bicor"),
    config=FeatureConfig(density_bins=7),
)

positives = load_label_set("known_positives.csv")
ranking = rank_candidates(feature_tables["pdm"], positives)
```

`DatasetSpec` explicitly describes each matrix:

- `kind="continuous"`, `"discrete"`, or `"binary"`
- `levels` for known discrete states
- `normalize="row_center"` or `None`
- `id_column` for any identifier header, in any input position

When no specs are passed to `load_matrices`, it performs only lossless
structural cleanup. The feature builder then applies the specifications once.

`build_features` returns one DataFrame for each requested method. The first
column is the output identifier; the remaining columns are numeric features.

## Ordered-pair tables

Use `build_pair_features` when each output row joins two different entities.
The original pair-table metadata is retained, while generated feature columns
are appended.

```python
from pdm_learn import FeaturePair, PairFeatureConfig, build_pair_features

pair_features = build_pair_features(
    datasets,
    pair_table,
    left_id_column="source_gene",
    right_id_column="target_gene",
    specs=specs,
    pairs=[FeaturePair("expression", "mutation")],
    methods=("pdm", "pearson"),
    config=PairFeatureConfig(density_bins=7),
)
```

For pair tables with non-numeric metadata, pass `feature_columns` to
`rank_candidates` or `evaluate_labeled_features` to select the generated
features explicitly.

## Optional workflows

The generic package has no DepMap, oncogene, or PPI assumptions. The optional
recipes below `pdm_learn.workflows` only select maintained DepMap defaults:

```python
from pdm_learn.workflows import oncogene, ppi

oncogene_datasets = oncogene.load_inputs("data")
oncogene_features = oncogene.build_features(oncogene_datasets)

ppi_datasets = ppi.load_inputs("data")
ppi_features = ppi.build_features(ppi_datasets, pair_table)
```

For a custom project, prefer the core API directly. Version 1.0 intentionally
removes the former root-level `pdm_learn.oncogene`, `pdm_learn.ppi`, and
`pdm_learn.preprocessing` modules.

## Development and release checks

```bash
uv run ruff check src tests
uv run pytest
uv build
```

For the reviewer-facing workflow and input layout, see
[REPRODUCE.md](REPRODUCE.md).
