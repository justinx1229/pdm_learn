# PDM Learn

This project contains notebook-driven analyses for CCLE/DepMap feature engineering, simulated associations, and ranking experiments. The repeated notebook logic now lives in a reusable Python package under `src/pdm_learn` so the notebooks can stay focused on analysis instead of carrying local copies of the same helper functions.

## Project layout

- `src/pdm_learn/`: shared preprocessing, modeling, and simulation helpers
- `notebooks/`: active notebooks grouped by workflow area
- `scripts/refactor_notebooks.py`: rewrites notebooks to import the shared package and use simple project-relative `Path` variables
- `data/`: datasets and generated tabular assets used by the notebooks
- `artifacts/`: generated pickles and result files that do not belong at the repo root
- `figures/`: exported plots and images
- `tests/`: lightweight regression checks for the shared Python modules

## Getting started

Create an environment and install the package in editable mode:

```bash
python3 -m pip install -e .
```

If you prefer a flat dependency install instead:

```bash
python3 -m pip install -r requirements.txt
```

## Notebook workflow

Notebooks under `notebooks/` can import shared helpers from `pdm_learn`. If you add new duplicated helper cells later, rerun:

```bash
python3 scripts/refactor_notebooks.py
```

The notebook setup cell defines `PROJECT_ROOT`, `DATA_DIR`, `ARTIFACTS_DIR`, and `FIGURES_DIR`, which keeps file references explicit and easy to follow without needing a separate path helper module.

## Version-control notes

This project includes large datasets, pickles, notebook checkpoints, and generated outputs. The provided `.gitignore` is set up so you can initialize a repository without accidentally committing bulky local artifacts. If you eventually want to track large data assets, Git LFS is the safer next step.
