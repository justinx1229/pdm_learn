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

`uv` is the recommended environment manager for this project. It reads
`pyproject.toml`, creates the local `.venv`, installs `pdm_learn` in editable
mode, and uses `uv.lock` as the reproducible dependency record.

```bash
uv sync
uv run python -m unittest discover -s tests
```

If you prefer to start from conda, use conda only as a lightweight bootstrap for
Python and `uv`, then let `uv` install the project dependencies:

```bash
conda env create -f environment.yml
conda activate pdm_learn_env
uv sync
uv run python -m unittest discover -s tests
```

Launch notebooks from the locked environment with:

```bash
uv run jupyter lab
```

## Notebook workflow

Notebooks under `notebooks/` can import shared helpers from `pdm_learn`. If you add new duplicated helper cells later, rerun:

```bash
python3 scripts/refactor_notebooks.py
```

The notebook setup cell defines `PROJECT_ROOT`, `DATA_DIR`, `ARTIFACTS_DIR`, and `FIGURES_DIR`, which keeps file references explicit and easy to follow without needing a separate path helper module.

## Version-control notes

This project includes large datasets, pickles, notebook checkpoints, and generated outputs. The provided `.gitignore` is set up so you can initialize a repository without accidentally committing bulky local artifacts. If you eventually want to track large data assets, Git LFS is the safer next step.
