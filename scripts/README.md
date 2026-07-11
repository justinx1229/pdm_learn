# Scripts

Command-line entry points are grouped by workflow:

- `data/`: prepare shared trimmed input tables.
- `generic/`: build and rank arbitrary gene-by-sample datasets.
- `oncogene/`: build gene-level features and run label-based rankings.
- `ppi/`: build PPI controls/features and rank BioGRID pairs.
- `simulation/`: generate and benchmark simulated feature tables.
- `validation/`: sanity-check generated PPI/control artifacts.
- `figures/`: export publication figures from saved results.
- `notebooks/`: maintenance helpers for notebook cleanup or refresh.

Run commands from the repository root with `PYTHONPATH=src`.
