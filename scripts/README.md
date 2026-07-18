# Scripts

Command-line entry points are grouped by workflow:

- `data/`: prepare shared trimmed input tables.
- `oncogene/`: a thin DepMap-oncogene recipe around the generic core.
- `ppi/`: a thin DepMap-PPI recipe around generic pair-table features.
- `simulation/`: generate and benchmark simulated feature tables.
- `validation/`: sanity-check generated PPI/control artifacts.
- `figures/`: export publication figures from saved results.

For arbitrary entity-by-sample matrices, use `pdm_learn.datasets`,
`pdm_learn.features`, `pdm_learn.labels`, and `pdm_learn.modeling` directly.
For a table with two entity IDs per observation, use
`pdm_learn.pair_features`.

Run commands from the repository root with `PYTHONPATH=src`.
