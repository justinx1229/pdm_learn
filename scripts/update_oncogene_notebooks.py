from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks" / "oncogene"


def _code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip("\n").splitlines()],
    }


COMMON_SETUP = """
# Shared project setup for imports and file locations
from pathlib import Path
import sys

PROJECT_ROOT = Path.cwd().resolve()
while PROJECT_ROOT != PROJECT_ROOT.parent and not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
FIGURES_DIR = PROJECT_ROOT / "figures"
"""


DATA_CONVERSION_CELLS = [
    _code_cell(
        COMMON_SETUP
        + """
import pandas as pd

from pdm_learn.oncogene import (
    build_oncogene_density_features,
    build_oncogene_statistic_features,
    load_oncogene_inputs,
)
"""
    ),
    _code_cell(
        """
datasets = load_oncogene_inputs(DATA_DIR)
density_dataset = build_oncogene_density_features(datasets)
density_dataset.head()
"""
    ),
    _code_cell(
        """
statistic_tables = {
    "pearson": build_oncogene_statistic_features(datasets, method="pearson"),
    "spearman": build_oncogene_statistic_features(datasets, method="spearman"),
    "mi": build_oncogene_statistic_features(datasets, method="mi"),
    "bicor": build_oncogene_statistic_features(datasets, method="bicor"),
}

{name: table.head() for name, table in statistic_tables.items()}
"""
    ),
    _code_cell(
        """
output_dir = DATA_DIR / "Trimmed data"
output_dir.mkdir(parents=True, exist_ok=True)

output_paths = {
    "density": output_dir / "dataset_trimmed_v3.csv",
    "pearson": output_dir / "pearson.csv",
    "spearman": output_dir / "spearman.csv",
    "mi": output_dir / "mi.csv",
    "bicor": output_dir / "bicor.csv",
}

density_dataset.to_csv(output_paths["density"], index=False)
for name, table in statistic_tables.items():
    table.to_csv(output_paths[name], index=False)

output_paths
"""
    ),
]


PEARSON_SPEARMAN_CELLS = [
    _code_cell(
        COMMON_SETUP
        + """
import pandas as pd

from pdm_learn.oncogene import (
    build_oncogene_statistic_features,
    load_oncogene_inputs,
)
"""
    ),
    _code_cell(
        """
datasets = load_oncogene_inputs(DATA_DIR)
pearson_dataset = build_oncogene_statistic_features(datasets, method="pearson")
pearson_dataset.head()
"""
    ),
    _code_cell(
        """
pearson_path = DATA_DIR / "Trimmed data" / "pearson.csv"
pearson_path.parent.mkdir(parents=True, exist_ok=True)
pearson_dataset.to_csv(pearson_path, index=False)
pearson_path
"""
    ),
    _code_cell(
        """
spearman_dataset = build_oncogene_statistic_features(datasets, method="spearman")
spearman_dataset.head()
"""
    ),
    _code_cell(
        """
spearman_path = DATA_DIR / "Trimmed data" / "spearman.csv"
spearman_dataset.to_csv(spearman_path, index=False)
spearman_path
"""
    ),
]


MACHINE_LEARNING_CELLS = [
    _code_cell(
        COMMON_SETUP
        + """
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pdm_learn.modeling import LOOCV_grouped_plot, area_table, heatmap, ks_pvalue
from pdm_learn.oncogene import (
    ONCOGENE_HEATMAP_DIMENSIONS,
    evaluate_method_curves,
    load_oncogene_feature_sets,
    plot_method_curves,
    rank_candidate_oncogenes,
)
"""
    ),
    _code_cell(
        """
data_dict, input_data = load_oncogene_feature_sets(DATA_DIR)
positive, negative = data_dict["PDM"]

pd.DataFrame(
    {
        "dataset": list(data_dict.keys()),
        "positive_genes": [len(value[0]) for value in data_dict.values()],
        "negative_genes": [len(value[1]) for value in data_dict.values()],
    }
)
"""
    ),
    _code_cell(
        """
input_data.head()
"""
    ),
    _code_cell(
        """
MODEL_LIST = ("SVR", "XGB", "GBR", "MLP", "LR", "GNB")
DEFAULT_MODEL = "XGB"

MODEL_LIST
"""
    ),
    _code_cell(
        """
ranking = rank_candidate_oncogenes(
    positive,
    negative,
    trials=100,
    model=DEFAULT_MODEL,
    ks_test=True,
)
ranking.head(20)
"""
    ),
    _code_cell(
        """
LOOCV_grouped_plot(
    data_dict,
    5,
    models=MODEL_LIST,
)
"""
    ),
    _code_cell(
        """
pr_results = evaluate_method_curves(
    data_dict,
    trials=25,
    model=DEFAULT_MODEL,
    ks_test=False,
    metric="pr",
)
plot_method_curves(
    pr_results,
    title="Oncogene Precision-Recall Curves",
    xlabel="Recall",
    ylabel="Precision",
)
pr_results
"""
    ),
    _code_cell(
        """
loocv_results = evaluate_method_curves(
    data_dict,
    trials=25,
    model=DEFAULT_MODEL,
    ks_test=False,
    metric="loocv",
)
plot_method_curves(
    loocv_results,
    title="Oncogene LOOCV Curves",
    xlabel="Rank",
    ylabel="Cumulative positives",
)
loocv_results
"""
    ),
    _code_cell(
        """
feature_arr = [10, 20, 50, 100, 124, 150, 200, 250, 300, 349]
areas = area_table(
    positive.iloc[:, 1:].to_numpy(),
    negative.iloc[:, 1:].to_numpy(),
    20,
    model=DEFAULT_MODEL,
    feat_arr=feature_arr,
)

plt.figure(figsize=(8, 4))
plt.plot(feature_arr, areas, marker="o")
plt.title("PDM Feature Sweep")
plt.xlabel("Selected Features")
plt.ylabel("LOOCV Area")
plt.tight_layout()
plt.show()
"""
    ),
    _code_cell(
        """
p_val = ks_pvalue(positive.iloc[:, 1:].to_numpy(), negative.iloc[:, 1:].to_numpy())
log_p_val = np.log10(np.clip(p_val, 1e-300, None))
heatmap(
    log_p_val,
    ONCOGENE_HEATMAP_DIMENSIONS,
    cmap="gist_heat_r",
    min=float(np.min(log_p_val)),
    max=float(np.max(log_p_val)),
    flip=True,
    axes=False,
    colorbar=False,
)
"""
    ),
]


def rewrite_notebook(path: Path, cells: list[dict]) -> None:
    notebook = json.loads(path.read_text())
    notebook["cells"] = [
        cell
        for cell in cells
        if cell.get("cell_type") != "code" or "".join(cell.get("source", [])).strip()
    ]
    path.write_text(json.dumps(notebook, indent=1))


def main() -> None:
    rewrite_notebook(NOTEBOOKS / "data conversion.ipynb", DATA_CONVERSION_CELLS)
    pearson_spearman_path = NOTEBOOKS / "data conversion (pearson_spearman).ipynb"
    if pearson_spearman_path.exists():
        rewrite_notebook(
            pearson_spearman_path,
            PEARSON_SPEARMAN_CELLS,
        )
    rewrite_notebook(NOTEBOOKS / "Machine Learning.ipynb", MACHINE_LEARNING_CELLS)


if __name__ == "__main__":
    main()
