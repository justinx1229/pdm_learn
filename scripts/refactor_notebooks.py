from __future__ import annotations

import argparse
import json
import re
from pathlib import Path, PurePosixPath


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = PROJECT_ROOT / "notebooks"
NOTEBOOKS = sorted(path for path in NOTEBOOK_ROOT.rglob("*.ipynb"))

SHARED_FUNCTIONS = {
    "build_density_map",
    "core_predict",
    "density_centers",
    "densitymap",
    "drop_nan",
    "eps",
    "extract",
    "heatmap",
    "importance_test",
    "KFold_PR",
    "ks_pvalue",
    "LOOCV",
    "LOOCV_grouped_plot",
    "mut_trim",
    "normalize",
    "partition",
    "trim",
    "trim_pairs",
    "area_table",
}

FUNCTION_PATTERN = re.compile(r"^def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.MULTILINE)
RESOLVE_PATTERN = re.compile(r"resolve_(?:legacy_)?path\((?P<literal>r?[\"'][^\"']+[\"'])\)")
TO_CSV_PATTERN = re.compile(
    r"\.to_csv\((?P<path>[^\n]+?)\)\s*,\s*(?:\n\s*)?index\s*=\s*(?P<index>[^)\n]+)\)"
)

SETUP_CELL_SOURCE = [
    "# Shared project setup for imports and file locations\n",
    "from pathlib import Path\n",
    "import sys\n",
    "\n",
    "PROJECT_ROOT = Path.cwd().resolve()\n",
    "while PROJECT_ROOT != PROJECT_ROOT.parent and not (PROJECT_ROOT / 'src').exists():\n",
    "    PROJECT_ROOT = PROJECT_ROOT.parent\n",
    "SRC_DIR = PROJECT_ROOT / 'src'\n",
    "if str(SRC_DIR) not in sys.path:\n",
    "    sys.path.insert(0, str(SRC_DIR))\n",
    "\n",
    "DATA_DIR = PROJECT_ROOT / 'data'\n",
    "ARTIFACTS_DIR = PROJECT_ROOT / 'artifacts'\n",
    "FIGURES_DIR = PROJECT_ROOT / 'figures'\n",
    "\n",
    "def resolve_path(path):\n",
    "    candidate = Path(path)\n",
    "    if candidate.exists():\n",
    "        return candidate\n",
    "    text = str(path).replace('\\\\', '/')\n",
    "    name = Path(text).name\n",
    "    special = {\n",
    "        'positive_controls.pkl': ARTIFACTS_DIR / 'controls' / 'positive_controls.pkl',\n",
    "        'negative_controls.pkl': ARTIFACTS_DIR / 'controls' / 'negative_controls.pkl',\n",
    "        'Ten_positive_controls_1119.pkl': ARTIFACTS_DIR / 'controls' / 'positive_controls.pkl',\n",
    "        'Ten_negative_controls_1119.pkl': ARTIFACTS_DIR / 'controls' / 'negative_controls.pkl',\n",
    "        'fcg.txt': DATA_DIR / 'fcg.txt',\n",
    "    }\n",
    "    if name in special:\n",
    "        return special[name]\n",
    "    matches = [p for p in PROJECT_ROOT.rglob(name) if '.ipynb_checkpoints' not in p.parts and '.git' not in p.parts]\n",
    "    if len(matches) == 1:\n",
    "        return matches[0]\n",
    "    if (text.startswith('/Users/') or text.startswith('/home/') or ':\\\\' in text) and '.' not in name:\n",
    "        return PROJECT_ROOT\n",
    "    return candidate\n",
    "\n",
    "from pdm_learn.preprocessing import build_density_map, density_centers, densitymap, drop_nan, extract, mut_trim, normalize, trim, trim_pairs\n",
    "from pdm_learn.modeling import KFold_PR, LOOCV, LOOCV_grouped_plot, area_table, core_predict, heatmap, importance_test, ks_pvalue\n",
    "from pdm_learn.simulation import build_heatmap_dataset, build_metric_dataset, eps, iter_simulated_pairs, partition, perturb_pair\n",
]

SPECIAL_PATHS = {
    "positive_controls.pkl": ("ARTIFACTS_DIR", "controls", "positive_controls.pkl"),
    "negative_controls.pkl": ("ARTIFACTS_DIR", "controls", "negative_controls.pkl"),
    "fcg.txt": ("DATA_DIR", "fcg.txt"),
    "FirstPositiveControl.csv": ("DATA_DIR", "PPI_Pairs", "FirstPositiveControl.csv"),
    "FirstNegativeControl.csv": ("DATA_DIR", "PPI_Pairs", "FirstNegativeControl.csv"),
    "SecondPositiveControl.csv": ("DATA_DIR", "PPI_Pairs", "SecondPositiveControl.csv"),
    "SecondNegativeControl.csv": ("DATA_DIR", "PPI_Pairs", "SecondNegativeControl.csv"),
    "ThirdPositiveControl.csv": ("DATA_DIR", "PPI_Pairs", "ThirdPositiveControl.csv"),
    "ThirdNegativeControl.csv": ("DATA_DIR", "PPI_Pairs", "ThirdNegativeControl.csv"),
    "FourthPositiveControl.csv": ("DATA_DIR", "PPI_Pairs", "FourthPositiveControl.csv"),
    "FourthNegativeControl.csv": ("DATA_DIR", "PPI_Pairs", "FourthNegativeControl.csv"),
    "FifthPositiveControl.csv": ("DATA_DIR", "PPI_Pairs", "FifthPositiveControl.csv"),
    "FifthNegativeControl.csv": ("DATA_DIR", "PPI_Pairs", "FifthNegativeControl.csv"),
    "Bicor_feature_sig_df_0217.csv": ("ARTIFACTS_DIR", "results", "Bicor_feature_sig_df_0217.csv"),
    "MI_feature_sig_df_0211.csv": ("ARTIFACTS_DIR", "results", "MI_feature_sig_df_0211.csv"),
}

MARKER_BASES = (
    ("Data/DepMap_Trimmed/", ("DATA_DIR", "DepMap_Trimmed")),
    ("DepMap_Trimmed/", ("DATA_DIR", "DepMap_Trimmed")),
    ("Data/DepMap_data/", ("DATA_DIR", "DepMap_data")),
    ("DepMap_data/", ("DATA_DIR", "DepMap_data")),
    ("Data/PPI_Pairs/", ("DATA_DIR", "PPI_Pairs")),
    ("PPI_Pairs/", ("DATA_DIR", "PPI_Pairs")),
    ("Data/Trimmed data/", ("DATA_DIR", "Trimmed data")),
    ("Trimmed data/", ("DATA_DIR", "Trimmed data")),
    ("Data/simulated/", ("DATA_DIR", "simulated")),
    ("Simulated Data/", ("DATA_DIR", "simulated")),
    ("simulated data/", ("DATA_DIR", "simulated")),
    ("artifacts/controls/", ("ARTIFACTS_DIR", "controls")),
    ("artifacts/results/", ("ARTIFACTS_DIR", "results")),
    ("Data/", ("DATA_DIR",)),
)


def _join_expression(root_name: str, parts: list[str]) -> str:
    expression = root_name
    for part in parts:
        expression += f" / {part!r}"
    return expression


def _literal_value(literal: str) -> str:
    if literal.startswith("r"):
        literal = literal[1:]
    return literal[1:-1]


def _suffix_after_marker(path_text: str, marker: str) -> str | None:
    normalized = path_text.replace("\\", "/")
    if marker not in normalized:
        return None
    return normalized.split(marker, 1)[1].lstrip("/")


def _relative_expression_for_path(path: Path) -> str:
    relative = path.relative_to(PROJECT_ROOT)
    parts = list(relative.parts)
    if not parts:
        return "PROJECT_ROOT"
    if parts[0] == "data":
        return _join_expression("DATA_DIR", parts[1:])
    if parts[0] == "artifacts":
        return _join_expression("ARTIFACTS_DIR", parts[1:])
    if parts[0] == "figures":
        return _join_expression("FIGURES_DIR", parts[1:])
    return _join_expression("PROJECT_ROOT", parts)


def _find_unique_repo_match(name: str) -> Path | None:
    matches = [
        path
        for path in PROJECT_ROOT.rglob(name)
        if ".ipynb_checkpoints" not in path.parts and ".git" not in path.parts
    ]
    if len(matches) == 1:
        return matches[0]
    return None


def _path_expression(path_text: str) -> str | None:
    normalized = path_text.replace("\\", "/")
    basename = PurePosixPath(normalized).name

    if basename in SPECIAL_PATHS:
        root_name, *parts = SPECIAL_PATHS[basename]
        return _join_expression(root_name, parts)

    for marker, base_parts in MARKER_BASES:
        suffix = _suffix_after_marker(normalized, marker)
        if suffix:
            root_name, *parts = base_parts
            return _join_expression(root_name, parts + list(PurePosixPath(suffix).parts))

    if (normalized.startswith("/Users/") or normalized.startswith("/home/") or ":\\" in path_text) and "." not in basename:
        return "PROJECT_ROOT"

    if basename:
        match = _find_unique_repo_match(basename)
        if match is not None:
            return _relative_expression_for_path(match)

    return None


def _replace_resolved_paths(source: str) -> str:
    def repl(match: re.Match[str]) -> str:
        path_expression = _path_expression(_literal_value(match.group("literal")))
        return path_expression or match.group(0)

    return RESOLVE_PATTERN.sub(repl, source)


def _replace_known_relative_literals(source: str) -> str:
    updated = source
    for name, (root_name, *parts) in SPECIAL_PATHS.items():
        expression = _join_expression(root_name, parts)
        updated = updated.replace(repr(name), expression)
        updated = updated.replace(f'"{name}"', expression)
    return updated


def _normalize_to_csv_calls(source: str) -> str:
    return TO_CSV_PATTERN.sub(
        lambda match: f".to_csv({match.group('path').strip()}, index={match.group('index').strip()})",
        source,
    )


def _normalize_path_expressions(source: str) -> str:
    updated = source
    updated = re.sub(r"ARTIFACTS_DIR\s*/\s*'controls'\s*/\s*ARTIFACTS_DIR\s*/\s*'controls'\s*/\s*", "ARTIFACTS_DIR / 'controls' / ", updated)
    updated = re.sub(r"DATA_DIR\s*/\s*'PPI_Pairs'\s*/\s*DATA_DIR\s*/\s*'PPI_Pairs'\s*/\s*", "DATA_DIR / 'PPI_Pairs' / ", updated)
    updated = re.sub(r"DATA_DIR\s*/\s*DATA_DIR\s*/\s*", "DATA_DIR / ", updated)
    updated = re.sub(r"DATA_DIR\s*/\s*'DepMap_data'\s*/\s*'DepMap_data'\s*/\s*", "DATA_DIR / 'DepMap_data' / ", updated)
    updated = re.sub(r"DATA_DIR\s*/\s*'Trimmed data'\s*/\s*'Trimmed data'\s*/\s*", "DATA_DIR / 'Trimmed data' / ", updated)
    return updated


def _replace_shared_function_cell(source: str) -> str:
    function_names = set(FUNCTION_PATTERN.findall(source))
    if function_names and function_names.issubset(SHARED_FUNCTIONS):
        return (
            "# Shared helper functions now live in src/pdm_learn.\n"
            "# See the project setup cell at the top of this notebook for imports.\n"
        )
    return source


def _is_legacy_setup_cell(source: str) -> bool:
    return (
        source.startswith("# Shared project setup")
        and "SRC_DIR = PROJECT_ROOT / 'src'" in source
        and ("from ccle_ml." in source or "from pdm_learn." in source)
    )


def _ensure_setup_cell(notebook: dict) -> bool:
    cells = notebook.setdefault("cells", [])
    if cells and cells[0].get("cell_type") == "code":
        source = "".join(cells[0].get("source", []))
        if source.startswith("# Shared project setup for imports and file locations"):
            cells[0]["source"] = SETUP_CELL_SOURCE
            changed = False
        else:
            changed = True
            setup_cell = {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": SETUP_CELL_SOURCE,
            }
            cells.insert(0, setup_cell)
    else:
        changed = True
        setup_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": SETUP_CELL_SOURCE,
        }
        cells.insert(0, setup_cell)

    filtered = [cells[0]]
    removed = False
    for cell in cells[1:]:
        source = "".join(cell.get("source", [])) if cell.get("cell_type") == "code" else ""
        if _is_legacy_setup_cell(source):
            removed = True
            continue
        filtered.append(cell)
    notebook["cells"] = filtered
    return changed or removed


def refactor_notebook(path: Path) -> bool:
    with path.open() as handle:
        notebook = json.load(handle)

    changed = _ensure_setup_cell(notebook)

    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        original_source = "".join(cell.get("source", []))
        if index == 0:
            continue
        updated_source = _replace_shared_function_cell(original_source)
        updated_source = updated_source.replace("ccle_ml.", "pdm_learn.")
        updated_source = updated_source.replace("resolve_legacy_path", "resolve_path")
        updated_source = updated_source.replace("src/ccle_ml", "src/pdm_learn")
        updated_source = _replace_resolved_paths(updated_source)
        updated_source = _normalize_to_csv_calls(updated_source)
        updated_source = _normalize_path_expressions(updated_source)
        if updated_source != original_source:
            cell["source"] = updated_source.splitlines(keepends=True)
            changed = True

    if changed:
        with path.open("w") as handle:
            json.dump(notebook, handle, indent=1)
            handle.write("\n")
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description="Refactor notebooks to use the shared pdm_learn package.")
    parser.add_argument("notebooks", nargs="*", type=Path, help="Specific notebooks or folders to rewrite.")
    args = parser.parse_args()

    if args.notebooks:
        notebook_paths = []
        for path in args.notebooks:
            if path.is_dir():
                notebook_paths.extend(sorted(path.rglob("*.ipynb")))
            else:
                notebook_paths.append(path)
    else:
        notebook_paths = NOTEBOOKS

    changed_paths = [path for path in notebook_paths if refactor_notebook(path)]

    if changed_paths:
        print("Updated notebooks:")
        for path in changed_paths:
            print(f" - {path.relative_to(PROJECT_ROOT)}")
    else:
        print("No notebook changes were needed.")


if __name__ == "__main__":
    main()
