from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "modeling": [
        "KFold_PR",
        "LOOCV",
        "LOOCV_grouped_plot",
        "area_table",
        "core_predict",
        "heatmap",
        "importance_test",
        "ks_pvalue",
    ],
    "oncogene": [
        "COPY_NUMBER_LEVELS",
        "ONCOGENE_HEATMAP_DIMENSIONS",
        "ONCOGENE_PAIRS",
        "bicor",
        "build_oncogene_correlation_features",
        "build_oncogene_density_features",
        "build_oncogene_statistic_features",
        "compute_mutual_information",
        "evaluate_method_curves",
        "load_oncogene_feature_sets",
        "load_oncogene_inputs",
        "oncogene_gene_list",
        "plot_method_curves",
        "rank_candidate_oncogenes",
        "save_oncogene_feature_tables",
        "snap_copy_number_levels",
    ],
    "preprocessing": [
        "build_density_map",
        "density_centers",
        "densitymap",
        "drop_nan",
        "extract",
        "mut_trim",
        "normalize",
        "trim",
        "trim_pairs",
    ],
    "simulation": [
        "build_heatmap_dataset",
        "build_metric_dataset",
        "clip_pair_to_centers",
        "eps",
        "iter_simulated_pairs",
        "partition",
        "perturb_pair",
        "standardize_pair",
    ],
}

__all__ = [name for names in _EXPORTS.values() for name in names]


def __getattr__(name: str):
    for module_name, exports in _EXPORTS.items():
        if name in exports:
            module = import_module(f"pdm_learn.{module_name}")
            value = getattr(module, name)
            globals()[name] = value
            return value
    raise AttributeError(f"module 'pdm_learn' has no attribute {name!r}")
