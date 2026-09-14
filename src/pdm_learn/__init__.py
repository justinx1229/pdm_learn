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
        "DEFAULT_ONCOGENE_DATASET_SPECS",
        "DEFAULT_ONCOGENE_PAIR_SPECS",
        "ONCOGENE_HEATMAP_DIMENSIONS",
        "ONCOGENE_PAIRS",
        "GeneDatasetSpec",
        "GenePairSpec",
        "OncogeneDatasetSpec",
        "OncogenePairSpec",
        "bicor",
        "build_gene_correlation_features",
        "build_gene_density_features",
        "build_gene_statistic_features",
        "build_oncogene_correlation_features",
        "build_oncogene_density_features",
        "build_oncogene_statistic_features",
        "compute_mutual_information",
        "evaluate_method_curves",
        "load_gene_feature_sets",
        "load_oncogene_feature_sets",
        "load_oncogene_inputs",
        "oncogene_gene_list",
        "plot_method_curves",
        "rank_candidate_genes",
        "rank_candidate_oncogenes",
        "save_gene_feature_tables",
        "save_oncogene_feature_tables",
        "SHARED_TRIMMED_FILENAMES",
        "snap_copy_number_levels",
        "standardize_oncogene_matrix",
        "standardize_oncogene_mutations",
        "trim_oncogene_input_tables",
        "load_shared_trimmed_oncogene_inputs",
    ],
    "ppi": [
        "build_control_feature_tables",
        "build_ppi_feature_table",
        "canonical_pair",
        "cancer_complex_pair_set",
        "derive_shared_trimmed_inputs",
        "evaluate_ppi_controls",
        "load_biogrid_pairs",
        "load_cancer_complexes",
        "load_ppi_inputs",
        "rank_ppi_pairs_cancer",
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
