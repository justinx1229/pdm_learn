"""PDM Learn: general feature learning from aligned matrix datasets.

For new work, start with :class:`DatasetSpec`, :class:`FeaturePair`,
:func:`load_matrices`, :func:`build_features`, and :func:`rank_candidates`.
Domain-specific DepMap recipes are available below :mod:`pdm_learn.workflows`.
Legacy ``oncogene`` and ``ppi`` modules remain import-compatible.
"""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "datasets": [
        "DatasetSpec",
        "align_matrices",
        "load_matrices",
        "matrix_id_column",
        "matrix_sample_columns",
        "prepare_matrix",
        "resolve_dataset_specs",
    ],
    "features": [
        "DatasetPair",
        "FEATURE_METHODS",
        "FeatureConfig",
        "FeaturePair",
        "STATISTIC_METHODS",
        "bicor",
        "build_density_features",
        "build_feature_tables",
        "build_features",
        "build_statistic_features",
        "compute_mutual_information",
        "load_gene_matrices",
        "load_labels",
        "rank_labeled_genes",
        "split_labeled_feature_tables",
    ],
    "labels": [
        "balance_labeled_rows",
        "load_label_set",
        "split_labeled_rows",
    ],
    "pair_features": [
        "PAIR_FEATURE_METHODS",
        "PairFeatureConfig",
        "build_pair_density_features",
        "build_pair_features",
        "build_pair_statistic_features",
    ],
    "modeling": [
        "KFold_PR",
        "KFold_ROC_AUC",
        "KFold_ROC_AUC_grouped_plot",
        "LOOCV",
        "LOOCV_grouped_plot",
        "area_table",
        "core_predict",
        "heatmap",
        "importance_test",
        "ks_pvalue",
        "evaluate_labeled_features",
        "rank_candidates",
        "rank_unlabeled",
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
    "presets": [
        "DEPMAP_ONCOGENE",
        "DEPMAP_PPI",
        "depmap_oncogene",
        "depmap_ppi",
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
        "parse_gene_list",
        "ppi_density_parameters",
        "ppi_gene_universe",
        "rank_ppi_pairs_cancer",
        "reverse_feature_source_columns",
        "reverse_pair_table",
        "sample_negative_control_pairs",
        "sample_positive_control_pairs",
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
        "collect_simulated_pairs",
        "eps",
        "iter_simulated_pairs",
        "partition",
        "perturb_pair",
        "standardize_pair",
    ],
}

__version__ = "0.3.0"

# Star imports intentionally expose the reusable center, not a particular
# scientific workflow.  ``__getattr__`` below still resolves historical
# oncogene/PPI names for compatibility with existing callers.
_PUBLIC_MODULES = ("datasets", "features", "labels", "pair_features", "modeling", "simulation")
__all__ = [name for module_name in _PUBLIC_MODULES for name in _EXPORTS[module_name]]
__all__.append("__version__")


def __getattr__(name: str):
    for module_name, exports in _EXPORTS.items():
        if name in exports:
            module = import_module(f"pdm_learn.{module_name}")
            value = getattr(module, name)
            globals()[name] = value
            return value
    raise AttributeError(f"module 'pdm_learn' has no attribute {name!r}")
