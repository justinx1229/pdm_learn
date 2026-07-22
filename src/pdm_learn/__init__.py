"""PDM Learn: generic feature learning from aligned matrix datasets.

Start with :class:`DatasetSpec`, :class:`FeaturePair`,
:func:`load_matrices`, :func:`build_features`, and :func:`rank_candidates`.
Optional domain recipes live in :mod:`pdm_learn.workflows`.
"""

from .datasets import DatasetSpec, align_matrices, load_matrices, prepare_matrix
from .features import (
    FEATURE_METHODS,
    STATISTIC_METHODS,
    FeatureConfig,
    FeaturePair,
    bicor,
    build_density_features,
    build_features,
    build_statistic_features,
    compute_mutual_information,
    density_map,
)
from .labels import balance_labeled_rows, load_label_set, split_labeled_rows
from .modeling import (
    MODEL_NAMES,
    evaluate_labeled_features,
    feature_ks_pvalues,
    rank_candidates,
    rank_unlabeled,
    select_features,
)
from .pair_features import (
    PAIR_FEATURE_METHODS,
    PairFeatureConfig,
    build_pair_density_features,
    build_pair_features,
    build_pair_statistic_features,
)

__version__ = "1.0.0"

__all__ = [
    "FEATURE_METHODS",
    "MODEL_NAMES",
    "PAIR_FEATURE_METHODS",
    "STATISTIC_METHODS",
    "DatasetSpec",
    "FeatureConfig",
    "FeaturePair",
    "PairFeatureConfig",
    "align_matrices",
    "balance_labeled_rows",
    "bicor",
    "build_density_features",
    "build_features",
    "build_pair_density_features",
    "build_pair_features",
    "build_pair_statistic_features",
    "build_statistic_features",
    "compute_mutual_information",
    "density_map",
    "evaluate_labeled_features",
    "feature_ks_pvalues",
    "load_label_set",
    "load_matrices",
    "prepare_matrix",
    "rank_candidates",
    "rank_unlabeled",
    "select_features",
    "split_labeled_rows",
]
