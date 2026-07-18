"""Domain recipes built on top of the generic :mod:`pdm_learn` API.

The core package deliberately contains data preparation, feature construction,
labels, and modeling without a disease or assay vocabulary.  Import a module
from this namespace when a maintained, opinionated recipe is useful:

.. code-block:: python

    from pdm_learn.workflows import oncogene

    datasets = oncogene.load_inputs("data")
    features = oncogene.build_features(datasets)

Submodules are loaded lazily so importing :mod:`pdm_learn.workflows` does not
load optional plotting or modeling dependencies.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


__all__ = ["oncogene", "ppi"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
