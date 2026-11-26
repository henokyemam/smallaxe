"""Search module - hyperparameter optimization.

Exposes :mod:`smallaxe.search.optimize`, a hyperopt-backed hyperparameter
search. hyperopt is a required dependency of smallaxe; without it the module still
imports, and :func:`smallaxe.search.optimize.run` raises a clear
:class:`~smallaxe.exceptions.DependencyError`.

Example:
    >>> from smallaxe.search import optimize
    >>> result = optimize.run(model, dataframe, label_col="target", param_space=space)
"""

from smallaxe import _deprecation
from smallaxe.search import optimize
from smallaxe.search.optimize import SearchResult

__all__ = ["optimize", "SearchResult"]

# Deprecated public names, removed in smallaxe 1.0.
_MOVED = {
    "HYPEROPT_AVAILABLE": (
        "hyperopt is a required dependency, so the flag is always True; drop the check.",
        lambda: optimize._HYPEROPT_AVAILABLE,
    ),
}
__getattr__ = _deprecation.module_getattr(__name__, _MOVED)
__dir__ = _deprecation.module_dir(globals(), _MOVED)
