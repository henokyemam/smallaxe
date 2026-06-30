"""Search module - hyperparameter optimization.

Exposes :mod:`smallaxe.search.optimize`, a hyperopt-backed hyperparameter
search. hyperopt is an optional dependency; the module imports cleanly without
it and raises a clear :class:`~smallaxe.exceptions.DependencyError` only when
:func:`smallaxe.search.optimize.run` is actually called.

Example:
    >>> from smallaxe.search import optimize
    >>> result = optimize.run(model, df, label_col="target", param_space=space)
"""

from smallaxe.search import optimize
from smallaxe.search.optimize import HYPEROPT_AVAILABLE, SearchResult

__all__ = ["optimize", "SearchResult", "HYPEROPT_AVAILABLE"]
