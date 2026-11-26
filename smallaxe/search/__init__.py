"""Search: hyperparameter optimization.

Exposes :mod:`smallaxe.search.optimize`, a hyperopt-backed hyperparameter search, and
its result type :class:`SearchResult`. hyperopt is a required dependency of smallaxe.

Example:
    >>> from smallaxe.search import optimize
    >>> result = optimize.run(model, dataframe, label_col="target", param_space=space)
"""

from smallaxe.search import optimize
from smallaxe.search.optimize import SearchResult

__all__ = ["optimize", "SearchResult"]
