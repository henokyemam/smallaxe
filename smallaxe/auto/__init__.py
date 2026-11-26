"""Automated training: every available algorithm on the same rows, compared on one metric.

Exposes :class:`AutomatedTraining`, which trains each algorithm through its factory (and
``optimize.run`` for the ones given a search space), scores them all on the same
validation rows, and keeps the best fitted model.

Example:
    >>> from smallaxe.auto import AutomatedTraining
    >>> auto = AutomatedTraining(task="binary", seed=42).fit(dataframe, label_col="label")
    >>> auto.comparison
    >>> auto.best_model.save("/models/churn")
"""

from smallaxe.auto._automated import AutomatedTraining

__all__ = ["AutomatedTraining"]
