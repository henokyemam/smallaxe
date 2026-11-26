"""Plotly charts for evaluating smallaxe models on Spark DataFrames of any size.

Every chart aggregates in Spark and collects only the aggregated rows, bounded by its
bins or classes and never by the number of rows: a confusion matrix is one
``groupBy(label, prediction).count()``, a curve or histogram one ``groupBy`` over score
or value bins. Each function returns a ``plotly.graph_objects.Figure`` and never calls
``show()``.

plotly is optional: install it with ``pip install smallaxe[viz]``. Importing this module
does not import plotly; calling a chart without it raises ``DependencyError``.

Example:
    >>> from smallaxe import viz
    >>> predictions = model.predict_proba(test)
    >>> viz.roc_curve(predictions, label_col="churn").show()
    >>> viz.confusion_matrix(predictions, label_col="churn", normalize=True)
    >>> viz.feature_importances(model, top_n=15)
"""

from smallaxe.viz._classification import confusion_matrix, precision_recall_curve, roc_curve
from smallaxe.viz._models import feature_importances, model_comparison
from smallaxe.viz._regression import actual_vs_predicted, residuals

__all__ = [
    "confusion_matrix",
    "roc_curve",
    "precision_recall_curve",
    "residuals",
    "actual_vs_predicted",
    "feature_importances",
    "model_comparison",
]
