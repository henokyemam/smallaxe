"""Metrics module - regression and classification metrics."""

from typing import Dict, FrozenSet

from smallaxe.metrics.classification import (
    MULTICLASS_AVERAGED_METRICS,
    accuracy,
    auc_pr,
    auc_roc,
    f1_score,
    log_loss,
    multiclass_scores,
    precision,
    recall,
)
from smallaxe.metrics.regression import (
    mae,
    mape,
    mse,
    r2,
    rmse,
)

# The scores a model's validation produces for each task.
TASK_METRICS: Dict[str, FrozenSet[str]] = {
    "simple_regression": frozenset({"mse", "rmse", "mae", "r2", "mape"}),
    "binary": frozenset(
        {"accuracy", "precision", "recall", "f1_score", "auc_roc", "auc_pr", "log_loss"}
    ),
    "multiclass": frozenset({"accuracy", "precision", "recall", "f1_score"})
    | MULTICLASS_AVERAGED_METRICS,
}

__all__ = [
    "TASK_METRICS",
    # Regression metrics
    "mse",
    "rmse",
    "mae",
    "r2",
    "mape",
    # Classification metrics
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "auc_roc",
    "auc_pr",
    "log_loss",
    "multiclass_scores",
]
