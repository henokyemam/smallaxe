"""Metrics module - regression and classification metrics."""

from smallaxe.metrics.classification import (
    accuracy,
    auc_pr,
    auc_roc,
    f1_score,
    log_loss,
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

__all__ = [
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
]
