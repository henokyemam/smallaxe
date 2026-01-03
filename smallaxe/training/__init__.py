"""Training module - model classes and factories."""

from smallaxe.training.base import BaseClassifier, BaseModel, BaseRegressor
from smallaxe.training.classifiers import Classifiers
from smallaxe.training.random_forest import RandomForestClassifier, RandomForestRegressor
from smallaxe.training.regressors import Regressors

__all__ = [
    "BaseModel",
    "BaseRegressor",
    "BaseClassifier",
    "RandomForestRegressor",
    "RandomForestClassifier",
    "Regressors",
    "Classifiers",
]

# Import XGBoost classes if available (optional dependency)
try:
    from smallaxe.training.xgboost import XGBoostClassifier, XGBoostRegressor

    __all__.extend(["XGBoostRegressor", "XGBoostClassifier"])
except ImportError:
    pass
