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
    from smallaxe.training.xgboost import (
        XGBoostClassifier as XGBoostClassifier,
    )
    from smallaxe.training.xgboost import (
        XGBoostRegressor as XGBoostRegressor,
    )

    __all__.extend(["XGBoostRegressor", "XGBoostClassifier"])
except ImportError:
    pass

# Import LightGBM classes if available (optional dependency)
try:
    from smallaxe.training.lightgbm import (
        LightGBMClassifier as LightGBMClassifier,
    )
    from smallaxe.training.lightgbm import (
        LightGBMRegressor as LightGBMRegressor,
    )

    __all__.extend(["LightGBMRegressor", "LightGBMClassifier"])
except ImportError:
    pass
