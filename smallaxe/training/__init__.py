"""Training module - model classes, algorithm records, and factories."""

from smallaxe.training import algorithm
from smallaxe.training.base import (
    BaseClassifier,
    BaseModel,
    BaseRegressor,
    Classifier,
    Model,
    Regressor,
)

# The CatBoost record is imported before LightGBM's on purpose: both JVM packages
# patch pyspark.ml.wrapper.JavaParams process-wide, and the CatBoost compatibility
# shim assumes SynapseML's patch was applied last.
from smallaxe.training.catboost import CatBoostClassifier, CatBoostRegressor
from smallaxe.training.classifiers import Classifiers
from smallaxe.training.lightgbm import LightGBMClassifier, LightGBMRegressor
from smallaxe.training.random_forest import RandomForestClassifier, RandomForestRegressor
from smallaxe.training.regressors import Regressors
from smallaxe.training.xgboost import XGBoostClassifier, XGBoostRegressor

__all__ = [
    "algorithm",
    "BaseModel",
    "BaseRegressor",
    "BaseClassifier",
    "Model",
    "Regressor",
    "Classifier",
    "Regressors",
    "Classifiers",
    # Deprecated 0.8.x class aliases, removed in 1.0.
    "RandomForestRegressor",
    "RandomForestClassifier",
    "XGBoostRegressor",
    "XGBoostClassifier",
    "LightGBMRegressor",
    "LightGBMClassifier",
    "CatBoostRegressor",
    "CatBoostClassifier",
]
