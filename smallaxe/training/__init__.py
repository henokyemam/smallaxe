"""Training module - model classes, algorithm records, and factories.

Importing it imports no optional package. The algorithm modules below hold only
declarative records; xgboost.spark, synapse.ml.lightgbm, and catboost_spark are
imported when an algorithm is first checked or used, and the two JVM packages,
which both patch pyspark.ml.wrapper process-wide, always in the order
``algorithm._PATCHES_PYSPARK`` fixes.
"""

from smallaxe.training import algorithm
from smallaxe.training.base import (
    BaseClassifier,
    BaseModel,
    BaseRegressor,
    Classifier,
    Model,
    Regressor,
)
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
