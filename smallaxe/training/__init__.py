"""Training: models, the algorithm records, and the factory modules.

Get a model from a factory module: ``regressors.<algorithm>()`` returns a
:class:`Regressor`, ``classifiers.<algorithm>(task=...)`` a :class:`Classifier`, both
:class:`Model` bound to one record of :mod:`smallaxe.training.algorithm`.

Importing it imports no optional package. The algorithm modules hold only declarative
records; xgboost.spark, synapse.ml.lightgbm, and catboost_spark are imported when an
algorithm is first checked or used, and the two JVM packages, which both patch
pyspark.ml.wrapper process-wide, always in the order ``algorithm._PATCHES_PYSPARK``
fixes.

Example:
    >>> from smallaxe.training import classifiers
    >>> model = classifiers.xgboost(task="binary", n_estimators=200)
"""

from smallaxe.training import algorithm, classifiers, regressors
from smallaxe.training.base import Classifier, Model, Regressor

__all__ = [
    "algorithm",
    "Model",
    "Regressor",
    "Classifier",
    "regressors",
    "classifiers",
]
