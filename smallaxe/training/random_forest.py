"""Random Forest: PySpark MLlib's native estimators."""

from typing import Any, List, Optional

from smallaxe.training.algorithm import Algorithm, Param
from smallaxe.training.base import Classifier, Regressor, _warn_deprecated_alias


def _importances(model: Any, n_features: int) -> Optional[List[float]]:
    """MLlib's Gini-based importances, normalised to sum to 1."""
    return list(model.featureImportances.toArray())


ALGORITHM = Algorithm(
    name="random_forest",
    params=(
        Param("n_estimators", "Number of trees in the forest", 20, spark="numTrees"),
        Param("max_depth", "Maximum depth of each tree (0 = unlimited)", 5, spark="maxDepth"),
        Param(
            "max_bins",
            "Maximum number of bins for discretizing continuous features",
            32,
            spark="maxBins",
        ),
        Param(
            "min_instances_per_node",
            "Minimum number of instances per node",
            1,
            spark="minInstancesPerNode",
        ),
        Param("min_info_gain", "Minimum information gain for a split", 0.0, spark="minInfoGain"),
        Param(
            "subsampling_rate",
            "Fraction of data used for training each tree",
            1.0,
            spark="subsamplingRate",
        ),
        Param(
            "feature_subset_strategy",
            "Strategy for selecting features: 'auto', 'all', 'sqrt', 'log2', 'onethird'",
            "auto",
            spark="featureSubsetStrategy",
        ),
        Param("seed", "Random seed for reproducibility", None, setter="setSeed"),
    ),
    estimators={
        "simple_regression": "pyspark.ml.regression.RandomForestRegressor",
        "binary": "pyspark.ml.classification.RandomForestClassifier",
        "multiclass": "pyspark.ml.classification.RandomForestClassifier",
    },
    models={
        "simple_regression": "pyspark.ml.regression.RandomForestRegressionModel",
        "binary": "pyspark.ml.classification.RandomForestClassificationModel",
        "multiclass": "pyspark.ml.classification.RandomForestClassificationModel",
    },
    importances=_importances,
)


class RandomForestRegressor(Regressor):
    """Deprecated alias; use ``Regressors.random_forest()``."""

    _alias_algorithm = "random_forest"

    def __init__(self, task: str = "simple_regression") -> None:
        _warn_deprecated_alias("RandomForestRegressor", "Regressors.random_forest()")
        super().__init__(ALGORITHM, task)


class RandomForestClassifier(Classifier):
    """Deprecated alias; use ``Classifiers.random_forest()``."""

    _alias_algorithm = "random_forest"

    def __init__(self, task: str = "binary") -> None:
        _warn_deprecated_alias("RandomForestClassifier", "Classifiers.random_forest()")
        super().__init__(ALGORITHM, task)
