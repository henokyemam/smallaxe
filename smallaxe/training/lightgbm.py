"""LightGBM via SynapseML (``pip install smallaxe[lightgbm]`` plus the Spark package).

smallaxe supports LightGBM on Spark 3.x only, where SynapseML publishes Scala 2.12 builds.
On Databricks the ``com.microsoft.azure:synapseml-lightgbm_2.12`` Maven library also puts
the ``synapse.ml`` Python package on the path. SynapseML's Spark 4.0 and 4.1 builds
(``synapseml_2.13:<version>-spark4.0`` / ``-spark4.1``) are not validated with smallaxe;
nothing here blocks them.
"""

from typing import Any, List, Optional

from smallaxe.training.algorithm import Algorithm, Dependency, Param

__all__ = ["ALGORITHM"]


def _importances(model: Any, n_features: int) -> Optional[List[float]]:
    """Split counts per feature (SynapseML's default importance type)."""
    return [float(v) for v in model.getFeatureImportances()]


ALGORITHM = Algorithm(
    name="lightgbm",
    params=(
        Param("n_estimators", "Number of boosting iterations", 100, spark="numIterations"),
        Param("max_depth", "Maximum depth of each tree (-1 for no limit)", -1, spark="maxDepth"),
        Param("learning_rate", "Boosting learning rate", 0.1, spark="learningRate"),
        Param("num_leaves", "Maximum number of leaves in one tree", 31, spark="numLeaves"),
        Param(
            "min_data_in_leaf",
            "Minimum number of data points in a leaf",
            20,
            spark="minDataInLeaf",
        ),
        Param(
            "feature_fraction",
            "Fraction of features used for training each tree",
            1.0,
            spark="featureFraction",
        ),
        Param(
            "bagging_fraction",
            "Fraction of data used for training each tree",
            1.0,
            spark="baggingFraction",
        ),
        Param(
            "bagging_freq",
            "Frequency for bagging (0 means disable bagging)",
            0,
            spark="baggingFreq",
        ),
        Param("lambda_l1", "L1 regularization term on weights", 0.0, spark="lambdaL1"),
        Param("lambda_l2", "L2 regularization term on weights", 0.0, spark="lambdaL2"),
        Param("seed", "Random seed for reproducibility", None, setter="setSeed"),
    ),
    estimators={
        "simple_regression": "synapse.ml.lightgbm.LightGBMRegressor",
        "binary": "synapse.ml.lightgbm.LightGBMClassifier",
        "multiclass": "synapse.ml.lightgbm.LightGBMClassifier",
    },
    models={
        "simple_regression": "synapse.ml.lightgbm.LightGBMRegressionModel",
        "binary": "synapse.ml.lightgbm.LightGBMClassificationModel",
        "multiclass": "synapse.ml.lightgbm.LightGBMClassificationModel",
    },
    cols={
        "features": "featuresCol",
        "label": "labelCol",
        "prediction": "predictionCol",
        "probability": "probabilityCol",
        "raw_prediction": "rawPredictionCol",
    },
    # SynapseML defaults to "binary" even for >2 classes, which silently trains a
    # two-class model on multiclass labels.
    task_params={
        "binary": {"objective": "binary"},
        "multiclass": {"objective": "multiclass"},
    },
    importances=_importances,
    # SynapseML LightGBM defaults to ``useMissing=true``.
    accepts_missing=True,
    dependency=Dependency(
        "synapseml",
        "pip install smallaxe[lightgbm] and configure Spark with the SynapseML package",
    ),
)
