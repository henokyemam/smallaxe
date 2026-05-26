"""LightGBM models for regression and classification."""

from typing import Any, Dict, Optional

from smallaxe.exceptions import DependencyError
from smallaxe.training.base import BaseClassifier, BaseRegressor

# Check for LightGBM availability (via SynapseML)
try:
    from synapse.ml.lightgbm import (
        LightGBMClassifier as SparkLightGBMClassifier,
    )
    from synapse.ml.lightgbm import (
        LightGBMClassifierModel as SparkLightGBMClassifierModel,
    )
    from synapse.ml.lightgbm import (
        LightGBMRegressor as SparkLightGBMRegressor,
    )
    from synapse.ml.lightgbm import (
        LightGBMRegressorModel as SparkLightGBMRegressorModel,
    )

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    SparkLightGBMRegressor = None
    SparkLightGBMRegressorModel = None
    SparkLightGBMClassifier = None
    SparkLightGBMClassifierModel = None


def _check_lightgbm_available() -> None:
    """Check if LightGBM is available and raise DependencyError if not."""
    if not LIGHTGBM_AVAILABLE:
        raise DependencyError(
            package="synapseml",
            install_command=(
                "pyspark --packages com.microsoft.azure:synapseml_2.12:1.1.0 "
                "--repositories https://mmlspark.azureedge.net/maven"
            ),
        )


class LightGBMRegressor(BaseRegressor):
    """LightGBM Regressor for regression tasks.

    This class wraps SynapseML's LightGBMRegressor to provide
    a scikit-learn-like interface with support for train/test and k-fold
    cross-validation.

    Note:
        This requires SynapseML (v1.1.0+) which provides LightGBM integration
        for Spark. Requires Scala 2.12, Spark 3.2+, and Python 3.8+.

        **Standalone Spark (Maven)**::

            pyspark --packages com.microsoft.azure:synapseml_2.12:1.1.0 \\
                --repositories https://mmlspark.azureedge.net/maven

        **Databricks**: Add Maven library to cluster with coordinates
        ``com.microsoft.azure:synapseml_2.12:1.1.0`` and repository
        ``https://mmlspark.azureedge.net/maven``

    Args:
        task: The regression task type. Default is 'simple_regression'.

    Example:
        >>> from smallaxe.training import LightGBMRegressor
        >>> model = LightGBMRegressor()
        >>> model.set_param({"n_estimators": 100, "max_depth": 6})
        >>> model.fit(df, label_col='target', feature_cols=['f1', 'f2'])
        >>> predictions = model.predict(df)

    Raises:
        DependencyError: If synapseml is not installed.
    """

    def __init__(self, task: str = "simple_regression") -> None:
        """Initialize the LightGBM regressor.

        Args:
            task: The regression task type.

        Raises:
            DependencyError: If synapseml is not installed.
            ValidationError: If task is not a valid regression task.
        """
        _check_lightgbm_available()
        super().__init__(task)

    @property
    def params(self) -> Dict[str, str]:
        """Get parameter descriptions.

        Returns:
            Dictionary mapping parameter names to their descriptions.
        """
        return {
            "n_estimators": "Number of boosting iterations",
            "max_depth": "Maximum depth of each tree (-1 for no limit)",
            "learning_rate": "Boosting learning rate",
            "num_leaves": "Maximum number of leaves in one tree",
            "min_data_in_leaf": "Minimum number of data points in a leaf",
            "feature_fraction": "Fraction of features used for training each tree",
            "bagging_fraction": "Fraction of data used for training each tree",
            "bagging_freq": "Frequency for bagging (0 means disable bagging)",
            "lambda_l1": "L1 regularization term on weights",
            "lambda_l2": "L2 regularization term on weights",
            "seed": "Random seed for reproducibility",
        }

    @property
    def default_params(self) -> Dict[str, Any]:
        """Get default parameter values.

        Returns:
            Dictionary mapping parameter names to their default values.
        """
        return {
            "n_estimators": 100,
            "max_depth": -1,
            "learning_rate": 0.1,
            "num_leaves": 31,
            "min_data_in_leaf": 20,
            "feature_fraction": 1.0,
            "bagging_fraction": 1.0,
            "bagging_freq": 0,
            "lambda_l1": 0.0,
            "lambda_l2": 0.0,
            "seed": None,
        }

    def _uses_constructor_col_params(self) -> bool:
        return True

    def _create_spark_estimator(
        self,
        features_col: Optional[str] = None,
        label_col: Optional[str] = None,
        prediction_col: Optional[str] = None,
    ) -> Any:
        """Create the underlying SparkLightGBMRegressor.

        Returns:
            Configured SparkLightGBMRegressor instance.
        """
        seed = self.get_param("seed")

        kwargs = {
            "numIterations": self.get_param("n_estimators"),
            "maxDepth": self.get_param("max_depth"),
            "learningRate": self.get_param("learning_rate"),
            "numLeaves": self.get_param("num_leaves"),
            "minDataInLeaf": self.get_param("min_data_in_leaf"),
            "featureFraction": self.get_param("feature_fraction"),
            "baggingFraction": self.get_param("bagging_fraction"),
            "baggingFreq": self.get_param("bagging_freq"),
            "lambdaL1": self.get_param("lambda_l1"),
            "lambdaL2": self.get_param("lambda_l2"),
        }

        if features_col is not None:
            kwargs["featuresCol"] = features_col
        if label_col is not None:
            kwargs["labelCol"] = label_col
        if prediction_col is not None:
            kwargs["predictionCol"] = prediction_col

        estimator = SparkLightGBMRegressor(**kwargs)

        if seed is not None:
            estimator.setSeed(seed)

        return estimator

    def _load_artifacts(self, path: str) -> None:
        """Load the Spark model from disk.

        Args:
            path: Directory path where the model is saved.
        """
        self._load_spark_model(path, SparkLightGBMRegressorModel)


class LightGBMClassifier(BaseClassifier):
    """LightGBM Classifier for classification tasks.

    This class wraps SynapseML's LightGBMClassifier to provide
    a scikit-learn-like interface with support for train/test and k-fold
    cross-validation, including stratified sampling for classification.

    Note:
        This requires SynapseML (v1.1.0+) which provides LightGBM integration
        for Spark. Requires Scala 2.12, Spark 3.2+, and Python 3.8+.

        **Standalone Spark (Maven)**::

            pyspark --packages com.microsoft.azure:synapseml_2.12:1.1.0 \\
                --repositories https://mmlspark.azureedge.net/maven

        **Databricks**: Add Maven library to cluster with coordinates
        ``com.microsoft.azure:synapseml_2.12:1.1.0`` and repository
        ``https://mmlspark.azureedge.net/maven``

    Args:
        task: The classification task type. Options are 'binary' or 'multiclass'.
            Default is 'binary'.

    Example:
        >>> from smallaxe.training import LightGBMClassifier
        >>> model = LightGBMClassifier(task='binary')
        >>> model.set_param({"n_estimators": 100, "max_depth": 6})
        >>> model.fit(df, label_col='label', feature_cols=['f1', 'f2'])
        >>> predictions = model.predict(df)
        >>> probabilities = model.predict_proba(df)

    Raises:
        DependencyError: If synapseml is not installed.
    """

    def __init__(self, task: str = "binary") -> None:
        """Initialize the LightGBM classifier.

        Args:
            task: The classification task type.

        Raises:
            DependencyError: If synapseml is not installed.
            ValidationError: If task is not a valid classification task.
        """
        _check_lightgbm_available()
        super().__init__(task)

    @property
    def params(self) -> Dict[str, str]:
        """Get parameter descriptions.

        Returns:
            Dictionary mapping parameter names to their descriptions.
        """
        return {
            "n_estimators": "Number of boosting iterations",
            "max_depth": "Maximum depth of each tree (-1 for no limit)",
            "learning_rate": "Boosting learning rate",
            "num_leaves": "Maximum number of leaves in one tree",
            "min_data_in_leaf": "Minimum number of data points in a leaf",
            "feature_fraction": "Fraction of features used for training each tree",
            "bagging_fraction": "Fraction of data used for training each tree",
            "bagging_freq": "Frequency for bagging (0 means disable bagging)",
            "lambda_l1": "L1 regularization term on weights",
            "lambda_l2": "L2 regularization term on weights",
            "seed": "Random seed for reproducibility",
        }

    @property
    def default_params(self) -> Dict[str, Any]:
        """Get default parameter values.

        Returns:
            Dictionary mapping parameter names to their default values.
        """
        return {
            "n_estimators": 100,
            "max_depth": -1,
            "learning_rate": 0.1,
            "num_leaves": 31,
            "min_data_in_leaf": 20,
            "feature_fraction": 1.0,
            "bagging_fraction": 1.0,
            "bagging_freq": 0,
            "lambda_l1": 0.0,
            "lambda_l2": 0.0,
            "seed": None,
        }

    def _uses_constructor_col_params(self) -> bool:
        return True

    def _create_spark_estimator(
        self,
        features_col: Optional[str] = None,
        label_col: Optional[str] = None,
        prediction_col: Optional[str] = None,
    ) -> Any:
        """Create the underlying SparkLightGBMClassifier.

        Returns:
            Configured SparkLightGBMClassifier instance.
        """
        seed = self.get_param("seed")

        kwargs = {
            "numIterations": self.get_param("n_estimators"),
            "maxDepth": self.get_param("max_depth"),
            "learningRate": self.get_param("learning_rate"),
            "numLeaves": self.get_param("num_leaves"),
            "minDataInLeaf": self.get_param("min_data_in_leaf"),
            "featureFraction": self.get_param("feature_fraction"),
            "baggingFraction": self.get_param("bagging_fraction"),
            "baggingFreq": self.get_param("bagging_freq"),
            "lambdaL1": self.get_param("lambda_l1"),
            "lambdaL2": self.get_param("lambda_l2"),
            "probabilityCol": self.PROBABILITY_COL,
            "rawPredictionCol": self.RAW_PREDICTION_COL,
        }

        if features_col is not None:
            kwargs["featuresCol"] = features_col
        if label_col is not None:
            kwargs["labelCol"] = label_col
        if prediction_col is not None:
            kwargs["predictionCol"] = prediction_col

        estimator = SparkLightGBMClassifier(**kwargs)

        if seed is not None:
            estimator.setSeed(seed)

        return estimator

    def _load_artifacts(self, path: str) -> None:
        """Load the Spark model from disk.

        Args:
            path: Directory path where the model is saved.
        """
        self._load_spark_model(path, SparkLightGBMClassifierModel)
