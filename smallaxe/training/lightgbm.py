"""LightGBM models for regression and classification."""

from typing import Any, Dict, List

from pyspark.sql import DataFrame

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

    def _create_spark_estimator(self) -> Any:
        """Create the underlying SparkLightGBMRegressor.

        Returns:
            Configured SparkLightGBMRegressor instance.
        """
        n_estimators = self.get_param("n_estimators")
        max_depth = self.get_param("max_depth")
        learning_rate = self.get_param("learning_rate")
        num_leaves = self.get_param("num_leaves")
        min_data_in_leaf = self.get_param("min_data_in_leaf")
        feature_fraction = self.get_param("feature_fraction")
        bagging_fraction = self.get_param("bagging_fraction")
        bagging_freq = self.get_param("bagging_freq")
        lambda_l1 = self.get_param("lambda_l1")
        lambda_l2 = self.get_param("lambda_l2")
        seed = self.get_param("seed")

        estimator = SparkLightGBMRegressor(
            numIterations=n_estimators,
            maxDepth=max_depth,
            learningRate=learning_rate,
            numLeaves=num_leaves,
            minDataInLeaf=min_data_in_leaf,
            featureFraction=feature_fraction,
            baggingFraction=bagging_fraction,
            baggingFreq=bagging_freq,
            lambdaL1=lambda_l1,
            lambdaL2=lambda_l2,
        )

        if seed is not None:
            estimator.setSeed(seed)

        return estimator

    def _fit_spark_model(
        self,
        df: DataFrame,
        label_col: str,
        feature_cols: List[str],
    ) -> Any:
        """Fit the LightGBM model.

        Override base class method to handle LightGBM's API.

        Args:
            df: PySpark DataFrame with training data.
            label_col: Name of the label column.
            feature_cols: List of feature column names.

        Returns:
            Fitted LightGBM model.
        """
        # Assemble features
        df_with_features = self._assemble_features(df, feature_cols)

        # Get parameters
        n_estimators = self.get_param("n_estimators")
        max_depth = self.get_param("max_depth")
        learning_rate = self.get_param("learning_rate")
        num_leaves = self.get_param("num_leaves")
        min_data_in_leaf = self.get_param("min_data_in_leaf")
        feature_fraction = self.get_param("feature_fraction")
        bagging_fraction = self.get_param("bagging_fraction")
        bagging_freq = self.get_param("bagging_freq")
        lambda_l1 = self.get_param("lambda_l1")
        lambda_l2 = self.get_param("lambda_l2")
        seed = self.get_param("seed")

        # Create estimator with all params including column names
        estimator = SparkLightGBMRegressor(
            numIterations=n_estimators,
            maxDepth=max_depth,
            learningRate=learning_rate,
            numLeaves=num_leaves,
            minDataInLeaf=min_data_in_leaf,
            featureFraction=feature_fraction,
            baggingFraction=bagging_fraction,
            baggingFreq=bagging_freq,
            lambdaL1=lambda_l1,
            lambdaL2=lambda_l2,
            featuresCol=self.FEATURES_COL,
            labelCol=label_col,
            predictionCol=self.PREDICTION_COL,
        )

        if seed is not None:
            estimator.setSeed(seed)

        # Store feature columns for prediction
        self._feature_cols = feature_cols
        self._label_col = label_col

        # Fit the model
        self._spark_model = estimator.fit(df_with_features)

        return self._spark_model

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

    def _create_spark_estimator(self) -> Any:
        """Create the underlying SparkLightGBMClassifier.

        Returns:
            Configured SparkLightGBMClassifier instance.
        """
        n_estimators = self.get_param("n_estimators")
        max_depth = self.get_param("max_depth")
        learning_rate = self.get_param("learning_rate")
        num_leaves = self.get_param("num_leaves")
        min_data_in_leaf = self.get_param("min_data_in_leaf")
        feature_fraction = self.get_param("feature_fraction")
        bagging_fraction = self.get_param("bagging_fraction")
        bagging_freq = self.get_param("bagging_freq")
        lambda_l1 = self.get_param("lambda_l1")
        lambda_l2 = self.get_param("lambda_l2")
        seed = self.get_param("seed")

        estimator = SparkLightGBMClassifier(
            numIterations=n_estimators,
            maxDepth=max_depth,
            learningRate=learning_rate,
            numLeaves=num_leaves,
            minDataInLeaf=min_data_in_leaf,
            featureFraction=feature_fraction,
            baggingFraction=bagging_fraction,
            baggingFreq=bagging_freq,
            lambdaL1=lambda_l1,
            lambdaL2=lambda_l2,
        )

        if seed is not None:
            estimator.setSeed(seed)

        return estimator

    def _fit_spark_model(
        self,
        df: DataFrame,
        label_col: str,
        feature_cols: List[str],
    ) -> Any:
        """Fit the LightGBM classifier.

        Override base class method to handle LightGBM's API.

        Args:
            df: PySpark DataFrame with training data.
            label_col: Name of the label column.
            feature_cols: List of feature column names.

        Returns:
            Fitted LightGBM model.
        """
        # Assemble features
        df_with_features = self._assemble_features(df, feature_cols)

        # Get parameters
        n_estimators = self.get_param("n_estimators")
        max_depth = self.get_param("max_depth")
        learning_rate = self.get_param("learning_rate")
        num_leaves = self.get_param("num_leaves")
        min_data_in_leaf = self.get_param("min_data_in_leaf")
        feature_fraction = self.get_param("feature_fraction")
        bagging_fraction = self.get_param("bagging_fraction")
        bagging_freq = self.get_param("bagging_freq")
        lambda_l1 = self.get_param("lambda_l1")
        lambda_l2 = self.get_param("lambda_l2")
        seed = self.get_param("seed")

        # Create estimator with all params including column names
        estimator = SparkLightGBMClassifier(
            numIterations=n_estimators,
            maxDepth=max_depth,
            learningRate=learning_rate,
            numLeaves=num_leaves,
            minDataInLeaf=min_data_in_leaf,
            featureFraction=feature_fraction,
            baggingFraction=bagging_fraction,
            baggingFreq=bagging_freq,
            lambdaL1=lambda_l1,
            lambdaL2=lambda_l2,
            featuresCol=self.FEATURES_COL,
            labelCol=label_col,
            predictionCol=self.PREDICTION_COL,
            probabilityCol=self.PROBABILITY_COL,
            rawPredictionCol=self.RAW_PREDICTION_COL,
        )

        if seed is not None:
            estimator.setSeed(seed)

        # Store feature columns for prediction
        self._feature_cols = feature_cols
        self._label_col = label_col

        # Fit the model
        self._spark_model = estimator.fit(df_with_features)

        return self._spark_model

    def _load_artifacts(self, path: str) -> None:
        """Load the Spark model from disk.

        Args:
            path: Directory path where the model is saved.
        """
        self._load_spark_model(path, SparkLightGBMClassifierModel)
