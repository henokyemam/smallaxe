"""Random Forest models for regression and classification."""

from typing import Any, Dict

from pyspark.ml.regression import RandomForestRegressionModel
from pyspark.ml.regression import RandomForestRegressor as SparkRFRegressor

from smallaxe.training.base import BaseRegressor


class RandomForestRegressor(BaseRegressor):
    """Random Forest Regressor for regression tasks.

    This class wraps PySpark MLlib's RandomForestRegressor to provide
    a scikit-learn-like interface with support for train/test and k-fold
    cross-validation.

    Args:
        task: The regression task type. Default is 'simple_regression'.

    Example:
        >>> from smallaxe.training import RandomForestRegressor
        >>> model = RandomForestRegressor()
        >>> model.set_param({"n_estimators": 100, "max_depth": 10})
        >>> model.fit(df, label_col='target', feature_cols=['f1', 'f2'])
        >>> predictions = model.predict(df)
    """

    @property
    def params(self) -> Dict[str, str]:
        """Get parameter descriptions.

        Returns:
            Dictionary mapping parameter names to their descriptions.
        """
        return {
            "n_estimators": "Number of trees in the forest",
            "max_depth": "Maximum depth of each tree (0 = unlimited)",
            "max_bins": "Maximum number of bins for discretizing continuous features",
            "min_instances_per_node": "Minimum number of instances per node",
            "min_info_gain": "Minimum information gain for a split",
            "subsampling_rate": "Fraction of data used for training each tree",
            "feature_subset_strategy": "Strategy for selecting features: 'auto', 'all', 'sqrt', 'log2', 'onethird'",
            "seed": "Random seed for reproducibility",
        }

    @property
    def default_params(self) -> Dict[str, Any]:
        """Get default parameter values.

        Returns:
            Dictionary mapping parameter names to their default values.
        """
        return {
            "n_estimators": 20,
            "max_depth": 5,
            "max_bins": 32,
            "min_instances_per_node": 1,
            "min_info_gain": 0.0,
            "subsampling_rate": 1.0,
            "feature_subset_strategy": "auto",
            "seed": None,
        }

    def _create_spark_estimator(self) -> Any:
        """Create the underlying Spark MLlib RandomForestRegressor.

        Returns:
            Configured Spark MLlib RandomForestRegressor instance.
        """
        n_estimators = self.get_param("n_estimators")
        max_depth = self.get_param("max_depth")
        max_bins = self.get_param("max_bins")
        min_instances_per_node = self.get_param("min_instances_per_node")
        min_info_gain = self.get_param("min_info_gain")
        subsampling_rate = self.get_param("subsampling_rate")
        feature_subset_strategy = self.get_param("feature_subset_strategy")
        seed = self.get_param("seed")

        estimator = SparkRFRegressor(
            numTrees=n_estimators,
            maxDepth=max_depth,
            maxBins=max_bins,
            minInstancesPerNode=min_instances_per_node,
            minInfoGain=min_info_gain,
            subsamplingRate=subsampling_rate,
            featureSubsetStrategy=feature_subset_strategy,
        )

        if seed is not None:
            estimator.setSeed(seed)

        return estimator

    def _load_artifacts(self, path: str) -> None:
        """Load the Spark model from disk.

        Args:
            path: Directory path where the model is saved.
        """
        self._load_spark_model(path, RandomForestRegressionModel)
