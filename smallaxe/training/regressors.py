"""Regressors factory for creating regression models."""

import os
from typing import Any, Dict, List

from smallaxe import _fs
from smallaxe.exceptions import ValidationError
from smallaxe.training import algorithm
from smallaxe.training.base import Model, Regressor


class Regressors:
    """Factory class for creating and loading regression models.

    Each factory method returns a :class:`~smallaxe.training.base.Regressor` bound
    to one algorithm, so no algorithm-specific class needs to be imported.

    Example:
        >>> from smallaxe.training import Regressors
        >>>
        >>> # Create a Random Forest regressor
        >>> model = Regressors.random_forest(n_estimators=100, max_depth=10)
        >>> model.fit(df, label_col='target', feature_cols=['f1', 'f2'])
        >>>
        >>> # Save and load the model
        >>> model.save('/path/to/model')
        >>> loaded_model = Regressors.load('/path/to/model')
    """

    @staticmethod
    def _create(name: str, params: Dict[str, Any]) -> Regressor:
        model = Regressor(name)
        if params:
            model.set_param(params)
        return model

    @staticmethod
    def random_forest(**kwargs: Any) -> Regressor:
        """Create a Random Forest regressor.

        Args:
            **kwargs: Parameters to pass to the model. Common parameters include:
                - n_estimators: Number of trees in the forest (default: 20)
                - max_depth: Maximum depth of each tree (default: 5)
                - max_bins: Maximum number of bins for discretizing features (default: 32)
                - min_instances_per_node: Minimum instances per node (default: 1)
                - min_info_gain: Minimum information gain for a split (default: 0.0)
                - subsampling_rate: Fraction of data for training each tree (default: 1.0)
                - feature_subset_strategy: Strategy for selecting features (default: 'auto')
                - seed: Random seed for reproducibility (default: None)

        Returns:
            A configured Random Forest regressor.

        Example:
            >>> model = Regressors.random_forest(n_estimators=100, max_depth=10)
            >>> model.fit(df, label_col='target', feature_cols=['f1', 'f2'])
        """
        return Regressors._create("random_forest", kwargs)

    @staticmethod
    def xgboost(**kwargs: Any) -> Regressor:
        """Create an XGBoost regressor.

        Note:
            This requires the xgboost package to be installed.
            Install with: pip install smallaxe[xgboost]

        Args:
            **kwargs: Parameters to pass to the model. Common parameters include:
                - n_estimators: Number of boosting rounds (default: 100)
                - max_depth: Maximum depth of each tree (default: 6)
                - learning_rate: Step size shrinkage (default: 0.3)
                - subsample: Fraction of samples for training each tree (default: 1.0)
                - colsample_bytree: Fraction of features for training each tree (default: 1.0)
                - min_child_weight: Minimum sum of instance weight in a child (default: 1)
                - reg_alpha: L1 regularization term (default: 0.0)
                - reg_lambda: L2 regularization term (default: 1.0)
                - gamma: Minimum loss reduction for a split (default: 0.0)
                - seed: Random seed for reproducibility (default: None)

        Returns:
            A configured XGBoost regressor.

        Raises:
            DependencyError: If xgboost is not installed.

        Example:
            >>> model = Regressors.xgboost(n_estimators=100, max_depth=6)
            >>> model.fit(df, label_col='target', feature_cols=['f1', 'f2'])
        """
        return Regressors._create("xgboost", kwargs)

    @staticmethod
    def lightgbm(**kwargs: Any) -> Regressor:
        """Create a LightGBM regressor.

        Note:
            This requires SynapseML LightGBM support to be installed and
            configured for the active Spark session.
            Install with: pip install smallaxe[lightgbm]

        Args:
            **kwargs: Parameters to pass to the model. Common parameters include:
                - n_estimators: Number of boosting iterations (default: 100)
                - max_depth: Maximum depth of each tree (default: -1)
                - learning_rate: Boosting learning rate (default: 0.1)
                - num_leaves: Maximum number of leaves in one tree (default: 31)
                - seed: Random seed for reproducibility (default: None)

        Returns:
            A configured LightGBM regressor.

        Raises:
            DependencyError: If SynapseML LightGBM support is not installed.
        """
        return Regressors._create("lightgbm", kwargs)

    @staticmethod
    def catboost(**kwargs: Any) -> Regressor:
        """Create a CatBoost regressor.

        Note:
            This requires CatBoost Spark support to be installed and configured
            for the active Spark session.
            Install with: pip install smallaxe[catboost]

        Args:
            **kwargs: Parameters to pass to the model. Common parameters include:
                - n_estimators: Number of boosting iterations (default: 100)
                - max_depth: Maximum tree depth (default: 6)
                - learning_rate: Boosting learning rate (default: 0.03)
                - seed: Random seed for reproducibility (default: None)

        Returns:
            A configured CatBoost regressor.

        Raises:
            DependencyError: If CatBoost Spark support is not installed.
        """
        return Regressors._create("catboost", kwargs)

    @staticmethod
    def load(path: str) -> Regressor:
        """Load a regressor from disk.

        The algorithm is read from the saved metadata, so any regressor saved by
        smallaxe (including 0.8.x artifacts) loads through this one method.

        Args:
            path: Directory path where the model was saved.

        Returns:
            The loaded regressor.

        Raises:
            FileNotFoundError: If the model directory or metadata file doesn't exist.
            ValidationError: If the saved model is not a regressor.
            DependencyError: If the saved model's algorithm is not installed.

        Example:
            >>> model = Regressors.random_forest(n_estimators=100)
            >>> model.fit(df, label_col='target', feature_cols=['f1', 'f2'])
            >>> model.save('/path/to/model')
            >>>
            >>> loaded_model = Regressors.load('/path/to/model')
            >>> predictions = loaded_model.predict(df)
        """
        metadata_path = os.path.join(path, "metadata.json")
        if not _fs.exists(metadata_path):
            raise FileNotFoundError(
                f"Model metadata not found at {metadata_path}. "
                "Ensure the path points to a valid model directory."
            )
        model = Model.load(path)
        if not isinstance(model, Regressor):
            saved = algorithm.legacy_class_name(model.algorithm.name, model.task_type)
            raise ValidationError(
                f"Model type '{saved}' is not a supported regressor. "
                f"Supported types are: {Regressors.list_models()}"
            )
        return model

    @staticmethod
    def list_models() -> List[str]:
        """List the regressor model types available in this environment.

        Returns:
            Class names of the regressors whose dependencies are installed.

        Example:
            >>> Regressors.list_models()
            ['RandomForestRegressor']
        """
        return [
            algorithm.legacy_class_name(name, "regression")
            for name in algorithm.names()
            if algorithm.get(name).available()
        ]

    @staticmethod
    def available_models() -> Dict[str, Dict[str, Any]]:
        """Report installed and unavailable regressor models with install hints.

        Returns:
            Dictionary keyed by factory method name. Each entry includes the
            implementation class name, availability status, optional dependency,
            and install hint when applicable.
        """
        report = {}
        for name in algorithm.names():
            record = algorithm.get(name)
            available = record.available()
            dependency = record.dependency
            report[name] = {
                "class_name": algorithm.legacy_class_name(name, "regression"),
                "available": available,
                "dependency": dependency.package if dependency else None,
                "install_hint": (
                    None if available or dependency is None else dependency.install_hint
                ),
            }
        return report
