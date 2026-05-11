"""Regressors factory for creating regression models."""

import json
import os
from typing import Any

from smallaxe.exceptions import DependencyError, ValidationError
from smallaxe.training.catboost import (
    CatBoostRegressor,
    catboost_install_hint,
    is_catboost_available,
)
from smallaxe.training.lightgbm import (
    LIGHTGBM_AVAILABLE,
    LightGBMRegressor,
)
from smallaxe.training.random_forest import RandomForestRegressor
from smallaxe.training.xgboost import XGBOOST_AVAILABLE, XGBoostRegressor

XGBOOST_INSTALL_HINT = "pip install smallaxe[xgboost]"
LIGHTGBM_INSTALL_HINT = (
    "pip install smallaxe[lightgbm] and configure Spark with the SynapseML package"
)
CATBOOST_INSTALL_HINT = catboost_install_hint()


def _dependency_error(model_name: str) -> DependencyError:
    """Build an actionable dependency error for an optional regressor."""
    if model_name == "xgboost":
        return DependencyError(package="xgboost", install_command=XGBOOST_INSTALL_HINT)
    if model_name == "lightgbm":
        return DependencyError(package="synapseml", install_command=LIGHTGBM_INSTALL_HINT)
    if model_name == "catboost":
        return DependencyError(package="catboost_spark", install_command=CATBOOST_INSTALL_HINT)
    return DependencyError()


class Regressors:
    """Factory class for creating and loading regression models.

    This class provides a convenient interface for creating regression models
    without needing to import specific model classes directly.

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

    # Registry of supported regressor types and their classes
    _REGISTRY = {
        "RandomForestRegressor": RandomForestRegressor,
    }

    # Add optional regressors to registry only when their dependencies are available.
    if XGBOOST_AVAILABLE:
        _REGISTRY["XGBoostRegressor"] = XGBoostRegressor
    if LIGHTGBM_AVAILABLE:
        _REGISTRY["LightGBMRegressor"] = LightGBMRegressor
    if is_catboost_available():
        _REGISTRY["CatBoostRegressor"] = CatBoostRegressor

    @staticmethod
    def _refresh_optional_registry() -> None:
        """Add optional model classes that became available after import time."""
        if is_catboost_available():
            Regressors._REGISTRY["CatBoostRegressor"] = CatBoostRegressor

    @staticmethod
    def random_forest(**kwargs: Any) -> RandomForestRegressor:
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
            RandomForestRegressor: A configured Random Forest regressor instance.

        Example:
            >>> model = Regressors.random_forest(n_estimators=100, max_depth=10)
            >>> model.fit(df, label_col='target', feature_cols=['f1', 'f2'])
        """
        model = RandomForestRegressor()
        if kwargs:
            model.set_param(kwargs)
        return model

    @staticmethod
    def xgboost(**kwargs: Any) -> "XGBoostRegressor":
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
            XGBoostRegressor: A configured XGBoost regressor instance.

        Raises:
            DependencyError: If xgboost is not installed.

        Example:
            >>> model = Regressors.xgboost(n_estimators=100, max_depth=6)
            >>> model.fit(df, label_col='target', feature_cols=['f1', 'f2'])
        """
        if not XGBOOST_AVAILABLE:
            raise _dependency_error("xgboost")

        model = XGBoostRegressor()
        if kwargs:
            model.set_param(kwargs)
        return model

    @staticmethod
    def lightgbm(**kwargs: Any) -> "LightGBMRegressor":
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
            LightGBMRegressor: A configured LightGBM regressor instance.

        Raises:
            DependencyError: If SynapseML LightGBM support is not installed.
        """
        if not LIGHTGBM_AVAILABLE:
            raise _dependency_error("lightgbm")

        model = LightGBMRegressor()
        if kwargs:
            model.set_param(kwargs)
        return model

    @staticmethod
    def catboost(**kwargs: Any) -> "CatBoostRegressor":
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
            CatBoostRegressor: A configured CatBoost regressor instance.

        Raises:
            DependencyError: If CatBoost Spark support is not installed.
        """
        if not is_catboost_available():
            raise _dependency_error("catboost")
        Regressors._refresh_optional_registry()

        model = CatBoostRegressor()
        if kwargs:
            model.set_param(kwargs)
        return model

    @staticmethod
    def load(path: str) -> Any:
        """Load a regressor from disk.

        This method automatically detects the model type from the saved metadata
        and loads the appropriate model class.

        Args:
            path: Directory path where the model was saved.

        Returns:
            The loaded regressor instance.

        Raises:
            ValidationError: If the saved model is not a supported regressor type.
            FileNotFoundError: If the model directory or metadata file doesn't exist.

        Example:
            >>> model = Regressors.random_forest(n_estimators=100)
            >>> model.fit(df, label_col='target', feature_cols=['f1', 'f2'])
            >>> model.save('/path/to/model')
            >>>
            >>> loaded_model = Regressors.load('/path/to/model')
            >>> predictions = loaded_model.predict(df)
        """
        # Read metadata to determine model type
        metadata_path = os.path.join(path, "metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"Model metadata not found at {metadata_path}. "
                "Ensure the path points to a valid model directory."
            )

        with open(metadata_path) as f:
            metadata = json.load(f)

        model_class_name = metadata.get("__class__")
        if model_class_name is None:
            raise ValidationError(
                "Model metadata does not contain '__class__'. "
                "This may be an older model format or corrupted metadata."
            )

        # Check if it's a regressor
        if model_class_name == "XGBoostRegressor" and not XGBOOST_AVAILABLE:
            raise _dependency_error("xgboost")
        if model_class_name == "LightGBMRegressor" and not LIGHTGBM_AVAILABLE:
            raise _dependency_error("lightgbm")
        if model_class_name == "CatBoostRegressor":
            if not is_catboost_available():
                raise _dependency_error("catboost")
            Regressors._refresh_optional_registry()

        if model_class_name not in Regressors._REGISTRY:
            raise ValidationError(
                f"Model type '{model_class_name}' is not a supported regressor. "
                f"Supported types are: {list(Regressors._REGISTRY.keys())}"
            )

        model_class = Regressors._REGISTRY[model_class_name]
        return model_class.load(path)

    @staticmethod
    def list_models() -> list:
        """List all available regressor model types.

        Returns:
            List of supported regressor model type names.

        Example:
            >>> Regressors.list_models()
            ['RandomForestRegressor']
        """
        return list(Regressors._REGISTRY.keys())

    @staticmethod
    def available_models() -> dict:
        """Report installed and unavailable regressor models with install hints.

        Returns:
            Dictionary keyed by factory method name. Each entry includes the
            implementation class name, availability status, optional dependency,
            and install hint when applicable.
        """
        return {
            "random_forest": {
                "class_name": "RandomForestRegressor",
                "available": True,
                "dependency": None,
                "install_hint": None,
            },
            "xgboost": {
                "class_name": "XGBoostRegressor",
                "available": XGBOOST_AVAILABLE,
                "dependency": "xgboost",
                "install_hint": None if XGBOOST_AVAILABLE else XGBOOST_INSTALL_HINT,
            },
            "lightgbm": {
                "class_name": "LightGBMRegressor",
                "available": LIGHTGBM_AVAILABLE,
                "dependency": "synapseml",
                "install_hint": None if LIGHTGBM_AVAILABLE else LIGHTGBM_INSTALL_HINT,
            },
            "catboost": {
                "class_name": "CatBoostRegressor",
                "available": is_catboost_available(),
                "dependency": "catboost_spark",
                "install_hint": None if is_catboost_available() else CATBOOST_INSTALL_HINT,
            },
        }
