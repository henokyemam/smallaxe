"""Classifiers factory for creating classification models."""

import json
import os
from typing import Any

from smallaxe.exceptions import DependencyError, ValidationError
from smallaxe.training.catboost import (
    CatBoostClassifier,
    catboost_install_hint,
    is_catboost_available,
)
from smallaxe.training.lightgbm import (
    LIGHTGBM_AVAILABLE,
    LightGBMClassifier,
)
from smallaxe.training.random_forest import RandomForestClassifier
from smallaxe.training.xgboost import XGBOOST_AVAILABLE, XGBoostClassifier

XGBOOST_INSTALL_HINT = "pip install smallaxe[xgboost]"
LIGHTGBM_INSTALL_HINT = (
    "pip install smallaxe[lightgbm] and configure Spark with the SynapseML package"
)
CATBOOST_INSTALL_HINT = catboost_install_hint()


def _dependency_error(model_name: str) -> DependencyError:
    """Build an actionable dependency error for an optional classifier."""
    if model_name == "xgboost":
        return DependencyError(package="xgboost", install_command=XGBOOST_INSTALL_HINT)
    if model_name == "lightgbm":
        return DependencyError(package="synapseml", install_command=LIGHTGBM_INSTALL_HINT)
    if model_name == "catboost":
        return DependencyError(package="catboost_spark", install_command=CATBOOST_INSTALL_HINT)
    return DependencyError()


class Classifiers:
    """Factory class for creating and loading classification models.

    This class provides a convenient interface for creating classification models
    without needing to import specific model classes directly.

    Example:
        >>> from smallaxe.training import Classifiers
        >>>
        >>> # Create a Random Forest classifier
        >>> model = Classifiers.random_forest(n_estimators=100, max_depth=10)
        >>> model.fit(df, label_col='label', feature_cols=['f1', 'f2'])
        >>>
        >>> # Save and load the model
        >>> model.save('/path/to/model')
        >>> loaded_model = Classifiers.load('/path/to/model')
    """

    # Registry of supported classifier types and their classes
    _REGISTRY = {
        "RandomForestClassifier": RandomForestClassifier,
    }

    # Add optional classifiers to registry only when their dependencies are available.
    if XGBOOST_AVAILABLE:
        _REGISTRY["XGBoostClassifier"] = XGBoostClassifier
    if LIGHTGBM_AVAILABLE:
        _REGISTRY["LightGBMClassifier"] = LightGBMClassifier
    if is_catboost_available():
        _REGISTRY["CatBoostClassifier"] = CatBoostClassifier

    @staticmethod
    def _refresh_optional_registry() -> None:
        """Add optional model classes that became available after import time."""
        if is_catboost_available():
            Classifiers._REGISTRY["CatBoostClassifier"] = CatBoostClassifier

    @staticmethod
    def random_forest(task: str = "binary", **kwargs: Any) -> RandomForestClassifier:
        """Create a Random Forest classifier.

        Args:
            task: The classification task type. Options are 'binary' or 'multiclass'.
                Default is 'binary'.
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
            RandomForestClassifier: A configured Random Forest classifier instance.

        Example:
            >>> model = Classifiers.random_forest(task='binary', n_estimators=100)
            >>> model.fit(df, label_col='label', feature_cols=['f1', 'f2'])
        """
        model = RandomForestClassifier(task=task)
        if kwargs:
            model.set_param(kwargs)
        return model

    @staticmethod
    def xgboost(task: str = "binary", **kwargs: Any) -> "XGBoostClassifier":
        """Create an XGBoost classifier.

        Note:
            This requires the xgboost package to be installed.
            Install with: pip install smallaxe[xgboost]

        Args:
            task: The classification task type. Options are 'binary' or 'multiclass'.
                Default is 'binary'.
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
            XGBoostClassifier: A configured XGBoost classifier instance.

        Raises:
            DependencyError: If xgboost is not installed.

        Example:
            >>> model = Classifiers.xgboost(task='binary', n_estimators=100)
            >>> model.fit(df, label_col='label', feature_cols=['f1', 'f2'])
        """
        if not XGBOOST_AVAILABLE:
            raise _dependency_error("xgboost")

        model = XGBoostClassifier(task=task)
        if kwargs:
            model.set_param(kwargs)
        return model

    @staticmethod
    def lightgbm(task: str = "binary", **kwargs: Any) -> "LightGBMClassifier":
        """Create a LightGBM classifier.

        Note:
            This requires SynapseML LightGBM support to be installed and
            configured for the active Spark session.
            Install with: pip install smallaxe[lightgbm]

        Args:
            task: The classification task type. Options are 'binary' or 'multiclass'.
                Default is 'binary'.
            **kwargs: Parameters to pass to the model. Common parameters include:
                - n_estimators: Number of boosting iterations (default: 100)
                - max_depth: Maximum depth of each tree (default: -1)
                - learning_rate: Boosting learning rate (default: 0.1)
                - num_leaves: Maximum number of leaves in one tree (default: 31)
                - seed: Random seed for reproducibility (default: None)

        Returns:
            LightGBMClassifier: A configured LightGBM classifier instance.

        Raises:
            DependencyError: If SynapseML LightGBM support is not installed.
        """
        if not LIGHTGBM_AVAILABLE:
            raise _dependency_error("lightgbm")

        model = LightGBMClassifier(task=task)
        if kwargs:
            model.set_param(kwargs)
        return model

    @staticmethod
    def catboost(task: str = "binary", **kwargs: Any) -> "CatBoostClassifier":
        """Create a CatBoost classifier.

        Note:
            This requires CatBoost Spark support to be installed and configured
            for the active Spark session.
            Install with: pip install smallaxe[catboost]

        Args:
            task: The classification task type. Options are 'binary' or 'multiclass'.
                Default is 'binary'.
            **kwargs: Parameters to pass to the model. Common parameters include:
                - n_estimators: Number of boosting iterations (default: 100)
                - max_depth: Maximum tree depth (default: 6)
                - learning_rate: Boosting learning rate (default: 0.03)
                - seed: Random seed for reproducibility (default: None)

        Returns:
            CatBoostClassifier: A configured CatBoost classifier instance.

        Raises:
            DependencyError: If CatBoost Spark support is not installed.
        """
        if not is_catboost_available():
            raise _dependency_error("catboost")
        Classifiers._refresh_optional_registry()

        model = CatBoostClassifier(task=task)
        if kwargs:
            model.set_param(kwargs)
        return model

    @staticmethod
    def load(path: str) -> Any:
        """Load a classifier from disk.

        This method automatically detects the model type from the saved metadata
        and loads the appropriate model class.

        Args:
            path: Directory path where the model was saved.

        Returns:
            The loaded classifier instance.

        Raises:
            ValidationError: If the saved model is not a supported classifier type.
            FileNotFoundError: If the model directory or metadata file doesn't exist.

        Example:
            >>> model = Classifiers.random_forest(n_estimators=100)
            >>> model.fit(df, label_col='label', feature_cols=['f1', 'f2'])
            >>> model.save('/path/to/model')
            >>>
            >>> loaded_model = Classifiers.load('/path/to/model')
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

        # Check if it's a classifier
        if model_class_name == "XGBoostClassifier" and not XGBOOST_AVAILABLE:
            raise _dependency_error("xgboost")
        if model_class_name == "LightGBMClassifier" and not LIGHTGBM_AVAILABLE:
            raise _dependency_error("lightgbm")
        if model_class_name == "CatBoostClassifier":
            if not is_catboost_available():
                raise _dependency_error("catboost")
            Classifiers._refresh_optional_registry()

        if model_class_name not in Classifiers._REGISTRY:
            raise ValidationError(
                f"Model type '{model_class_name}' is not a supported classifier. "
                f"Supported types are: {list(Classifiers._REGISTRY.keys())}"
            )

        model_class = Classifiers._REGISTRY[model_class_name]
        return model_class.load(path)

    @staticmethod
    def list_models() -> list:
        """List all available classifier model types.

        Returns:
            List of supported classifier model type names.

        Example:
            >>> Classifiers.list_models()
            ['RandomForestClassifier']
        """
        return list(Classifiers._REGISTRY.keys())

    @staticmethod
    def available_models() -> dict:
        """Report installed and unavailable classifier models with install hints.

        Returns:
            Dictionary keyed by factory method name. Each entry includes the
            implementation class name, availability status, optional dependency,
            and install hint when applicable.
        """
        return {
            "random_forest": {
                "class_name": "RandomForestClassifier",
                "available": True,
                "dependency": None,
                "install_hint": None,
            },
            "xgboost": {
                "class_name": "XGBoostClassifier",
                "available": XGBOOST_AVAILABLE,
                "dependency": "xgboost",
                "install_hint": None if XGBOOST_AVAILABLE else XGBOOST_INSTALL_HINT,
            },
            "lightgbm": {
                "class_name": "LightGBMClassifier",
                "available": LIGHTGBM_AVAILABLE,
                "dependency": "synapseml",
                "install_hint": None if LIGHTGBM_AVAILABLE else LIGHTGBM_INSTALL_HINT,
            },
            "catboost": {
                "class_name": "CatBoostClassifier",
                "available": is_catboost_available(),
                "dependency": "catboost_spark",
                "install_hint": None if is_catboost_available() else CATBOOST_INSTALL_HINT,
            },
        }
