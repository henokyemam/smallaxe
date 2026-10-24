"""Classifiers factory for creating classification models."""

import os
from typing import Any, Dict, List

from smallaxe import _fs
from smallaxe.exceptions import ValidationError
from smallaxe.training import algorithm
from smallaxe.training.base import Classifier, Model


class Classifiers:
    """Factory class for creating and loading classification models.

    Each factory method returns a :class:`~smallaxe.training.base.Classifier` bound
    to one algorithm, so no algorithm-specific class needs to be imported.

    Example:
        >>> from smallaxe.training import Classifiers
        >>>
        >>> # Create a Random Forest classifier
        >>> model = Classifiers.random_forest(task='binary', n_estimators=100)
        >>> model.fit(df, label_col='label', feature_cols=['f1', 'f2'])
        >>>
        >>> # Save and load the model
        >>> model.save('/path/to/model')
        >>> loaded_model = Classifiers.load('/path/to/model')
    """

    @staticmethod
    def _create(name: str, task: str, params: Dict[str, Any]) -> Classifier:
        model = Classifier(name, task=task)
        if params:
            model.set_param(params)
        return model

    @staticmethod
    def random_forest(task: str = "binary", **kwargs: Any) -> Classifier:
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
            A configured Random Forest classifier.

        Example:
            >>> model = Classifiers.random_forest(task='binary', n_estimators=100)
            >>> model.fit(df, label_col='label', feature_cols=['f1', 'f2'])
        """
        return Classifiers._create("random_forest", task, kwargs)

    @staticmethod
    def xgboost(task: str = "binary", **kwargs: Any) -> Classifier:
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
            A configured XGBoost classifier.

        Raises:
            DependencyError: If xgboost is not installed.

        Example:
            >>> model = Classifiers.xgboost(task='binary', n_estimators=100)
            >>> model.fit(df, label_col='label', feature_cols=['f1', 'f2'])
        """
        return Classifiers._create("xgboost", task, kwargs)

    @staticmethod
    def lightgbm(task: str = "binary", **kwargs: Any) -> Classifier:
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
            A configured LightGBM classifier.

        Raises:
            DependencyError: If SynapseML LightGBM support is not installed.
        """
        return Classifiers._create("lightgbm", task, kwargs)

    @staticmethod
    def catboost(task: str = "binary", **kwargs: Any) -> Classifier:
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
            A configured CatBoost classifier.

        Raises:
            DependencyError: If CatBoost Spark support is not installed.
        """
        return Classifiers._create("catboost", task, kwargs)

    @staticmethod
    def load(path: str) -> Classifier:
        """Load a classifier from disk.

        The algorithm is read from the saved metadata, so any classifier saved by
        smallaxe (including 0.8.x artifacts) loads through this one method.

        Args:
            path: Directory path where the model was saved.

        Returns:
            The loaded classifier.

        Raises:
            FileNotFoundError: If the model directory or metadata file doesn't exist.
            ValidationError: If the saved model is not a classifier.
            DependencyError: If the saved model's algorithm is not installed.

        Example:
            >>> model = Classifiers.random_forest(task='binary', n_estimators=100)
            >>> model.fit(df, label_col='label', feature_cols=['f1', 'f2'])
            >>> model.save('/path/to/model')
            >>>
            >>> loaded_model = Classifiers.load('/path/to/model')
            >>> predictions = loaded_model.predict(df)
        """
        metadata_path = os.path.join(path, "metadata.json")
        if not _fs.exists(metadata_path):
            raise FileNotFoundError(
                f"Model metadata not found at {metadata_path}. "
                "Ensure the path points to a valid model directory."
            )
        model = Model.load(path)
        if not isinstance(model, Classifier):
            saved = algorithm.legacy_class_name(model.algorithm.name, model.task_type)
            raise ValidationError(
                f"Model type '{saved}' is not a supported classifier. "
                f"Supported types are: {Classifiers.list_models()}"
            )
        return model

    @staticmethod
    def list_models() -> List[str]:
        """List the classifier model types available in this environment.

        Returns:
            Class names of the classifiers whose dependencies are installed.

        Example:
            >>> Classifiers.list_models()
            ['RandomForestClassifier']
        """
        return [
            algorithm.legacy_class_name(name, "classification")
            for name in algorithm.names()
            if algorithm.get(name).available()
        ]

    @staticmethod
    def available_models() -> Dict[str, Dict[str, Any]]:
        """Report installed and unavailable classifier models with install hints.

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
                "class_name": algorithm.legacy_class_name(name, "classification"),
                "available": available,
                "dependency": dependency.package if dependency else None,
                "install_hint": (
                    None if available or dependency is None else dependency.install_hint
                ),
            }
        return report
