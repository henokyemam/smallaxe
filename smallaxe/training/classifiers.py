"""Classifiers factory for creating classification models."""

import json
import os
from typing import Any

from smallaxe.exceptions import ValidationError
from smallaxe.training.random_forest import RandomForestClassifier


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
