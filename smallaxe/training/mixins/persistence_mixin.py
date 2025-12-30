"""PersistenceMixin for saving and loading models."""

import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type, TypeVar

from smallaxe.exceptions import ModelNotFittedError, ValidationError

T = TypeVar("T", bound="PersistenceMixin")


class PersistenceMixin(ABC):
    """Mixin providing model persistence functionality.

    This mixin provides a consistent interface for saving and loading
    trained models to/from disk.
    """

    @abstractmethod
    def _get_persistence_state(self) -> Dict[str, Any]:
        """Get the state dict to persist.

        Subclasses must implement this to return all state needed
        to reconstruct the model.

        Returns:
            Dictionary containing all model state.
        """
        pass

    @abstractmethod
    def _set_persistence_state(self, state: Dict[str, Any]) -> None:
        """Restore model state from a dict.

        Subclasses must implement this to restore the model from
        saved state.

        Args:
            state: Dictionary containing saved model state.
        """
        pass

    @property
    def _is_fitted(self) -> bool:
        """Check if the model has been fitted.

        Override in subclasses to provide custom logic.
        """
        return getattr(self, "_fitted", False)

    @_is_fitted.setter
    def _is_fitted(self, value: bool) -> None:
        """Set the fitted state."""
        self._fitted = value

    def save(self, path: str) -> None:
        """Save the model to disk.

        Creates a directory at the specified path containing:
        - metadata.json: Model parameters and configuration
        - model artifacts (model-specific files)

        Args:
            path: Directory path to save the model to.

        Raises:
            ModelNotFittedError: If the model has not been fitted.
            ValidationError: If the path is invalid.

        Example:
            >>> model.fit(df, label_col='target', feature_cols=['f1', 'f2'])
            >>> model.save('/path/to/model')
        """
        if not self._is_fitted:
            raise ModelNotFittedError(
                "Model has not been fitted. Call fit() before save()."
            )

        if not path:
            raise ValidationError("Path cannot be empty.")

        os.makedirs(path, exist_ok=True)

        # Get model state
        state = self._get_persistence_state()

        # Add class information for loading
        state["__class__"] = self.__class__.__name__
        state["__module__"] = self.__class__.__module__

        # Save metadata
        metadata_path = os.path.join(path, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(state, f, indent=2, default=str)

        # Allow subclasses to save additional artifacts
        self._save_artifacts(path)

    def _save_artifacts(self, path: str) -> None:
        """Save additional model artifacts.

        Override in subclasses to save model-specific files
        (e.g., Spark ML model files).

        Args:
            path: Directory path where artifacts should be saved.
        """
        pass

    @classmethod
    def load(cls: Type[T], path: str) -> T:
        """Load a model from disk.

        Args:
            path: Directory path to load the model from.

        Returns:
            The loaded model instance.

        Raises:
            ValidationError: If the path doesn't exist or is invalid.
            ValidationError: If the saved model type doesn't match.

        Example:
            >>> model = RandomForestRegressor.load('/path/to/model')
            >>> predictions = model.predict(df)
        """
        if not path or not os.path.exists(path):
            raise ValidationError(f"Model path does not exist: {path}")

        metadata_path = os.path.join(path, "metadata.json")
        if not os.path.exists(metadata_path):
            raise ValidationError(
                f"Invalid model directory: metadata.json not found at {path}"
            )

        # Load metadata
        with open(metadata_path) as f:
            state = json.load(f)

        # Verify class matches
        saved_class = state.pop("__class__", None)
        saved_module = state.pop("__module__", None)

        if saved_class and saved_class != cls.__name__:
            raise ValidationError(
                f"Model type mismatch: expected {cls.__name__}, "
                f"got {saved_class}."
            )

        # Create new instance and restore state
        instance = cls.__new__(cls)
        instance._set_persistence_state(state)

        # Allow subclasses to load additional artifacts
        instance._load_artifacts(path)

        return instance

    def _load_artifacts(self, path: str) -> None:
        """Load additional model artifacts.

        Override in subclasses to load model-specific files.

        Args:
            path: Directory path where artifacts are stored.
        """
        pass
