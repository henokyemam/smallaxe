"""PersistenceMixin: how a model is saved, and how its saved state is read back.

A model artifact is a directory holding the model's own files (the Spark model, written
by :meth:`PersistenceMixin._save_artifacts`) and a versioned ``metadata.json``
(:mod:`smallaxe._artifact`, kind ``"model"``) written last, so a directory with a
``metadata.json`` is a complete artifact.

Turning saved state back into a model is :meth:`smallaxe.training.base.Model.load`, the
one load rule for concrete models; this mixin only reads and checks the metadata
(:meth:`PersistenceMixin._read_saved_state`).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

from smallaxe import _artifact, _fs
from smallaxe.exceptions import ModelNotFittedError, ValidationError

ARTIFACT_KIND = "model"


class PersistenceMixin(ABC):
    """Mixin providing the save rule for models and reading their saved state."""

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
        """Save the model as a directory at ``path``.

        The directory holds the model's files (``spark_model/``) and ``metadata.json``
        with ``format_version``, ``kind`` (``"model"``), ``smallaxe_version`` and the
        model's state. Saving over an existing artifact first removes its
        ``metadata.json``, then writes the model files, then the new ``metadata.json``:
        if saving fails part way, the directory is not mistaken for the old artifact.
        All IO goes through the Hadoop FileSystem layer, so ``dbfs:/``, ``s3://`` and
        ``hdfs://`` paths work.

        Args:
            path: Directory path to save the model to.

        Raises:
            ModelNotFittedError: If the model has not been fitted.
            ValidationError: If the path is empty.

        Example:
            >>> model.fit(dataframe, label_col='target', feature_cols=['f1', 'f2'])
            >>> model.save('/path/to/model')
        """
        if not self._is_fitted:
            raise ModelNotFittedError("Model has not been fitted. Call fit() before save().")

        if not path:
            raise ValidationError("Path cannot be empty.")

        state = self._get_persistence_state()
        state["__class__"] = self.__class__.__name__
        state["__module__"] = self.__class__.__module__

        _fs.delete(_fs.join(path, _artifact.METADATA_FILE))
        self._save_artifacts(path)
        _artifact.write_metadata(path, ARTIFACT_KIND, state)

    def _save_artifacts(self, path: str) -> None:  # noqa: B027
        """Save additional model artifacts.

        Override in subclasses to save model-specific files
        (e.g., Spark ML model files).

        Args:
            path: Directory path where artifacts should be saved.
        """
        pass

    @classmethod
    def _read_saved_state(cls, path: str) -> Dict[str, Any]:
        """Read and check the ``metadata.json`` of the model artifact at ``path``.

        Artifacts written by smallaxe 0.10 and earlier have no header and are returned
        as they are; the caller validates their keys.

        Args:
            path: The artifact directory.

        Returns:
            The saved state, without the header keys.

        Raises:
            ValidationError: If the path or its ``metadata.json`` does not exist, the
                file cannot be read, it holds another kind of artifact, or it was
                written by a newer smallaxe.
        """
        metadata = _artifact.read_metadata(path, ARTIFACT_KIND, allow_unversioned=True)
        return {key: value for key, value in metadata.items() if key not in _artifact.RESERVED_KEYS}
