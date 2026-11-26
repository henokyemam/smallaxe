"""Custom exception classes for smallaxe.

Exception Hierarchy:
    SmallaxeError (base)
    ├── ValidationError
    ├── PreprocessingError
    ├── ModelNotFittedError
    ├── ColumnNotFoundError
    ├── DependencyError
    └── ConfigurationError
"""

from typing import Any, List, Optional

__all__ = [
    "SmallaxeError",
    "ValidationError",
    "PreprocessingError",
    "ModelNotFittedError",
    "ColumnNotFoundError",
    "DependencyError",
    "ConfigurationError",
]


class _NoValue:
    """ConfigurationError's "no value given" marker: None is a value a setting can reject."""

    def __repr__(self) -> str:
        return "<no value>"


_NO_VALUE: Any = _NoValue()


class SmallaxeError(Exception):
    """Base exception for all smallaxe errors: ``except SmallaxeError`` catches every one.

    Args:
        message: The error message.

    Attributes:
        message: The error message.
    """

    def __init__(self, message: str = "An error occurred in smallaxe.") -> None:
        self.message = message
        super().__init__(self.message)


class ValidationError(SmallaxeError):
    """Raised when an argument, the data, or a saved artifact is invalid.

    Args:
        message: The error message.
    """

    def __init__(self, message: str = "Invalid input parameters or data.") -> None:
        super().__init__(message)


class PreprocessingError(SmallaxeError):
    """Raised when a pipeline lacks a preprocessing step its model needs.

    Args:
        message: The message, used when ``algorithm`` and ``missing_step`` are not both
            given.
        algorithm: The model's algorithm name (``"random_forest"``).
        missing_step: The step the pipeline needs (``"Encoder"``).

    Attributes:
        algorithm: The algorithm name, or None.
        missing_step: The missing step, or None.
    """

    def __init__(
        self,
        message: str = "Missing required preprocessing steps.",
        algorithm: Optional[str] = None,
        missing_step: Optional[str] = None,
    ) -> None:
        if algorithm and missing_step:
            message = (
                f"{algorithm} requires {missing_step} in pipeline. "
                f"Add {missing_step} before the model step."
            )
        self.algorithm = algorithm
        self.missing_step = missing_step
        super().__init__(message)


class ModelNotFittedError(SmallaxeError):
    """Raised when a model, pipeline or preprocessing step is used before ``fit``.

    Args:
        message: The error message.
    """

    def __init__(
        self, message: str = "Model has not been fitted. Call fit() before predict()."
    ) -> None:
        super().__init__(message)


class ColumnNotFoundError(SmallaxeError):
    """Raised when a required column is missing from the DataFrame.

    Args:
        message: The message, used when ``column`` is not given.
        column: The missing column.
        available_columns: The DataFrame's columns, appended to the message.

    Attributes:
        column: The missing column, or None.
        available_columns: The DataFrame's columns, or None.
    """

    def __init__(
        self,
        message: str = "Required column not found in DataFrame.",
        column: Optional[str] = None,
        available_columns: Optional[List[str]] = None,
    ) -> None:
        if column:
            message = f"Column '{column}' not found in DataFrame."
            if available_columns:
                message += f" Available columns: {available_columns}"
        self.column = column
        self.available_columns = available_columns
        super().__init__(message)


class DependencyError(SmallaxeError):
    """Raised when an algorithm's optional package is not installed or cannot be imported.

    Args:
        message: The message, used when ``package`` is not given.
        package: The missing package.
        install_command: How to install it, appended to the message.

    Attributes:
        package: The missing package, or None.
        install_command: The install hint, or None.
    """

    def __init__(
        self,
        message: str = "Missing optional dependency.",
        package: Optional[str] = None,
        install_command: Optional[str] = None,
    ) -> None:
        if package:
            message = f"{package} is not installed."
            if install_command:
                message += f" Install with: {install_command}"
        self.package = package
        self.install_command = install_command
        super().__init__(message)


class ConfigurationError(SmallaxeError):
    """Raised when a global setting (``set_verbosity``, ``set_seed``) gets an invalid value.

    Args:
        message: The message, used when ``setting`` and ``value`` are not both given.
        setting: The setting's name (``"verbosity"``).
        value: The rejected value. Any value counts, falsy ones (``""``, ``0``, ``None``)
            included; leave it out to keep ``message``.
        allowed_values: The values the setting accepts, appended to the message.

    Attributes:
        setting: The setting's name, or None.
        value: The rejected value, or None when none was given.
        allowed_values: The accepted values, or None.
    """

    def __init__(
        self,
        message: str = "Invalid configuration settings.",
        setting: Optional[str] = None,
        value: Any = _NO_VALUE,
        allowed_values: Optional[List[str]] = None,
    ) -> None:
        if setting is not None and value is not _NO_VALUE:
            message = f"Invalid value {value!r} for setting '{setting}'."
            if allowed_values:
                message += f" Allowed values: {allowed_values}"
        self.setting = setting
        self.value = None if value is _NO_VALUE else value
        self.allowed_values = allowed_values
        super().__init__(message)
