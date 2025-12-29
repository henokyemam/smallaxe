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

from typing import List, Optional

__all__ = [
    "SmallaxeError",
    "ValidationError",
    "PreprocessingError",
    "ModelNotFittedError",
    "ColumnNotFoundError",
    "DependencyError",
    "ConfigurationError",
]


class SmallaxeError(Exception):
    """Base exception for all smallaxe errors."""

    def __init__(self, message: str = "An error occurred in smallaxe."):
        self.message = message
        super().__init__(self.message)


class ValidationError(SmallaxeError):
    """Raised when input parameters or data are invalid."""

    def __init__(self, message: str = "Invalid input parameters or data."):
        super().__init__(message)


class PreprocessingError(SmallaxeError):
    """Raised when required preprocessing steps are missing."""

    def __init__(
        self,
        message: str = "Missing required preprocessing steps.",
        algorithm: Optional[str] = None,
        missing_step: Optional[str] = None,
    ):
        if algorithm and missing_step:
            message = (
                f"{algorithm} requires {missing_step} in pipeline. "
                f"Add {missing_step} before the model step."
            )
        self.algorithm = algorithm
        self.missing_step = missing_step
        super().__init__(message)


class ModelNotFittedError(SmallaxeError):
    """Raised when predict() is called before fit()."""

    def __init__(self, message: str = "Model has not been fitted. Call fit() before predict()."):
        super().__init__(message)


class ColumnNotFoundError(SmallaxeError):
    """Raised when a required column is missing from the DataFrame."""

    def __init__(
        self,
        message: str = "Required column not found in DataFrame.",
        column: Optional[str] = None,
        available_columns: Optional[List[str]] = None,
    ):
        if column:
            message = f"Column '{column}' not found in DataFrame."
            if available_columns:
                message += f" Available columns: {available_columns}"
        self.column = column
        self.available_columns = available_columns
        super().__init__(message)


class DependencyError(SmallaxeError):
    """Raised when an optional dependency is not installed."""

    def __init__(
        self,
        message: str = "Missing optional dependency.",
        package: Optional[str] = None,
        install_command: Optional[str] = None,
    ):
        if package:
            message = f"{package} is not installed."
            if install_command:
                message += f" Install with: {install_command}"
        self.package = package
        self.install_command = install_command
        super().__init__(message)


class ConfigurationError(SmallaxeError):
    """Raised when configuration settings are invalid."""

    def __init__(
        self,
        message: str = "Invalid configuration settings.",
        setting: Optional[str] = None,
        value: Optional[str] = None,
        allowed_values: Optional[List[str]] = None,
    ):
        if setting and value:
            message = f"Invalid value '{value}' for setting '{setting}'."
            if allowed_values:
                message += f" Allowed values: {allowed_values}"
        self.setting = setting
        self.value = value
        self.allowed_values = allowed_values
        super().__init__(message)
