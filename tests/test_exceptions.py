"""Tests for smallaxe exception classes."""

import pytest

from smallaxe.exceptions import (
    ColumnNotFoundError,
    ConfigurationError,
    DependencyError,
    ModelNotFittedError,
    PreprocessingError,
    SmallaxeError,
    ValidationError,
)


class TestSmallaxeError:
    """Tests for the base SmallaxeError class."""

    def test_raise_and_catch(self):
        """Test that SmallaxeError can be raised and caught."""
        with pytest.raises(SmallaxeError):
            raise SmallaxeError()

    def test_default_message(self):
        """Test default error message."""
        error = SmallaxeError()
        assert "error occurred in smallaxe" in str(error).lower()

    def test_custom_message(self):
        """Test custom error message."""
        error = SmallaxeError("Custom error message")
        assert str(error) == "Custom error message"
        assert error.message == "Custom error message"


class TestValidationError:
    """Tests for ValidationError."""

    def test_inheritance(self):
        """Test that ValidationError inherits from SmallaxeError."""
        error = ValidationError()
        assert isinstance(error, SmallaxeError)
        assert isinstance(error, Exception)

    def test_catch_as_base(self):
        """Test that ValidationError can be caught as SmallaxeError."""
        with pytest.raises(SmallaxeError):
            raise ValidationError()

    def test_default_message(self):
        """Test default error message."""
        error = ValidationError()
        assert "invalid" in str(error).lower()

    def test_custom_message(self):
        """Test custom error message."""
        error = ValidationError("Parameter 'x' must be positive")
        assert str(error) == "Parameter 'x' must be positive"


class TestPreprocessingError:
    """Tests for PreprocessingError."""

    def test_inheritance(self):
        """Test that PreprocessingError inherits from SmallaxeError."""
        error = PreprocessingError()
        assert isinstance(error, SmallaxeError)

    def test_default_message(self):
        """Test default error message."""
        error = PreprocessingError()
        assert "preprocessing" in str(error).lower()

    def test_with_algorithm_and_step(self):
        """Test error message with algorithm and missing step."""
        error = PreprocessingError(algorithm="Random Forest", missing_step="Scaler")
        assert "Random Forest" in str(error)
        assert "Scaler" in str(error)
        assert error.algorithm == "Random Forest"
        assert error.missing_step == "Scaler"


class TestModelNotFittedError:
    """Tests for ModelNotFittedError."""

    def test_inheritance(self):
        """Test that ModelNotFittedError inherits from SmallaxeError."""
        error = ModelNotFittedError()
        assert isinstance(error, SmallaxeError)

    def test_default_message(self):
        """Test default error message."""
        error = ModelNotFittedError()
        assert "fit" in str(error).lower()
        assert "predict" in str(error).lower()

    def test_custom_message(self):
        """Test custom error message."""
        error = ModelNotFittedError("Pipeline not fitted")
        assert str(error) == "Pipeline not fitted"


class TestColumnNotFoundError:
    """Tests for ColumnNotFoundError."""

    def test_inheritance(self):
        """Test that ColumnNotFoundError inherits from SmallaxeError."""
        error = ColumnNotFoundError()
        assert isinstance(error, SmallaxeError)

    def test_default_message(self):
        """Test default error message."""
        error = ColumnNotFoundError()
        assert "column" in str(error).lower()

    def test_with_column_name(self):
        """Test error message with column name."""
        error = ColumnNotFoundError(column="target")
        assert "target" in str(error)
        assert error.column == "target"

    def test_with_available_columns(self):
        """Test error message with available columns."""
        error = ColumnNotFoundError(column="target", available_columns=["id", "age", "income"])
        assert "target" in str(error)
        assert "id" in str(error)
        assert "age" in str(error)
        assert error.available_columns == ["id", "age", "income"]


class TestDependencyError:
    """Tests for DependencyError."""

    def test_inheritance(self):
        """Test that DependencyError inherits from SmallaxeError."""
        error = DependencyError()
        assert isinstance(error, SmallaxeError)

    def test_default_message(self):
        """Test default error message."""
        error = DependencyError()
        assert "dependency" in str(error).lower()

    def test_with_package(self):
        """Test error message with package name."""
        error = DependencyError(package="xgboost")
        assert "xgboost" in str(error)
        assert error.package == "xgboost"

    def test_with_install_command(self):
        """Test error message with install command."""
        error = DependencyError(package="xgboost", install_command="pip install smallaxe[xgboost]")
        assert "xgboost" in str(error)
        assert "pip install smallaxe[xgboost]" in str(error)
        assert error.install_command == "pip install smallaxe[xgboost]"


class TestConfigurationError:
    """Tests for ConfigurationError."""

    def test_inheritance(self):
        """Test that ConfigurationError inherits from SmallaxeError."""
        error = ConfigurationError()
        assert isinstance(error, SmallaxeError)

    def test_default_message(self):
        """Test default error message."""
        error = ConfigurationError()
        assert "configuration" in str(error).lower()

    def test_with_setting_and_value(self):
        """Test error message with setting and value."""
        error = ConfigurationError(setting="verbosity", value="invalid")
        assert "verbosity" in str(error)
        assert "invalid" in str(error)
        assert error.setting == "verbosity"
        assert error.value == "invalid"

    def test_with_allowed_values(self):
        """Test error message with allowed values."""
        error = ConfigurationError(
            setting="verbosity", value="invalid", allowed_values=["quiet", "normal", "verbose"]
        )
        assert "verbosity" in str(error)
        assert "invalid" in str(error)
        assert "quiet" in str(error)
        assert error.allowed_values == ["quiet", "normal", "verbose"]


class TestExceptionHierarchy:
    """Tests for the exception hierarchy."""

    def test_all_exceptions_inherit_from_base(self):
        """Test that all exceptions inherit from SmallaxeError."""
        exceptions = [
            ValidationError,
            PreprocessingError,
            ModelNotFittedError,
            ColumnNotFoundError,
            DependencyError,
            ConfigurationError,
        ]
        for exc_class in exceptions:
            assert issubclass(exc_class, SmallaxeError)
            assert issubclass(exc_class, Exception)

    def test_catch_all_with_base(self):
        """Test that all exceptions can be caught with SmallaxeError."""
        exceptions = [
            ValidationError("test"),
            PreprocessingError("test"),
            ModelNotFittedError("test"),
            ColumnNotFoundError("test"),
            DependencyError("test"),
            ConfigurationError("test"),
        ]
        for exc in exceptions:
            with pytest.raises(SmallaxeError):
                raise exc
