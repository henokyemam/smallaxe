"""Tests for smallaxe configuration module."""

import pytest

import smallaxe
from smallaxe._config import _config
from smallaxe.exceptions import ConfigurationError


@pytest.fixture(autouse=True)
def reset_config():
    """Reset configuration to defaults before each test."""
    _config.reset()
    yield
    _config.reset()


class TestVerbosity:
    """Tests for verbosity configuration."""

    def test_get_verbosity_default(self):
        """Test that default verbosity is 'normal'."""
        assert smallaxe.get_verbosity() == "normal"

    def test_set_verbosity_quiet(self):
        """Test setting verbosity to 'quiet'."""
        smallaxe.set_verbosity("quiet")
        assert smallaxe.get_verbosity() == "quiet"

    def test_set_verbosity_normal(self):
        """Test setting verbosity to 'normal'."""
        smallaxe.set_verbosity("verbose")  # Change first
        smallaxe.set_verbosity("normal")
        assert smallaxe.get_verbosity() == "normal"

    def test_set_verbosity_verbose(self):
        """Test setting verbosity to 'verbose'."""
        smallaxe.set_verbosity("verbose")
        assert smallaxe.get_verbosity() == "verbose"

    def test_set_verbosity_invalid_raises_error(self):
        """Test that invalid verbosity raises ConfigurationError."""
        with pytest.raises(ConfigurationError) as exc_info:
            smallaxe.set_verbosity("invalid")

        assert exc_info.value.setting == "verbosity"
        assert exc_info.value.value == "invalid"
        assert "quiet" in str(exc_info.value.allowed_values)
        assert "normal" in str(exc_info.value.allowed_values)
        assert "verbose" in str(exc_info.value.allowed_values)

    def test_set_verbosity_none_raises_error(self):
        """Test that None verbosity raises ConfigurationError."""
        with pytest.raises(ConfigurationError):
            smallaxe.set_verbosity(None)


class TestVerbosityContextManager:
    """Tests for verbosity context manager."""

    def test_verbosity_context_manager_changes_level(self):
        """Test that context manager changes verbosity level."""
        assert smallaxe.get_verbosity() == "normal"

        with smallaxe.verbosity("quiet"):
            assert smallaxe.get_verbosity() == "quiet"

    def test_verbosity_context_manager_restores_level(self):
        """Test that context manager restores previous level on exit."""
        smallaxe.set_verbosity("verbose")

        with smallaxe.verbosity("quiet"):
            assert smallaxe.get_verbosity() == "quiet"

        assert smallaxe.get_verbosity() == "verbose"

    def test_verbosity_context_manager_restores_on_exception(self):
        """Test that context manager restores level even on exception."""
        smallaxe.set_verbosity("verbose")

        with pytest.raises(ValueError):
            with smallaxe.verbosity("quiet"):
                assert smallaxe.get_verbosity() == "quiet"
                raise ValueError("Test exception")

        assert smallaxe.get_verbosity() == "verbose"

    def test_verbosity_context_manager_invalid_raises_error(self):
        """Test that invalid level in context manager raises error."""
        with pytest.raises(ConfigurationError):
            with smallaxe.verbosity("invalid"):
                pass

    def test_verbosity_context_manager_nested(self):
        """Test nested context managers work correctly."""
        assert smallaxe.get_verbosity() == "normal"

        with smallaxe.verbosity("quiet"):
            assert smallaxe.get_verbosity() == "quiet"

            with smallaxe.verbosity("verbose"):
                assert smallaxe.get_verbosity() == "verbose"

            assert smallaxe.get_verbosity() == "quiet"

        assert smallaxe.get_verbosity() == "normal"


class TestSparkSession:
    """Tests for Spark session configuration."""

    def test_get_spark_session_default(self):
        """Test that default spark session is None."""
        assert smallaxe.get_spark_session() is None

    def test_set_spark_session(self, spark_session):
        """Test setting a Spark session."""
        smallaxe.set_spark_session(spark_session)
        assert smallaxe.get_spark_session() is spark_session

    def test_set_spark_session_none(self):
        """Test setting spark session to None."""
        smallaxe.set_spark_session("mock_session")
        smallaxe.set_spark_session(None)
        assert smallaxe.get_spark_session() is None

    def test_set_spark_session_accepts_any_object(self):
        """Test that set_spark_session accepts any object (duck typing)."""
        mock_session = object()
        smallaxe.set_spark_session(mock_session)
        assert smallaxe.get_spark_session() is mock_session


class TestSeed:
    """Tests for random seed configuration."""

    def test_get_seed_default(self):
        """Test that default seed is None."""
        assert smallaxe.get_seed() is None

    def test_set_seed(self):
        """Test setting a seed value."""
        smallaxe.set_seed(42)
        assert smallaxe.get_seed() == 42

    def test_set_seed_zero(self):
        """Test setting seed to zero."""
        smallaxe.set_seed(0)
        assert smallaxe.get_seed() == 0

    def test_set_seed_negative(self):
        """Test setting a negative seed value."""
        smallaxe.set_seed(-1)
        assert smallaxe.get_seed() == -1

    def test_set_seed_large_number(self):
        """Test setting a large seed value."""
        large_seed = 2**31 - 1
        smallaxe.set_seed(large_seed)
        assert smallaxe.get_seed() == large_seed

    def test_set_seed_float_raises_error(self):
        """Test that float seed raises ConfigurationError."""
        with pytest.raises(ConfigurationError) as exc_info:
            smallaxe.set_seed(42.0)

        assert "integer" in str(exc_info.value).lower()

    def test_set_seed_string_raises_error(self):
        """Test that string seed raises ConfigurationError."""
        with pytest.raises(ConfigurationError):
            smallaxe.set_seed("42")

    def test_set_seed_none_raises_error(self):
        """Test that None seed raises ConfigurationError."""
        with pytest.raises(ConfigurationError):
            smallaxe.set_seed(None)


class TestCacheStrategy:
    """Tests for cache strategy configuration."""

    def test_get_cache_strategy_default(self):
        """Test that default cache strategy is 'auto'."""
        assert smallaxe.get_cache_strategy() == "auto"

    def test_set_cache_strategy_auto(self):
        """Test setting cache strategy to 'auto'."""
        smallaxe.set_cache_strategy("never")  # Change first
        smallaxe.set_cache_strategy("auto")
        assert smallaxe.get_cache_strategy() == "auto"

    def test_set_cache_strategy_always(self):
        """Test setting cache strategy to 'always'."""
        smallaxe.set_cache_strategy("always")
        assert smallaxe.get_cache_strategy() == "always"

    def test_set_cache_strategy_never(self):
        """Test setting cache strategy to 'never'."""
        smallaxe.set_cache_strategy("never")
        assert smallaxe.get_cache_strategy() == "never"

    def test_set_cache_strategy_invalid_raises_error(self):
        """Test that invalid cache strategy raises ConfigurationError."""
        with pytest.raises(ConfigurationError) as exc_info:
            smallaxe.set_cache_strategy("invalid")

        assert exc_info.value.setting == "cache_strategy"
        assert exc_info.value.value == "invalid"
        assert "auto" in str(exc_info.value.allowed_values)
        assert "always" in str(exc_info.value.allowed_values)
        assert "never" in str(exc_info.value.allowed_values)

    def test_set_cache_strategy_none_raises_error(self):
        """Test that None cache strategy raises ConfigurationError."""
        with pytest.raises(ConfigurationError):
            smallaxe.set_cache_strategy(None)


class TestConfigReset:
    """Tests for config reset functionality."""

    def test_reset_restores_defaults(self):
        """Test that reset restores all defaults."""
        # Change all settings
        smallaxe.set_verbosity("quiet")
        smallaxe.set_cache_strategy("never")
        smallaxe.set_seed(42)
        smallaxe.set_spark_session("mock")

        # Reset
        _config.reset()

        # Verify defaults
        assert smallaxe.get_verbosity() == "normal"
        assert smallaxe.get_cache_strategy() == "auto"
        assert smallaxe.get_seed() is None
        assert smallaxe.get_spark_session() is None
