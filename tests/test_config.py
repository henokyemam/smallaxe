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
        with pytest.raises(ConfigurationError) as exc_info:
            smallaxe.set_verbosity(None)

        assert "None" in str(exc_info.value)
        assert "verbosity" in str(exc_info.value)

    @pytest.mark.parametrize("value", ["", 0, None])
    def test_set_verbosity_falsy_value_names_setting_and_value(self, value):
        """A falsy invalid value still gets the message naming the setting and value."""
        with pytest.raises(ConfigurationError) as exc_info:
            smallaxe.set_verbosity(value)

        message = str(exc_info.value)
        assert message.startswith(f"Invalid value {value!r} for setting 'verbosity'.")
        assert "Allowed values: ['quiet', 'normal', 'verbose']" in message


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

    @pytest.mark.parametrize("value", [True, False])
    def test_set_seed_bool_raises_error(self, value):
        """A bool is an int to Python, but never a seed."""
        with pytest.raises(ConfigurationError, match="bool"):
            smallaxe.set_seed(value)
        assert smallaxe.get_seed() is None

    def test_set_seed_none_resets_seed(self):
        """Test that None seed resets the seed (valid behavior)."""
        smallaxe.set_seed(42)
        assert smallaxe.get_seed() == 42
        smallaxe.set_seed(None)
        assert smallaxe.get_seed() is None


class TestConfigReset:
    """Tests for config reset functionality."""

    def test_reset_restores_defaults(self):
        """Test that reset restores all defaults."""
        # Change all settings
        smallaxe.set_verbosity("quiet")
        smallaxe.set_seed(42)

        # Reset
        _config.reset()

        # Verify defaults
        assert smallaxe.get_verbosity() == "normal"
        assert smallaxe.get_seed() is None
