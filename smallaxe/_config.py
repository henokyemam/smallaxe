"""Internal configuration state for smallaxe."""

from typing import Any, Optional

# Valid configuration values
VALID_VERBOSITY_LEVELS = ("quiet", "normal", "verbose")
VALID_CACHE_STRATEGIES = ("auto", "always", "never")

# Default configuration values
DEFAULT_VERBOSITY = "normal"
DEFAULT_CACHE_STRATEGY = "auto"
DEFAULT_SEED = None


class _Config:
    """Internal configuration state container.

    This class holds the global configuration state for smallaxe.
    It should not be accessed directly - use the module-level functions instead.
    """

    def __init__(self) -> None:
        self._verbosity: str = DEFAULT_VERBOSITY
        self._cache_strategy: str = DEFAULT_CACHE_STRATEGY
        self._seed: Optional[int] = DEFAULT_SEED
        self._spark_session: Optional[Any] = None

    def reset(self) -> None:
        """Reset configuration to defaults."""
        self._verbosity = DEFAULT_VERBOSITY
        self._cache_strategy = DEFAULT_CACHE_STRATEGY
        self._seed = DEFAULT_SEED
        self._spark_session = None


# Global configuration instance
_config = _Config()
