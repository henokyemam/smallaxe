"""Internal configuration state for smallaxe: the global verbosity and seed."""

from typing import Optional

__all__ = ["VALID_VERBOSITY_LEVELS", "DEFAULT_VERBOSITY", "DEFAULT_SEED"]

# Valid configuration values
VALID_VERBOSITY_LEVELS = ("quiet", "normal", "verbose")

# Default configuration values
DEFAULT_VERBOSITY = "normal"
DEFAULT_SEED: Optional[int] = None


class _Config:
    """Internal configuration state container.

    Holds the global settings. Use ``smallaxe.set_verbosity`` / ``set_seed`` and their
    getters instead of reading it.
    """

    def __init__(self) -> None:
        self._verbosity: str = DEFAULT_VERBOSITY
        self._seed: Optional[int] = DEFAULT_SEED

    def reset(self) -> None:
        """Reset configuration to defaults."""
        self._verbosity = DEFAULT_VERBOSITY
        self._seed = DEFAULT_SEED


# Global configuration instance
_config = _Config()
