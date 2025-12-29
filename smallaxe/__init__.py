"""
smallaxe - A PySpark MLOps library for simplified model training and optimization.
"""

from contextlib import contextmanager
from typing import Any, Generator, Optional

from smallaxe._config import (
    VALID_CACHE_STRATEGIES,
    VALID_VERBOSITY_LEVELS,
    _config,
)
from smallaxe.exceptions import ConfigurationError

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "set_verbosity",
    "get_verbosity",
    "verbosity",
    "set_spark_session",
    "get_spark_session",
    "set_seed",
    "get_seed",
    "set_cache_strategy",
    "get_cache_strategy",
]


def set_verbosity(level: str) -> None:
    """Set the global verbosity level.

    Args:
        level: Verbosity level. One of 'quiet', 'normal', or 'verbose'.
            - 'quiet': Only errors, no progress bars or info messages
            - 'normal': Progress bars and key info (default)
            - 'verbose': Detailed logging for debugging

    Raises:
        ConfigurationError: If level is not a valid verbosity level.
    """
    if level not in VALID_VERBOSITY_LEVELS:
        raise ConfigurationError(
            setting="verbosity",
            value=level,
            allowed_values=list(VALID_VERBOSITY_LEVELS),
        )
    _config._verbosity = level


def get_verbosity() -> str:
    """Get the current verbosity level.

    Returns:
        The current verbosity level ('quiet', 'normal', or 'verbose').
    """
    return _config._verbosity


@contextmanager
def verbosity(level: str) -> Generator[None, None, None]:
    """Context manager for temporarily changing verbosity level.

    Args:
        level: Verbosity level to use within the context.

    Raises:
        ConfigurationError: If level is not a valid verbosity level.

    Example:
        >>> with smallaxe.verbosity('quiet'):
        ...     model.fit(df, label_col='target')  # runs silently
    """
    previous_level = get_verbosity()
    set_verbosity(level)
    try:
        yield
    finally:
        _config._verbosity = previous_level


def set_spark_session(spark: Any) -> None:
    """Set the Spark session to use.

    Args:
        spark: A SparkSession instance. If None, smallaxe will attempt
            to get or create a session when needed.
    """
    _config._spark_session = spark


def get_spark_session() -> Optional[Any]:
    """Get the configured Spark session.

    Returns:
        The configured SparkSession, or None if not set.
    """
    return _config._spark_session


def set_seed(seed: int) -> None:
    """Set the global random seed for reproducibility.

    This affects all random operations including train/test splits,
    k-fold cross-validation, and hyperopt sampling.

    Args:
        seed: Integer seed value.

    Raises:
        ConfigurationError: If seed is not an integer.
    """
    if not isinstance(seed, int):
        raise ConfigurationError(message=f"Seed must be an integer, got {type(seed).__name__}.")
    _config._seed = seed


def get_seed() -> Optional[int]:
    """Get the current random seed.

    Returns:
        The current seed value, or None if not set.
    """
    return _config._seed


def set_cache_strategy(strategy: str) -> None:
    """Set the caching strategy for PySpark operations.

    Args:
        strategy: Cache strategy. One of 'auto', 'always', or 'never'.
            - 'auto': Smart caching - cache after preprocessing, unpersist after training
            - 'always': Cache at every stage (use for debugging or small datasets)
            - 'never': No automatic caching (manual control)

    Raises:
        ConfigurationError: If strategy is not a valid cache strategy.
    """
    if strategy not in VALID_CACHE_STRATEGIES:
        raise ConfigurationError(
            setting="cache_strategy",
            value=strategy,
            allowed_values=list(VALID_CACHE_STRATEGIES),
        )
    _config._cache_strategy = strategy


def get_cache_strategy() -> str:
    """Get the current cache strategy.

    Returns:
        The current cache strategy ('auto', 'always', or 'never').
    """
    return _config._cache_strategy
