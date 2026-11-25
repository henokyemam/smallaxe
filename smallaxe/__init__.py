"""
smallaxe - A PySpark MLOps library for simplified model training and optimization.
"""

import warnings
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any, Generator, Optional, Union, cast

from smallaxe._config import (
    VALID_CACHE_STRATEGIES,
    VALID_VERBOSITY_LEVELS,
    _config,
)
from smallaxe.exceptions import ConfigurationError, ValidationError

if TYPE_CHECKING:  # pragma: no cover - these are imported only when used
    from smallaxe.pipeline import Pipeline
    from smallaxe.preprocessing import Encoder, Imputer, Scaler
    from smallaxe.training.base import Classifier, Regressor

try:
    __version__ = version("smallaxe")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"

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
    "load_model",
    "load_pipeline",
    "load",
]


def load_model(path: str) -> Union["Regressor", "Classifier"]:
    """Load the model saved at ``path``, whatever its algorithm and task.

    Reads artifacts written by ``model.save`` in this and every earlier smallaxe
    release (0.8.x artifacts record the old class name, 0.9 and 0.10 ones have no
    ``format_version``). The training package is imported here, so ``import smallaxe``
    stays light.

    Args:
        path: The artifact directory (local, ``dbfs:/``, ``s3://``, ``hdfs://``, ...).

    Returns:
        The fitted :class:`~smallaxe.training.Regressor` or
        :class:`~smallaxe.training.Classifier`.

    Raises:
        ValidationError: If the path or its ``metadata.json`` does not exist, it holds
            another kind of artifact or was written by a newer smallaxe, or its metadata
            is invalid (see :meth:`smallaxe.training.base.Model.load`).
        DependencyError: If the saved model's algorithm is not installed.

    Example:
        >>> model = smallaxe.load_model('/path/to/model')
        >>> predictions = model.predict(dataframe)
    """
    from smallaxe.training.base import Model

    return cast("Union[Regressor, Classifier]", Model.load(path))


def load_pipeline(path: str) -> "Pipeline":
    """Load the pipeline saved at ``path``.

    Reads pipelines written by ``Pipeline.save`` in this release and preprocessing-only
    pipelines written by smallaxe 0.10 (see :meth:`smallaxe.pipeline.Pipeline.load`).

    Args:
        path: The artifact directory (local, ``dbfs:/``, ``s3://``, ``hdfs://``, ...).

    Returns:
        The fitted :class:`~smallaxe.pipeline.Pipeline`.

    Raises:
        ValidationError: If the path or its ``metadata.json`` does not exist, it holds
            another kind of artifact or was written by a newer smallaxe, a step is
            missing, or its metadata is invalid.
        DependencyError: If the model step's algorithm is not installed.

    Example:
        >>> pipeline = smallaxe.load_pipeline('/path/to/pipeline')
        >>> predictions = pipeline.predict(dataframe)
    """
    from smallaxe.pipeline import Pipeline

    return Pipeline.load(path)


def load(
    path: str,
) -> Union["Regressor", "Classifier", "Pipeline", "Imputer", "Scaler", "Encoder"]:
    """Load any smallaxe artifact, returning the kind of object that saved it.

    The kind is read from the artifact (``kind`` in its ``metadata.json``); unversioned
    model and pipeline artifacts of smallaxe 0.10 and earlier, and 0.10 Imputer and
    Scaler artifacts, are recognised by their layout. Use :func:`load_model` or
    :func:`load_pipeline` to also check that the artifact is the kind you expect.

    Args:
        path: The artifact directory (local, ``dbfs:/``, ``s3://``, ``hdfs://``, ...).

    Returns:
        A fitted ``Regressor`` / ``Classifier``, ``Pipeline``, ``Imputer``, ``Scaler`` or
        ``Encoder``.

    Raises:
        ValidationError: If the path does not exist, holds no smallaxe artifact or one of
            a kind this smallaxe does not know, or the artifact itself is invalid.
        DependencyError: If a saved model's algorithm is not installed.

    Example:
        >>> obj = smallaxe.load('/path/to/artifact')
    """
    from smallaxe import _artifact

    kind = _artifact.saved_kind(path)
    if kind == "model":
        return load_model(path)
    if kind == "pipeline":
        return load_pipeline(path)
    if kind in ("imputer", "scaler", "encoder"):
        from smallaxe.preprocessing import Encoder, Imputer, Scaler

        if kind == "imputer":
            return Imputer.load(path)
        if kind == "scaler":
            return Scaler.load(path)
        return Encoder.load(path)
    raise ValidationError(
        f"Cannot load artifact: {path!r} holds a {kind!r} artifact, which this smallaxe "
        f"({__version__}) does not know. It may have been written by a newer smallaxe."
    )


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
        ...     model.fit(dataframe, label_col='target')  # runs silently
    """
    previous_level = get_verbosity()
    set_verbosity(level)
    try:
        yield
    finally:
        _config._verbosity = previous_level


def set_spark_session(spark: Any) -> None:
    """Deprecated: smallaxe reads the session from the DataFrame it is given.

    Stores the value for :func:`get_spark_session`, but no library code uses it.
    Will be removed in 1.0.
    """
    warnings.warn(
        "smallaxe.set_spark_session has no effect and will be removed in 1.0. smallaxe "
        "uses the session of the DataFrame it is given, or the active session.",
        DeprecationWarning,
        stacklevel=2,
    )
    _config._spark_session = spark


def get_spark_session() -> Optional[Any]:
    """Deprecated: returns whatever :func:`set_spark_session` stored. Removed in 1.0."""
    warnings.warn(
        "smallaxe.get_spark_session will be removed in 1.0; use dataframe.sparkSession or "
        "SparkSession.getActiveSession().",
        DeprecationWarning,
        stacklevel=2,
    )
    return _config._spark_session


def set_seed(seed: Optional[int]) -> None:
    """Set the global random seed for reproducibility.

    Used wherever a call or model has no seed of its own: train/test splits and
    k-fold assignment, the estimator's seed when the model's ``seed`` parameter
    is None, and hyperopt's sampler in ``optimize.run`` when its ``seed`` is None.
    With the same data and seed, folds and trained models are reproducible,
    independent of partitioning and caching. A model's own ``seed`` parameter
    takes precedence.

    Args:
        seed: Integer seed value, or None to reset to no seed.

    Raises:
        ConfigurationError: If seed is not an integer or None.
    """
    if seed is not None and not isinstance(seed, int):
        raise ConfigurationError(
            message=f"Seed must be an integer or None, got {type(seed).__name__}."
        )
    _config._seed = seed


def get_seed() -> Optional[int]:
    """Get the current random seed.

    Returns:
        The current seed value, or None if not set.
    """
    return _config._seed


def set_cache_strategy(strategy: str) -> None:
    """Deprecated: has no effect and will be removed in 1.0.

    Caching is decided per ``fit`` call through its ``cache_strategy`` argument
    (``None``, ``'none'``, ``'memory'``, ``'disk'``), because it depends on the
    DataFrame, not on the process.

    Raises:
        ConfigurationError: If strategy is not one of 'auto', 'always', 'never'.
    """
    warnings.warn(
        "smallaxe.set_cache_strategy has no effect and will be removed in 1.0. Pass "
        "cache_strategy= to fit() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if strategy not in VALID_CACHE_STRATEGIES:
        raise ConfigurationError(
            setting="cache_strategy",
            value=strategy,
            allowed_values=list(VALID_CACHE_STRATEGIES),
        )
    _config._cache_strategy = strategy


def get_cache_strategy() -> str:
    """Deprecated: returns whatever :func:`set_cache_strategy` stored. Removed in 1.0."""
    warnings.warn(
        "smallaxe.get_cache_strategy will be removed in 1.0; caching is set per fit() call.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _config._cache_strategy
