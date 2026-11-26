"""smallaxe: train, validate, tune and save PySpark ML models with a scikit-learn-like API.

The top-level package holds the global settings (verbosity and seed) and the loaders
that read any saved artifact. ``import smallaxe`` imports no training code and no
optional algorithm package; each subpackage is imported when it is first used:

- :mod:`smallaxe.training`: models (``regressors.<algorithm>()``,
  ``classifiers.<algorithm>()``) and the algorithm records.
- :mod:`smallaxe.preprocessing`: ``Imputer``, ``Scaler``, ``Encoder``.
- :mod:`smallaxe.pipeline`: ``Pipeline``.
- :mod:`smallaxe.metrics`: regression and classification metrics.
- :mod:`smallaxe.search`: hyperparameter search (``optimize.run``).
- :mod:`smallaxe.auto`: ``AutomatedTraining``, every installed algorithm compared on one split.
- :mod:`smallaxe.viz`: plotly charts that aggregate in Spark (the ``viz`` extra).
- :mod:`smallaxe.datasets`: sample datasets.
- :mod:`smallaxe.exceptions`: the exception hierarchy.
"""

from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Generator, Optional, Union, cast

from smallaxe._config import VALID_VERBOSITY_LEVELS, _config
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
    "set_seed",
    "get_seed",
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

    The level decides two things today: whether :func:`smallaxe.search.optimize.run`
    shows hyperopt's progress bar when its ``verbose`` argument is None, and whether
    :class:`smallaxe.auto.AutomatedTraining` prints a progress line per algorithm.
    Warnings (``UserWarning``) are emitted at every level; silence them with the
    ``warnings`` module.

    Args:
        level: One of:
            - ``'quiet'``: no progress bar, no progress lines.
            - ``'normal'`` (default): ``optimize.run`` shows its progress bar and
              ``AutomatedTraining.fit`` prints its progress lines.
            - ``'verbose'``: the same as ``'normal'``; reserved for more output in a
              later release.

    Raises:
        ConfigurationError: If level is not a valid verbosity level.

    Example:
        >>> smallaxe.set_verbosity('quiet')
    """
    if level not in VALID_VERBOSITY_LEVELS:
        raise ConfigurationError(
            setting="verbosity",
            value=level,
            allowed_values=list(VALID_VERBOSITY_LEVELS),
        )
    _config._verbosity = level


def get_verbosity() -> str:
    """Get the current verbosity level (see :func:`set_verbosity`).

    Returns:
        The current verbosity level ('quiet', 'normal', or 'verbose').
    """
    return _config._verbosity


@contextmanager
def verbosity(level: str) -> Generator[None, None, None]:
    """Context manager that sets the verbosity level for the duration of a block.

    The previous level is restored on exit, also when the block raises.

    Args:
        level: Verbosity level to use within the context (see :func:`set_verbosity`).

    Yields:
        None.

    Raises:
        ConfigurationError: If level is not a valid verbosity level.

    Example:
        >>> with smallaxe.verbosity('quiet'):
        ...     result = optimize.run(model, dataframe, 'target', space)  # no progress bar
    """
    previous_level = get_verbosity()
    set_verbosity(level)
    try:
        yield
    finally:
        _config._verbosity = previous_level


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
        ConfigurationError: If seed is not an integer or None (a bool is rejected).

    Example:
        >>> smallaxe.set_seed(42)
    """
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
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
