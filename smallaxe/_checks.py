"""Argument checks shared by the public entry points.

A DataFrame, a list of column names, a new column's name, and Spark's own argument
errors turned into ``ValidationError``; plus :func:`one_row`, for reading a global
aggregate.
"""

from contextlib import contextmanager
from typing import Any, Iterator, List, Optional, Sequence

from pyspark.sql import DataFrame, Row
from pyspark.sql.utils import IllegalArgumentException

from smallaxe.exceptions import ValidationError

__all__ = [
    "require_dataframe",
    "column_names",
    "output_column",
    "spark_argument_errors",
    "one_row",
]


def require_dataframe(dataframe: Any, name: str = "dataframe") -> None:
    """Raise ValidationError unless ``dataframe`` is a Spark DataFrame.

    Checked by shape (``schema`` and ``columns``) rather than class, so Spark Connect
    DataFrames, which are not ``pyspark.sql.DataFrame`` subclasses before Spark 4, pass.

    Args:
        dataframe: The value to check.
        name: The argument's name, for the message.

    Raises:
        ValidationError: If ``dataframe`` has no ``schema`` or ``columns``.
    """
    if not (hasattr(dataframe, "schema") and hasattr(dataframe, "columns")):
        raise ValidationError(
            f"{name} must be a PySpark DataFrame, got {type(dataframe).__name__}."
        )


def column_names(name: str, value: Optional[Sequence[Any]]) -> Optional[List[str]]:
    """``value`` as a list of column names, or None; a lone string is rejected.

    Args:
        name: The argument's name, for the message.
        value: None, or a list or tuple of column names.

    Returns:
        A new list, or None when ``value`` is None.

    Raises:
        ValidationError: If ``value`` is a string or anything but a list or tuple of
            strings.
    """
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or not all(isinstance(col, str) for col in value):
        raise ValidationError(f"{name} must be a list of column names, got {value!r}.")
    return list(value)


def output_column(name: str, value: Any) -> str:
    """``value`` if it is a usable name for a new column.

    Args:
        name: The argument's name, for the message.
        value: The proposed column name.

    Returns:
        ``value``, unchanged.

    Raises:
        ValidationError: If ``value`` is not a non-empty string.
    """
    if not isinstance(value, str) or not value:
        raise ValidationError(f"{name} must be a non-empty column name, got {value!r}.")
    return value


@contextmanager
def spark_argument_errors(context: str) -> Iterator[None]:
    """Turn Spark's ``IllegalArgumentException`` inside the block into ``ValidationError``.

    Spark checks some arguments only when a job starts: an estimator parameter out of its
    range (``maxDepth=-3``), an evaluator column of the wrong type. Those are user errors,
    so they surface as ``ValidationError`` with Spark's message, chained to the original.

    Args:
        context: What was being done, for the message (``"Cannot fit 'random_forest'"``).

    Raises:
        ValidationError: If the block raises Spark's ``IllegalArgumentException``.
    """
    try:
        yield
    except IllegalArgumentException as exc:
        detail = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
        raise ValidationError(f"{context}: {detail}") from exc


def one_row(dataframe: DataFrame) -> Row:
    """The single row of a frame that always has exactly one.

    A global aggregate (``select`` of aggregate expressions, or ``agg`` without
    ``groupBy``) and ``spark.range(1).select(...)`` always return one row, even over an
    empty input, so ``first()`` is never None here; the assertion records that.

    Args:
        dataframe: A one-row frame, such as ``df.select(F.mean("x"))``.

    Returns:
        Its row.
    """
    row = dataframe.first()
    assert row is not None, "a global aggregate always returns one row"
    return row
