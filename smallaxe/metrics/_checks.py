"""Input checks shared by the metric functions."""

from typing import List

from pyspark.sql import DataFrame
from pyspark.sql.types import NumericType

from smallaxe._checks import require_dataframe
from smallaxe.exceptions import ColumnNotFoundError, ValidationError

__all__ = ["validate_columns", "require_numeric", "require_rows"]


def validate_columns(dataframe: DataFrame, *cols: str) -> None:
    """Check ``dataframe`` is a DataFrame holding every one of ``cols``.

    Raises:
        ValidationError: If ``dataframe`` is not a DataFrame or a column name is not a
            string.
        ColumnNotFoundError: If any of ``cols`` is not a column of ``dataframe``.
    """
    require_dataframe(dataframe)
    for col in cols:
        if not isinstance(col, str):
            raise ValidationError(f"Column names must be strings, got {col!r}.")
    available_columns: List[str] = dataframe.columns
    for col in cols:
        if col not in available_columns:
            raise ColumnNotFoundError(column=col, available_columns=available_columns)


def require_numeric(dataframe: DataFrame, metric: str, *cols: str) -> None:
    """Raise ``ValidationError`` naming ``metric`` if any of ``cols`` is not numeric.

    For the metrics computed with column arithmetic rather than a Spark evaluator:
    without ANSI mode Spark casts a string to null in arithmetic, which would turn a
    wrong input into an "undefined" NaN instead of an error.
    """
    for col in cols:
        data_type = dataframe.schema[col].dataType
        if not isinstance(data_type, NumericType):
            raise ValidationError(
                f"Cannot compute {metric}: column '{col}' must be numeric, "
                f"got {data_type.simpleString()}."
            )


def require_rows(dataframe: DataFrame, metric: str) -> None:
    """Raise ``ValidationError`` naming ``metric`` if ``dataframe`` has no rows.

    A metric of nothing is not a score: an empty frame usually means an empty
    validation split or a filter that removed everything.
    """
    if dataframe.limit(1).count() == 0:
        raise ValidationError(f"Cannot compute {metric}: the DataFrame has no rows.")
