"""Input checks and column conversions shared by the metric functions."""

from typing import List

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import BooleanType, NumericType, StringType

from smallaxe._checks import require_dataframe
from smallaxe.exceptions import ColumnNotFoundError, ValidationError

__all__ = ["validate_columns", "require_numeric", "require_rows", "as_number", "comparable"]


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


def _quoted(col: str) -> str:
    """``col`` as a SQL identifier, so a name holding a dot or a backtick is one column."""
    return "`" + col.replace("`", "``") + "`"


def as_number(dataframe: DataFrame, col: str) -> Column:
    """``col`` for comparing with a number, with the same result in and out of ANSI mode.

    Spark compares a boolean or string column with a number through an implicit cast,
    and ANSI mode (the default from Spark 4) changes it: a boolean compared with a
    number is an analysis error, and a string that is not a number raises at run time
    instead of becoming null. The conversion is made explicit here instead:

    - a numeric column is returned as it is;
    - a boolean column as the integer 0 or 1;
    - a string column as the number it spells (``try_cast`` to double, which is null,
      in either mode, for a string that is not a number); ``try_cast`` is a SQL function
      from Spark 3.2, so it is written as a SQL expression;
    - any other type unchanged.

    Args:
        dataframe: The DataFrame holding ``col``.
        col: A column of ``dataframe``.

    Returns:
        The column expression.
    """
    data_type = dataframe.schema[col].dataType
    if isinstance(data_type, BooleanType):
        return F.col(_quoted(col)).cast("int")
    if isinstance(data_type, StringType):
        return F.expr(f"try_cast({_quoted(col)} AS DOUBLE)")
    return F.col(_quoted(col))


def comparable(dataframe: DataFrame, label_col: str, prediction_col: str) -> List[Column]:
    """The label and prediction columns, converted so ``label == prediction`` is mode-independent.

    Two string columns are compared as strings. Otherwise, when either column is a
    number or a boolean, both go through :func:`as_number`, so the comparison is between
    numbers in and out of ANSI mode. Columns of other types are returned unchanged.

    Args:
        dataframe: The DataFrame holding both columns.
        label_col: The label column.
        prediction_col: The prediction column.

    Returns:
        ``[label, prediction]`` column expressions.
    """
    types = [dataframe.schema[col].dataType for col in (label_col, prediction_col)]
    if any(isinstance(t, (NumericType, BooleanType)) for t in types):
        return [as_number(dataframe, col) for col in (label_col, prediction_col)]
    return [F.col(_quoted(col)) for col in (label_col, prediction_col)]
