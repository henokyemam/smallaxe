"""Input checks and column conversions shared by the metric functions."""

from typing import List, Optional

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    BooleanType,
    DataType,
    DecimalType,
    DoubleType,
    FloatType,
    IntegerType,
    IntegralType,
    NumericType,
    StringType,
)

from smallaxe._checks import require_dataframe
from smallaxe._sql import integral_from_string, quoted
from smallaxe.exceptions import ColumnNotFoundError, ValidationError

__all__ = [
    "validate_columns",
    "require_numeric",
    "require_rows",
    "comparable",
    "is_one",
    "exact_integers",
    "widened",
]


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


# Holds every integral value, and the sum or difference of two of them, exactly.
_WIDE_INTEGER = DecimalType(20, 0)


def exact_integers(dataframe: DataFrame, *cols: str) -> Optional[List[Column]]:
    """``cols`` as ``decimal(20,0)`` when every one is an integer column, else None.

    Integer arithmetic overflows near the type's limits: without ANSI mode the value
    wraps around, with it (the default from Spark 4) Spark raises. A ``decimal(20,0)``
    holds every integral value, and ``a - b`` of two of them is a ``decimal(21,0)``, so
    the arithmetic is exact in either mode; cast to double afterwards, the result is
    the one Spark 3's default mode gives wherever the integer arithmetic did not
    overflow. Columns of other types are left to Spark's own arithmetic, so their
    results do not change.

    Args:
        dataframe: The DataFrame holding ``cols``.
        *cols: Columns of ``dataframe``.

    Returns:
        The cast column expressions, or None if any column is not an integer column.
    """
    if not all(isinstance(dataframe.schema[col].dataType, IntegralType) for col in cols):
        return None
    return [F.col(quoted(col)).cast(_WIDE_INTEGER) for col in cols]


def widened(dataframe: DataFrame, *cols: str) -> List[str]:
    """``cols`` as SQL, with an integer column cast to float when another one is a float.

    Spark 3's default mode computes an integer with a float in float, and ANSI mode
    (the default from Spark 4) in double, so ``int_col - float_col`` and
    ``int_col = float_col`` change with the option. Casting the integer column to float
    gives the default mode's result in either mode. Every other pair of numeric types
    widens the same way in both modes and is left as it is.

    Args:
        dataframe: The DataFrame holding ``cols``.
        *cols: Numeric columns of ``dataframe``.

    Returns:
        One SQL expression per column.
    """
    types = [dataframe.schema[col].dataType for col in cols]
    has_float = any(isinstance(data_type, FloatType) for data_type in types)
    return [
        (
            f"CAST({quoted(col)} AS FLOAT)"
            if has_float and isinstance(data_type, IntegralType)
            else quoted(col)
        )
        for col, data_type in zip(cols, types)
    ]


def _string_as(col: str, data_type: DataType) -> Column:
    """String column ``col`` cast to ``data_type`` as Spark 3's default cast does, in either mode.

    An integral type goes through :func:`smallaxe._sql.integral_from_string`. For a
    floating-point or boolean type, ``try_cast`` (a SQL function from Spark 3.2) reads
    the same strings as the default cast and is null for the others in either mode,
    where ANSI mode's cast raises.
    """
    if isinstance(data_type, IntegralType):
        return integral_from_string(F.col(quoted(col)), data_type)
    return F.expr(f"try_cast({quoted(col)} AS {data_type.simpleString()})")


def comparable(dataframe: DataFrame, label_col: str, prediction_col: str) -> List[Column]:
    """The label and prediction columns, cast so ``label == prediction`` is mode-independent.

    Spark 3's default mode compares two columns of different types through an implicit
    cast, and ANSI mode (the default from Spark 4) changes it: a boolean compared with a
    number is an analysis error, and a string that does not parse raises instead of
    becoming null. The result is Spark 3's default one, with the casts made explicit:

    - a string compared with an integer column is read as that integer type (``"7.9"``
      is 7), with a float or double column as that type, with a decimal column as a
      double (both sides), and with a boolean column as a boolean (``"true"``, ``"1"``,
      ``"yes"``); a string that does not parse is null, so it never matches;
    - a boolean compared with a number is cast to the number's type (0 or 1);
    - an integer compared with a float is cast to float (:func:`widened`);
    - any other pair (two strings, two numbers, two booleans) is compared as it is.

    Args:
        dataframe: The DataFrame holding both columns.
        label_col: The label column.
        prediction_col: The prediction column.

    Returns:
        ``[label, prediction]`` column expressions.
    """
    cols = [label_col, prediction_col]
    types = [dataframe.schema[col].dataType for col in cols]
    result = [F.col(quoted(col)) for col in cols]
    if all(isinstance(data_type, NumericType) for data_type in types):
        return [F.expr(sql) for sql in widened(dataframe, *cols)]
    for this, other in ((0, 1), (1, 0)):
        this_type, other_type = types[this], types[other]
        if isinstance(this_type, StringType) and isinstance(other_type, (NumericType, BooleanType)):
            target = DoubleType() if isinstance(other_type, DecimalType) else other_type
            result[this] = _string_as(cols[this], target)
            result[other] = F.col(quoted(cols[other])).cast(target)
        elif isinstance(this_type, BooleanType) and isinstance(other_type, NumericType):
            # try_cast: a decimal with no integer digits cannot hold 1 (ANSI mode raises).
            result[this] = F.expr(f"try_cast({quoted(cols[this])} AS {other_type.simpleString()})")
    return result


def is_one(dataframe: DataFrame, col: str) -> Column:
    """Whether ``col`` is the positive class 1, as Spark 3's default ``col = 1`` reads it.

    A string column is read as an integer (``"1"``, ``" 1 "`` and ``"1.5"`` are 1;
    ``"1e0"`` and ``"yes"`` are not; :func:`smallaxe._sql.integral_from_string`), a
    boolean column is its own value (True is 1), and a number is compared with 1.

    Args:
        dataframe: The DataFrame holding ``col``.
        col: A column of ``dataframe``.

    Returns:
        A boolean column expression (null where ``col`` is null or does not parse).
    """
    data_type = dataframe.schema[col].dataType
    if isinstance(data_type, StringType):
        return integral_from_string(F.col(quoted(col)), IntegerType()) == 1
    if isinstance(data_type, BooleanType):
        return F.col(quoted(col))
    return F.col(quoted(col)) == 1
