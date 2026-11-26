"""The Spark side every DataFrame chart shares: which rows count, and collecting a summary.

A chart aggregates in Spark and collects only the aggregated rows. :func:`collect` is
the one place a chart's aggregate reaches the driver; its callers hand it a grouped or
globally aggregated DataFrame, never row-level data.
"""

from typing import List, Optional

from pyspark.sql import Column, DataFrame, Row
from pyspark.sql import functions as F

from smallaxe import _checks

__all__ = ["present", "collect", "summary"]


def present(column: Column) -> Column:
    """True where ``column`` holds a number: not null and, as a double, not NaN.

    Args:
        column: A numeric column expression.

    Returns:
        A boolean column expression.
    """
    as_double = column.cast("double")
    return as_double.isNotNull() & ~F.isnan(as_double)


def collect(table: DataFrame, context: str, limit: Optional[int] = None) -> List[Row]:
    """The rows of an aggregated ``table``, at most ``limit`` of them.

    Spark's own argument errors become ``ValidationError`` (see
    :func:`smallaxe._checks.spark_argument_errors`).

    Args:
        table: A grouped or globally aggregated DataFrame: its row count is bounded by
            the chart's bins or classes, not by the input's rows.
        context: What the chart was doing, for an error message.
        limit: Collect at most this many rows; None collects them all.

    Returns:
        The collected rows.
    """
    with _checks.spark_argument_errors(context):
        if limit is not None:
            table = table.limit(limit)
        return table.collect()


def summary(table: DataFrame, context: str) -> Row:
    """The single row of a global aggregate (``select`` of aggregate expressions).

    Args:
        table: A global aggregate, which always has exactly one row.
        context: What the chart was doing, for an error message.

    Returns:
        Its row.
    """
    with _checks.spark_argument_errors(context):
        return _checks.one_row(table)
