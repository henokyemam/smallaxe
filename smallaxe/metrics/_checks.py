"""Input checks shared by the metric functions."""

from typing import List

from pyspark.sql import DataFrame

from smallaxe.exceptions import ColumnNotFoundError, ValidationError


def validate_columns(df: DataFrame, *cols: str) -> None:
    """Raise ``ColumnNotFoundError`` if any of ``cols`` is not a column of ``df``."""
    available_columns: List[str] = df.columns
    for col in cols:
        if col not in available_columns:
            raise ColumnNotFoundError(column=col, available_columns=available_columns)


def require_rows(df: DataFrame, metric: str) -> None:
    """Raise ``ValidationError`` naming ``metric`` if ``df`` has no rows.

    A metric of nothing is not a score: an empty frame usually means an empty
    validation split or a filter that removed everything.
    """
    if df.limit(1).count() == 0:
        raise ValidationError(f"Cannot compute {metric}: the DataFrame has no rows.")
