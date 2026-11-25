"""Argument checks shared by the public entry points: a DataFrame, a list of column names."""

from typing import Any, List, Optional, Sequence

from smallaxe.exceptions import ValidationError

__all__ = ["require_dataframe", "column_names"]


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
