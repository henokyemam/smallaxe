"""Imputer for handling missing values in PySpark DataFrames."""

import re
from typing import Dict, List, Optional, Union

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DataType,
    DecimalType,
    DoubleType,
    FloatType,
    IntegralType,
    StringType,
)

from smallaxe import _artifact, _checks, _fs
from smallaxe.exceptions import (
    ColumnNotFoundError,
    ModelNotFittedError,
    ValidationError,
)
from smallaxe.preprocessing._common import column_ref, read_step_metadata, require_keys

__all__ = ["Imputer"]

_VALID_NUMERICAL_STRATEGIES = {"mean", "median", "mode"}
_VALID_CATEGORICAL_STRATEGIES = {"most_frequent"}

_ARTIFACT_KIND = "imputer"
_LEGACY_METADATA_FILE = "imputer_metadata.json"
_PAYLOAD_KEYS = [
    "numerical_strategy",
    "categorical_strategy",
    "numerical_cols",
    "categorical_cols",
    "numerical_fill_values",
    "categorical_fill_values",
]


def _present(col: str) -> Column:
    """``col`` with NaN turned into null, so aggregates skip both."""
    return F.when(F.isnan(F.col(col)), None).otherwise(F.col(col))


# A fill for an exact numeric column must keep its value when cast to the column's type;
# comparing both as this decimal catches truncation ("1.5" -> 1) and rounding.
_EXACT = DecimalType(38, 18)


# The strings Spark's non-ANSI string-to-integer cast reads: an optional sign, digits and
# an optional fraction, after trimming the characters up to the space (no exponent, ASCII
# digits only). ANSI mode reads only the integers among them ("7", not "7.0"), so the
# fill goes through ``_EXACT`` instead, which reads them all in either mode.
_INTEGRAL_FILL = re.compile(r"[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)")
_TRIMMED = "".join(chr(code) for code in range(33))


def _integral_fill(fill: str) -> bool:
    """Whether ``fill`` is spelled the way Spark reads a string as an integer."""
    return _INTEGRAL_FILL.fullmatch(fill.strip(_TRIMMED)) is not None


def _fill_literal(fill: str, data_type: DataType) -> Column:
    """The categorical fill value as a literal of the column's own type.

    ``fit`` checked that the cast succeeds and keeps the value (``_check_categorical_fill``),
    so it raises in neither ANSI mode. An integer column's fill is read through
    ``_EXACT``: ANSI mode's string-to-integer cast rejects ``"7.0"``, which the non-ANSI
    one reads as 7; the decimal reads it as 7 in both.
    """
    if isinstance(data_type, StringType):
        return F.lit(fill)
    if isinstance(data_type, IntegralType):
        return F.lit(fill).cast(_EXACT).cast(data_type)
    return F.lit(fill).cast(data_type)


def _categorical_value(col: str, data_type: DataType) -> Column:
    """``col``, with NaN made null for a floating column (NaN counts as missing)."""
    if isinstance(data_type, (FloatType, DoubleType)):
        return F.when(F.isnan(column_ref(col)), None).otherwise(column_ref(col))
    return column_ref(col)


class Imputer:
    """Impute missing values in numerical and categorical columns.

    Supports different strategies for numerical columns (mean, median, mode)
    and categorical columns (most_frequent). Custom values can also be specified.

    Categorical columns are filled in their own type: the most frequent value (or the
    custom ``categorical_strategy``) is kept as Spark's string form of the value and cast
    back to the column's type, so an integer or boolean categorical column is imputed
    too. ``fit`` raises ``ValidationError`` when the fill cannot be represented exactly in
    the column's type (for example ``categorical_strategy='UNKNOWN'`` or ``'1.5'`` on an
    integer column). In a float or double column NaN counts as missing, both for
    ``most_frequent`` and at transform.

    Args:
        numerical_strategy: Strategy for imputing numerical columns.
            One of 'mean', 'median', 'mode', or a custom numeric value.
            Default is None (must be specified if numerical_cols are used).
        categorical_strategy: Strategy for imputing categorical columns.
            One of 'most_frequent' or a custom string value.
            Default is None (must be specified if categorical_cols are used).

    Example:
        >>> # Impute only numerical columns
        >>> imputer = Imputer(numerical_strategy='median')
        >>> imputer.fit(dataframe, numerical_cols=['age', 'income'])
        >>> transformed_df = imputer.transform(dataframe)

        >>> # Impute both numerical and categorical columns
        >>> imputer = Imputer(numerical_strategy='median', categorical_strategy='most_frequent')
        >>> imputer.fit(dataframe, numerical_cols=['age', 'income'], categorical_cols=['category'])
        >>> transformed_df = imputer.transform(dataframe)
    """

    def __init__(
        self,
        numerical_strategy: Optional[Union[str, int, float]] = None,
        categorical_strategy: Optional[str] = None,
    ):
        self._validate_strategies(numerical_strategy, categorical_strategy)
        self._numerical_strategy = numerical_strategy
        self._categorical_strategy = categorical_strategy
        self._numerical_cols: List[str] = []
        self._categorical_cols: List[str] = []
        self._numerical_fill_values: Dict[str, float] = {}
        self._categorical_fill_values: Dict[str, str] = {}
        self._is_fitted = False

    def _validate_strategies(
        self,
        numerical_strategy: Optional[Union[str, int, float]],
        categorical_strategy: Optional[str],
    ) -> None:
        """Validate the imputation strategies."""
        if numerical_strategy is not None:
            if isinstance(numerical_strategy, str):
                if numerical_strategy not in _VALID_NUMERICAL_STRATEGIES:
                    raise ValidationError(
                        f"Invalid numerical_strategy '{numerical_strategy}'. "
                        f"Must be one of {sorted(_VALID_NUMERICAL_STRATEGIES)} or a numeric value."
                    )
            elif not isinstance(numerical_strategy, (int, float)):
                raise ValidationError(
                    f"numerical_strategy must be a string strategy or numeric value, "
                    f"got {type(numerical_strategy).__name__}."
                )

        if categorical_strategy is not None:
            if not isinstance(categorical_strategy, str):
                raise ValidationError(
                    f"categorical_strategy must be a string, got {type(categorical_strategy).__name__}."
                )

    def _validate_columns(
        self,
        dataframe: DataFrame,
        numerical_cols: Optional[List[str]],
        categorical_cols: Optional[List[str]],
    ) -> None:
        """Validate that all specified columns exist in the DataFrame."""
        available_columns = dataframe.columns
        all_cols = (numerical_cols or []) + (categorical_cols or [])

        for col in all_cols:
            if col not in available_columns:
                raise ColumnNotFoundError(column=col, available_columns=available_columns)

    def fit(
        self,
        dataframe: DataFrame,
        numerical_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
    ) -> "Imputer":
        """Fit the imputer by computing fill values from the DataFrame.

        Args:
            dataframe: PySpark DataFrame to fit on.
            numerical_cols: List of numerical column names to impute.
            categorical_cols: List of categorical column names to impute.

        Returns:
            self: The fitted Imputer instance.

        Raises:
            ValidationError: If ``dataframe`` is not a DataFrame, no columns are given or a
                column list is not a list of names, a strategy is missing for the columns
                given, or a categorical fill does not fit its column's type.
            ColumnNotFoundError: If any specified column is not in the DataFrame.
        """
        _checks.require_dataframe(dataframe)
        numerical_cols = _checks.column_names("numerical_cols", numerical_cols) or []
        categorical_cols = _checks.column_names("categorical_cols", categorical_cols) or []

        if not numerical_cols and not categorical_cols:
            raise ValidationError(
                "numerical_cols and categorical_cols cannot both be empty for Imputer."
            )
        if numerical_cols and self._numerical_strategy is None:
            raise ValidationError(
                "numerical_cols provided but numerical_strategy is None. "
                "Specify a numerical_strategy in the constructor."
            )
        if categorical_cols and self._categorical_strategy is None:
            raise ValidationError(
                "categorical_cols provided but categorical_strategy is None. "
                "Specify a categorical_strategy in the constructor."
            )

        self._validate_columns(dataframe, numerical_cols, categorical_cols)

        self._numerical_cols = numerical_cols
        self._categorical_cols = categorical_cols
        self._numerical_fill_values = {}
        self._categorical_fill_values = {}
        self._is_fitted = False

        # Compute fill values for numerical columns
        if numerical_cols:
            self._compute_numerical_fill_values(dataframe, numerical_cols)

        # Compute fill values for categorical columns
        if categorical_cols:
            self._compute_categorical_fill_values(dataframe, categorical_cols)

        self._is_fitted = True
        return self

    def _compute_numerical_fill_values(
        self, dataframe: DataFrame, numerical_cols: List[str]
    ) -> None:
        """Compute fill values for numerical columns based on strategy."""
        if isinstance(self._numerical_strategy, (int, float)):
            # Custom value - use it for all columns
            for col in numerical_cols:
                self._numerical_fill_values[col] = float(self._numerical_strategy)
        elif self._numerical_strategy == "mean":
            # Compute mean for each column, ignoring NaN as well as null
            agg_exprs = [F.mean(_present(col)).alias(col) for col in numerical_cols]
            stats = dataframe.select(agg_exprs).first()
            if stats:
                for col in numerical_cols:
                    value = stats[col]
                    self._numerical_fill_values[col] = float(value) if value is not None else 0.0
        elif self._numerical_strategy == "median":
            # Compute median for each column using approx_percentile
            for col in numerical_cols:
                median_value = _checks.one_row(
                    dataframe.select(F.percentile_approx(_present(col), 0.5))
                )[0]
                self._numerical_fill_values[col] = (
                    float(median_value) if median_value is not None else 0.0
                )
        elif self._numerical_strategy == "mode":
            # Compute mode for each column
            for col in numerical_cols:
                mode_row = (
                    dataframe.filter(_present(col).isNotNull())
                    .groupBy(col)
                    .count()
                    .orderBy(F.col("count").desc())
                    .first()
                )
                if mode_row:
                    self._numerical_fill_values[col] = float(mode_row[col])
                else:
                    self._numerical_fill_values[col] = 0.0

    def _compute_categorical_fill_values(
        self, dataframe: DataFrame, categorical_cols: List[str]
    ) -> None:
        """Compute fill values for categorical columns based on strategy.

        A fill value is kept as Spark's string form of the value (``cast("string")``), so
        it round-trips through JSON and casts back to the column's type at transform.
        """
        types = {field.name: field.dataType for field in dataframe.schema.fields}
        for col in categorical_cols:
            if self._categorical_strategy == "most_frequent":
                # Most frequent value; ties go to the smallest value, so the result does
                # not depend on partitioning.
                mode_row = (
                    dataframe.select(
                        _categorical_value(col, types[col]).cast("string").alias("value")
                    )
                    .filter(F.col("value").isNotNull())
                    .groupBy("value")
                    .count()
                    .orderBy(F.col("count").desc(), F.col("value").asc())
                    .first()
                )
                fill = mode_row["value"] if mode_row else "UNKNOWN"
            else:
                # Custom value - use it for all columns
                fill = str(self._categorical_strategy)
            self._check_categorical_fill(dataframe, col, types[col], fill)
            self._categorical_fill_values[col] = fill

    @staticmethod
    def _check_categorical_fill(
        dataframe: DataFrame, col: str, data_type: DataType, fill: str
    ) -> None:
        """Raise ValidationError if ``fill`` cannot be stored exactly in a ``data_type`` column.

        The cast must succeed (not null) and, for an integer or decimal column, keep the
        value: ``"1.5"`` on an integer column would silently fill 1.
        """
        if isinstance(data_type, StringType):
            return
        # try_cast (a SQL function from Spark 3.2) is null for a value that does not fit,
        # in and out of ANSI mode, where a plain cast would raise with ANSI mode on.
        target = data_type.simpleString()
        exact_sql = f"try_cast(fill AS {_EXACT.simpleString()})"
        if isinstance(data_type, IntegralType):
            typed_sql = f"try_cast({exact_sql} AS {target})"
        else:
            typed_sql = f"try_cast(fill AS {target})"
        exact = isinstance(data_type, (IntegralType, DecimalType))
        if isinstance(data_type, IntegralType) and not _integral_fill(fill):
            valid = False
        else:
            try:
                row = _checks.one_row(
                    dataframe.sparkSession.range(1)
                    .select(F.lit(fill).alias("fill"))
                    .selectExpr(
                        f"{typed_sql} IS NOT NULL AS cast_ok",
                        (
                            f"try_cast({typed_sql} AS {_EXACT.simpleString()}) <=> {exact_sql}"
                            if exact
                            else "true"
                        )
                        + " AS exact",
                    )
                )
                valid = bool(row["cast_ok"]) and bool(row["exact"])
            except _fs.jvm_error_types():  # a cast to a complex type fails at analysis
                valid = False
        if not valid:
            raise ValidationError(
                f"Cannot impute categorical column '{col}' of type "
                f"{data_type.simpleString()} with {fill!r}: the value is not exactly "
                f"representable as {data_type.simpleString()}. Use a categorical_strategy "
                "of that type, or cast the column to string before imputing."
            )

    def transform(self, dataframe: DataFrame) -> DataFrame:
        """Transform the DataFrame by imputing missing values.

        Args:
            dataframe: PySpark DataFrame to transform.

        Returns:
            DataFrame with missing values imputed.

        Raises:
            ModelNotFittedError: If transform is called before fit.
            ValidationError: If ``dataframe`` is not a DataFrame.
            ColumnNotFoundError: If any fitted column is not in the DataFrame.
        """
        if not self._is_fitted:
            raise ModelNotFittedError("Imputer has not been fitted. Call fit() before transform().")
        _checks.require_dataframe(dataframe)

        # Validate that all fitted columns exist in the transform DataFrame
        all_cols = self._numerical_cols + self._categorical_cols
        available_columns = dataframe.columns
        for col in all_cols:
            if col not in available_columns:
                raise ColumnNotFoundError(column=col, available_columns=available_columns)

        result_df = dataframe

        # Fill numerical columns
        if self._numerical_fill_values:
            # fillna's dict parameter is typed with every literal type it accepts.
            numerical_fills: Dict[str, Union[bool, float, int, str]] = dict(
                self._numerical_fill_values
            )
            result_df = result_df.fillna(numerical_fills)

        # Fill categorical columns in their own type (fit checked the fill fits it). For a
        # floating column NaN counts as missing, as DataFrame.fillna treats it.
        if self._categorical_fill_values:
            types = {field.name: field.dataType for field in result_df.schema.fields}
            result_df = result_df.withColumns(
                {
                    col: F.coalesce(
                        _categorical_value(col, types[col]), _fill_literal(fill, types[col])
                    )
                    for col, fill in self._categorical_fill_values.items()
                }
            )

        return result_df

    def fit_transform(
        self,
        dataframe: DataFrame,
        numerical_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
    ) -> DataFrame:
        """Fit and transform in a single step.

        Args:
            dataframe: PySpark DataFrame to fit and transform.
            numerical_cols: List of numerical column names to impute.
            categorical_cols: List of categorical column names to impute.

        Returns:
            DataFrame with missing values imputed.

        Raises:
            ValidationError: As :meth:`fit`.
            ColumnNotFoundError: As :meth:`fit`.
        """
        self.fit(dataframe, numerical_cols, categorical_cols)
        return self.transform(dataframe)

    @property
    def numerical_fill_values(self) -> Dict[str, float]:
        """Get the computed fill values for numerical columns.

        Returns:
            Dictionary mapping column names to their fill values.

        Raises:
            ModelNotFittedError: If accessed before fit.
        """
        if not self._is_fitted:
            raise ModelNotFittedError(
                "Imputer has not been fitted. Call fit() to compute fill values."
            )
        return self._numerical_fill_values.copy()

    @property
    def categorical_fill_values(self) -> Dict[str, str]:
        """Get the computed fill values for categorical columns.

        Values are Spark's string form of the fill (``"3"`` for an integer column); they
        are cast to the column's type at transform.

        Returns:
            Dictionary mapping column names to their fill values.

        Raises:
            ModelNotFittedError: If accessed before fit.
        """
        if not self._is_fitted:
            raise ModelNotFittedError(
                "Imputer has not been fitted. Call fit() to compute fill values."
            )
        return self._categorical_fill_values.copy()

    def save(self, path: str) -> None:
        """Save the imputer as a directory holding ``metadata.json``.

        IO goes through the Hadoop FileSystem layer, so local paths, ``dbfs:/``,
        ``s3://`` and ``hdfs://`` all work.

        Args:
            path: The directory path to save the imputer to.

        Raises:
            ModelNotFittedError: If save is called before fit.
            ValidationError: If path is empty or not a string.
        """
        if not self._is_fitted:
            raise ModelNotFittedError("Imputer has not been fitted. Call fit() before saving.")

        _artifact.write_metadata(
            path,
            _ARTIFACT_KIND,
            {
                "numerical_strategy": self._numerical_strategy,
                "categorical_strategy": self._categorical_strategy,
                "numerical_cols": list(self._numerical_cols),
                "categorical_cols": list(self._categorical_cols),
                "numerical_fill_values": dict(self._numerical_fill_values),
                "categorical_fill_values": dict(self._categorical_fill_values),
            },
        )

    @classmethod
    def load(cls, path: str) -> "Imputer":
        """Load an imputer saved by :meth:`save`, or by smallaxe 0.10 and earlier.

        Args:
            path: The directory path to load the imputer from.

        Returns:
            The loaded, fitted Imputer instance.

        Raises:
            ValidationError: If the path does not exist, is not an imputer artifact, was
                written by a newer smallaxe, or its metadata is incomplete.
        """
        metadata = read_step_metadata(path, _ARTIFACT_KIND, _LEGACY_METADATA_FILE)
        require_keys(metadata, _PAYLOAD_KEYS, _ARTIFACT_KIND, path)

        imputer = cls(
            numerical_strategy=metadata["numerical_strategy"],
            categorical_strategy=metadata["categorical_strategy"],
        )
        try:
            imputer._numerical_cols = [str(col) for col in metadata["numerical_cols"]]
            imputer._categorical_cols = [str(col) for col in metadata["categorical_cols"]]
            imputer._numerical_fill_values = {
                str(col): float(value) for col, value in metadata["numerical_fill_values"].items()
            }
            imputer._categorical_fill_values = {
                str(col): str(value) for col, value in metadata["categorical_fill_values"].items()
            }
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValidationError(
                f"Cannot load {_ARTIFACT_KIND}: corrupted fill values at {path!r} ({exc})."
            ) from exc
        imputer._is_fitted = True

        return imputer
