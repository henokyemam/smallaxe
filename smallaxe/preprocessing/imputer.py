"""Imputer for handling missing values in PySpark DataFrames."""

from typing import Dict, List, Optional, Union

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DataType, StringType

from smallaxe import _artifact
from smallaxe.exceptions import (
    ColumnNotFoundError,
    ModelNotFittedError,
    ValidationError,
)
from smallaxe.preprocessing._common import column_ref, read_step_metadata, require_keys

VALID_NUMERICAL_STRATEGIES = {"mean", "median", "mode"}
VALID_CATEGORICAL_STRATEGIES = {"most_frequent"}

ARTIFACT_KIND = "imputer"
LEGACY_METADATA_FILE = "imputer_metadata.json"
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


def _fill_literal(fill: str, data_type: DataType) -> Column:
    """The categorical fill value as a literal of the column's own type."""
    if isinstance(data_type, StringType):
        return F.lit(fill)
    return F.lit(fill).cast(data_type)


class Imputer:
    """Impute missing values in numerical and categorical columns.

    Supports different strategies for numerical columns (mean, median, mode)
    and categorical columns (most_frequent). Custom values can also be specified.

    Categorical columns are filled in their own type: the most frequent value (or the
    custom ``categorical_strategy``) is kept as Spark's string form of the value and cast
    back to the column's type, so an integer or boolean categorical column is imputed
    too. ``fit`` raises ``ValidationError`` when the fill cannot be represented in the
    column's type (for example ``categorical_strategy='UNKNOWN'`` on an integer column).

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
        >>> imputer.fit(df, numerical_cols=['age', 'income'])
        >>> transformed_df = imputer.transform(df)

        >>> # Impute both numerical and categorical columns
        >>> imputer = Imputer(numerical_strategy='median', categorical_strategy='most_frequent')
        >>> imputer.fit(df, numerical_cols=['age', 'income'], categorical_cols=['category'])
        >>> transformed_df = imputer.transform(df)
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
                if numerical_strategy not in VALID_NUMERICAL_STRATEGIES:
                    raise ValidationError(
                        f"Invalid numerical_strategy '{numerical_strategy}'. "
                        f"Must be one of {sorted(VALID_NUMERICAL_STRATEGIES)} or a numeric value."
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
        df: DataFrame,
        numerical_cols: Optional[List[str]],
        categorical_cols: Optional[List[str]],
    ) -> None:
        """Validate that all specified columns exist in the DataFrame."""
        available_columns = df.columns
        all_cols = (numerical_cols or []) + (categorical_cols or [])

        for col in all_cols:
            if col not in available_columns:
                raise ColumnNotFoundError(column=col, available_columns=available_columns)

    def fit(
        self,
        df: DataFrame,
        numerical_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
    ) -> "Imputer":
        """Fit the imputer by computing fill values from the DataFrame.

        Args:
            df: PySpark DataFrame to fit on.
            numerical_cols: List of numerical column names to impute.
            categorical_cols: List of categorical column names to impute.

        Returns:
            self: The fitted Imputer instance.

        Raises:
            ValidationError: If no columns are given, a strategy is missing for the
                columns given, or a categorical fill does not fit its column's type.
            ColumnNotFoundError: If any specified column is not in the DataFrame.
        """
        numerical_cols = list(numerical_cols or [])
        categorical_cols = list(categorical_cols or [])

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

        self._validate_columns(df, numerical_cols, categorical_cols)

        self._numerical_cols = numerical_cols
        self._categorical_cols = categorical_cols
        self._numerical_fill_values = {}
        self._categorical_fill_values = {}
        self._is_fitted = False

        # Compute fill values for numerical columns
        if numerical_cols:
            self._compute_numerical_fill_values(df, numerical_cols)

        # Compute fill values for categorical columns
        if categorical_cols:
            self._compute_categorical_fill_values(df, categorical_cols)

        self._is_fitted = True
        return self

    def _compute_numerical_fill_values(self, df: DataFrame, numerical_cols: List[str]) -> None:
        """Compute fill values for numerical columns based on strategy."""
        if isinstance(self._numerical_strategy, (int, float)):
            # Custom value - use it for all columns
            for col in numerical_cols:
                self._numerical_fill_values[col] = float(self._numerical_strategy)
        elif self._numerical_strategy == "mean":
            # Compute mean for each column, ignoring NaN as well as null
            agg_exprs = [F.mean(_present(col)).alias(col) for col in numerical_cols]
            stats = df.select(agg_exprs).first()
            if stats:
                for col in numerical_cols:
                    value = stats[col]
                    self._numerical_fill_values[col] = float(value) if value is not None else 0.0
        elif self._numerical_strategy == "median":
            # Compute median for each column using approx_percentile
            for col in numerical_cols:
                median_value = df.select(F.percentile_approx(_present(col), 0.5)).first()[0]
                self._numerical_fill_values[col] = (
                    float(median_value) if median_value is not None else 0.0
                )
        elif self._numerical_strategy == "mode":
            # Compute mode for each column
            for col in numerical_cols:
                mode_row = (
                    df.filter(_present(col).isNotNull())
                    .groupBy(col)
                    .count()
                    .orderBy(F.col("count").desc())
                    .first()
                )
                if mode_row:
                    self._numerical_fill_values[col] = float(mode_row[col])
                else:
                    self._numerical_fill_values[col] = 0.0

    def _compute_categorical_fill_values(self, df: DataFrame, categorical_cols: List[str]) -> None:
        """Compute fill values for categorical columns based on strategy.

        A fill value is kept as Spark's string form of the value (``cast("string")``), so
        it round-trips through JSON and casts back to the column's type at transform.
        """
        types = {field.name: field.dataType for field in df.schema.fields}
        for col in categorical_cols:
            if self._categorical_strategy == "most_frequent":
                # Most frequent value; ties go to the smallest value, so the result does
                # not depend on partitioning.
                mode_row = (
                    df.select(column_ref(col).cast("string").alias("value"))
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
            self._check_categorical_fill(df, col, types[col], fill)
            self._categorical_fill_values[col] = fill

    @staticmethod
    def _check_categorical_fill(df: DataFrame, col: str, data_type: DataType, fill: str) -> None:
        """Raise ValidationError if ``fill`` cannot be stored in a ``data_type`` column."""
        if isinstance(data_type, StringType):
            return
        try:
            value = (
                df.sparkSession.range(1)
                .select(_fill_literal(fill, data_type).alias("fill"))
                .first()["fill"]
            )
        except Exception:  # an invalid cast fails under ANSI mode, or for complex types
            value = None
        if value is None:
            raise ValidationError(
                f"Cannot impute categorical column '{col}' of type "
                f"{data_type.simpleString()} with {fill!r}: the value is not a valid "
                f"{data_type.simpleString()}. Use a categorical_strategy of that type, or "
                "cast the column to string before imputing."
            )

    def transform(self, df: DataFrame) -> DataFrame:
        """Transform the DataFrame by imputing missing values.

        Args:
            df: PySpark DataFrame to transform.

        Returns:
            DataFrame with missing values imputed.

        Raises:
            ModelNotFittedError: If transform is called before fit.
            ColumnNotFoundError: If any fitted column is not in the DataFrame.
        """
        if not self._is_fitted:
            raise ModelNotFittedError("Imputer has not been fitted. Call fit() before transform().")

        # Validate that all fitted columns exist in the transform DataFrame
        all_cols = self._numerical_cols + self._categorical_cols
        available_columns = df.columns
        for col in all_cols:
            if col not in available_columns:
                raise ColumnNotFoundError(column=col, available_columns=available_columns)

        result_df = df

        # Fill numerical columns
        if self._numerical_fill_values:
            result_df = result_df.fillna(self._numerical_fill_values)

        # Fill categorical columns in their own type. DataFrame.fillna with a string value
        # silently skips non-string columns, so it is not used here.
        if self._categorical_fill_values:
            types = {field.name: field.dataType for field in result_df.schema.fields}
            result_df = result_df.withColumns(
                {
                    col: F.coalesce(column_ref(col), _fill_literal(fill, types[col]))
                    for col, fill in self._categorical_fill_values.items()
                }
            )

        return result_df

    def fit_transform(
        self,
        df: DataFrame,
        numerical_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
    ) -> DataFrame:
        """Fit and transform in a single step.

        Args:
            df: PySpark DataFrame to fit and transform.
            numerical_cols: List of numerical column names to impute.
            categorical_cols: List of categorical column names to impute.

        Returns:
            DataFrame with missing values imputed.
        """
        self.fit(df, numerical_cols, categorical_cols)
        return self.transform(df)

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
            ValidationError: If path is empty.
        """
        if not self._is_fitted:
            raise ModelNotFittedError("Imputer has not been fitted. Call fit() before saving.")

        _artifact.write_metadata(
            path,
            ARTIFACT_KIND,
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
        metadata = read_step_metadata(path, ARTIFACT_KIND, LEGACY_METADATA_FILE)
        require_keys(metadata, _PAYLOAD_KEYS, ARTIFACT_KIND, path)

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
                f"Cannot load {ARTIFACT_KIND}: corrupted fill values at {path!r} ({exc})."
            ) from exc
        imputer._is_fitted = True

        return imputer
