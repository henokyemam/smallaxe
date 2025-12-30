"""Imputer for handling missing values in PySpark DataFrames."""

from typing import Dict, List, Optional, Union

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from smallaxe.exceptions import (
    ColumnNotFoundError,
    ModelNotFittedError,
    ValidationError,
)

VALID_NUMERICAL_STRATEGIES = {"mean", "median", "mode"}
VALID_CATEGORICAL_STRATEGIES = {"mode"}


class Imputer:
    """Impute missing values in numerical and categorical columns.

    Supports different strategies for numerical columns (mean, median, mode)
    and categorical columns (mode). Custom values can also be specified.

    Args:
        numerical_strategy: Strategy for imputing numerical columns.
            One of 'mean', 'median', 'mode', or a custom numeric value.
            Default is 'mean'.
        categorical_strategy: Strategy for imputing categorical columns.
            One of 'mode' or a custom string value. Default is 'mode'.

    Example:
        >>> imputer = Imputer(numerical_strategy='median', categorical_strategy='mode')
        >>> imputer.fit(df, numerical_cols=['age', 'income'], categorical_cols=['category'])
        >>> transformed_df = imputer.transform(df)
    """

    def __init__(
        self,
        numerical_strategy: Union[str, int, float] = "mean",
        categorical_strategy: Union[str] = "mode",
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
        numerical_strategy: Union[str, int, float],
        categorical_strategy: Union[str],
    ) -> None:
        """Validate the imputation strategies."""
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

        if isinstance(categorical_strategy, str):
            if categorical_strategy not in VALID_CATEGORICAL_STRATEGIES:
                # It's a custom string value, which is allowed
                pass
        else:
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
            ColumnNotFoundError: If any specified column is not in the DataFrame.
        """
        numerical_cols = numerical_cols or []
        categorical_cols = categorical_cols or []

        self._validate_columns(df, numerical_cols, categorical_cols)

        self._numerical_cols = numerical_cols
        self._categorical_cols = categorical_cols
        self._numerical_fill_values = {}
        self._categorical_fill_values = {}

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
            # Compute mean for each column
            agg_exprs = [F.mean(F.col(col)).alias(col) for col in numerical_cols]
            stats = df.select(agg_exprs).first()
            if stats:
                for col in numerical_cols:
                    value = stats[col]
                    self._numerical_fill_values[col] = float(value) if value is not None else 0.0
        elif self._numerical_strategy == "median":
            # Compute median for each column using approx_percentile
            for col in numerical_cols:
                median_value = df.select(F.expr(f"percentile_approx({col}, 0.5)")).first()[0]
                self._numerical_fill_values[col] = (
                    float(median_value) if median_value is not None else 0.0
                )
        elif self._numerical_strategy == "mode":
            # Compute mode for each column
            for col in numerical_cols:
                mode_row = (
                    df.filter(F.col(col).isNotNull())
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
        """Compute fill values for categorical columns based on strategy."""
        if self._categorical_strategy == "mode":
            # Compute mode for each column
            for col in categorical_cols:
                mode_row = (
                    df.filter(F.col(col).isNotNull())
                    .groupBy(col)
                    .count()
                    .orderBy(F.col("count").desc())
                    .first()
                )
                if mode_row:
                    self._categorical_fill_values[col] = str(mode_row[col])
                else:
                    self._categorical_fill_values[col] = "UNKNOWN"
        else:
            # Custom value - use it for all columns
            for col in categorical_cols:
                self._categorical_fill_values[col] = str(self._categorical_strategy)

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

        # Fill categorical columns
        if self._categorical_fill_values:
            result_df = result_df.fillna(self._categorical_fill_values)

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
