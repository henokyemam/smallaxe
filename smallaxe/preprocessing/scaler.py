"""Scaler for numerical feature scaling in PySpark DataFrames."""

from typing import List

from pyspark.ml.feature import (
    MaxAbsScaler as SparkMaxAbsScaler,
)
from pyspark.ml.feature import (
    MinMaxScaler as SparkMinMaxScaler,
)
from pyspark.ml.feature import (
    StandardScaler as SparkStandardScaler,
)
from pyspark.ml.feature import (
    VectorAssembler,
)
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

from smallaxe.exceptions import (
    ColumnNotFoundError,
    ModelNotFittedError,
    ValidationError,
)

VALID_METHODS = {"standard", "minmax", "maxabs"}


class Scaler:
    """Scale numerical features using various scaling methods.

    Supports StandardScaler (zero mean, unit variance), MinMaxScaler (0-1 range),
    and MaxAbsScaler (scales by maximum absolute value).

    Args:
        method: Scaling method. One of 'standard', 'minmax', 'maxabs'.
            Default is 'standard'.

    Example:
        >>> scaler = Scaler(method='standard')
        >>> scaler.fit(df, numerical_cols=['age', 'income'])
        >>> scaled_df = scaler.transform(df)
    """

    def __init__(self, method: str = "standard"):
        self._validate_method(method)
        self._method = method
        self._numerical_cols: List[str] = []
        self._scaler_model = None
        self._is_fitted = False

    def _validate_method(self, method: str) -> None:
        """Validate the scaling method."""
        if method not in VALID_METHODS:
            raise ValidationError(
                f"Invalid method '{method}'. Must be one of {sorted(VALID_METHODS)}."
            )

    def _validate_columns(
        self,
        df: DataFrame,
        numerical_cols: List[str],
    ) -> None:
        """Validate that all specified columns exist in the DataFrame."""
        available_columns = df.columns
        for col in numerical_cols:
            if col not in available_columns:
                raise ColumnNotFoundError(column=col, available_columns=available_columns)

    def _create_scaler(self):
        """Create the appropriate PySpark scaler based on method."""
        if self._method == "standard":
            return SparkStandardScaler(
                inputCol="_assembled_features",
                outputCol="_scaled_features",
                withMean=True,
                withStd=True,
            )
        elif self._method == "minmax":
            return SparkMinMaxScaler(
                inputCol="_assembled_features",
                outputCol="_scaled_features",
                min=0.0,
                max=1.0,
            )
        elif self._method == "maxabs":
            return SparkMaxAbsScaler(
                inputCol="_assembled_features",
                outputCol="_scaled_features",
            )

    def fit(
        self,
        df: DataFrame,
        numerical_cols: List[str],
    ) -> "Scaler":
        """Fit the scaler by computing scaling parameters from the DataFrame.

        Args:
            df: PySpark DataFrame to fit on.
            numerical_cols: List of numerical column names to scale.

        Returns:
            self: The fitted Scaler instance.

        Raises:
            ValidationError: If numerical_cols is empty.
            ColumnNotFoundError: If any specified column is not in the DataFrame.
        """
        if not numerical_cols:
            raise ValidationError("numerical_cols cannot be empty for Scaler.")

        self._validate_columns(df, numerical_cols)
        self._numerical_cols = numerical_cols

        # Assemble features into a vector
        assembler = VectorAssembler(
            inputCols=numerical_cols,
            outputCol="_assembled_features",
            handleInvalid="keep",
        )
        assembled_df = assembler.transform(df)

        # Create and fit the scaler
        scaler = self._create_scaler()
        self._scaler_model = scaler.fit(assembled_df)
        self._is_fitted = True

        return self

    def transform(self, df: DataFrame) -> DataFrame:
        """Transform the DataFrame by scaling numerical columns.

        Args:
            df: PySpark DataFrame to transform.

        Returns:
            DataFrame with scaled numerical columns.

        Raises:
            ModelNotFittedError: If transform is called before fit.
            ColumnNotFoundError: If any fitted column is not in the DataFrame.
        """
        if not self._is_fitted:
            raise ModelNotFittedError("Scaler has not been fitted. Call fit() before transform().")

        # Validate that all fitted columns exist
        self._validate_columns(df, self._numerical_cols)

        # Assemble features into a vector
        assembler = VectorAssembler(
            inputCols=self._numerical_cols,
            outputCol="_assembled_features",
            handleInvalid="keep",
        )
        assembled_df = assembler.transform(df)

        # Apply the scaler
        scaled_df = self._scaler_model.transform(assembled_df)

        # Extract scaled values back to individual columns
        result_df = self._extract_scaled_columns(scaled_df)

        return result_df

    def _extract_scaled_columns(self, df: DataFrame) -> DataFrame:
        """Extract scaled values from vector column back to individual columns."""
        # Create UDFs to extract each element from the scaled vector
        result_df = df

        for idx, col_name in enumerate(self._numerical_cols):
            # Create a UDF to extract the idx-th element from the vector
            extract_element = F.udf(
                lambda v, i=idx: float(v[i]) if v is not None else None,
                DoubleType(),
            )
            result_df = result_df.withColumn(col_name, extract_element(F.col("_scaled_features")))

        # Drop temporary columns
        result_df = result_df.drop("_assembled_features", "_scaled_features")

        return result_df

    def fit_transform(
        self,
        df: DataFrame,
        numerical_cols: List[str],
    ) -> DataFrame:
        """Fit and transform in a single step.

        Args:
            df: PySpark DataFrame to fit and transform.
            numerical_cols: List of numerical column names to scale.

        Returns:
            DataFrame with scaled numerical columns.
        """
        self.fit(df, numerical_cols)
        return self.transform(df)

    @property
    def method(self) -> str:
        """Get the scaling method.

        Returns:
            The scaling method ('standard', 'minmax', or 'maxabs').
        """
        return self._method

    @property
    def numerical_cols(self) -> List[str]:
        """Get the fitted numerical columns.

        Returns:
            List of column names that were fitted.

        Raises:
            ModelNotFittedError: If accessed before fit.
        """
        if not self._is_fitted:
            raise ModelNotFittedError(
                "Scaler has not been fitted. Call fit() to access numerical_cols."
            )
        return self._numerical_cols.copy()
