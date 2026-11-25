"""Scaler for numerical feature scaling in PySpark DataFrames."""

from typing import Dict, List, Type, Union

from pyspark.ml.feature import (
    MaxAbsScaler as SparkMaxAbsScaler,
)
from pyspark.ml.feature import (
    MaxAbsScalerModel,
    MinMaxScalerModel,
    StandardScalerModel,
    VectorAssembler,
)
from pyspark.ml.feature import (
    MinMaxScaler as SparkMinMaxScaler,
)
from pyspark.ml.feature import (
    StandardScaler as SparkStandardScaler,
)
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame
from pyspark.sql.types import BooleanType, NumericType

from smallaxe import _artifact, _fs
from smallaxe.exceptions import (
    ColumnNotFoundError,
    ModelNotFittedError,
    ValidationError,
)
from smallaxe.preprocessing._common import (
    column_list,
    column_ref,
    read_step_metadata,
    require_keys,
    unique_column_name,
)

VALID_METHODS = {"standard", "minmax", "maxabs"}

ARTIFACT_KIND = "scaler"
LEGACY_METADATA_FILE = "scaler_metadata.json"
SPARK_MODEL_DIR = "spark_model"

_ScalerModel = Union[StandardScalerModel, MinMaxScalerModel, MaxAbsScalerModel]
_MODEL_CLASSES: Dict[str, Type[_ScalerModel]] = {
    "standard": StandardScalerModel,
    "minmax": MinMaxScalerModel,
    "maxabs": MaxAbsScalerModel,
}


class Scaler:
    """Scale numerical features using various scaling methods.

    Supports StandardScaler (zero mean, unit variance), MinMaxScaler (0-1 range),
    and MaxAbsScaler (scales by maximum absolute value).

    The scaled columns replace the input columns in place; every other column, including
    user columns named like the Scaler's internal working columns, is returned unchanged.

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

    def _validate_numeric(self, df: DataFrame, numerical_cols: List[str]) -> None:
        """Raise ValidationError naming the first input column that is not numeric or boolean."""
        types = {field.name: field.dataType for field in df.schema.fields}
        for col in numerical_cols:
            data_type = types[col]
            # Booleans scale as 0/1, as VectorAssembler and feature inference treat them.
            if not isinstance(data_type, (NumericType, BooleanType)):
                raise ValidationError(
                    f"Scaler can only scale numeric columns, but column '{col}' has type "
                    f"{data_type.simpleString()}. Encode or cast it before scaling."
                )

    def _create_scaler(
        self, input_col: str, output_col: str
    ) -> Union[SparkStandardScaler, SparkMinMaxScaler, SparkMaxAbsScaler]:
        """Create the appropriate PySpark scaler based on method."""
        if self._method == "standard":
            return SparkStandardScaler(
                inputCol=input_col,
                outputCol=output_col,
                withMean=True,
                withStd=True,
            )
        if self._method == "minmax":
            return SparkMinMaxScaler(
                inputCol=input_col,
                outputCol=output_col,
                min=0.0,
                max=1.0,
            )
        if self._method == "maxabs":
            return SparkMaxAbsScaler(
                inputCol=input_col,
                outputCol=output_col,
            )
        raise ValidationError(
            f"Invalid method '{self._method}'. Must be one of {sorted(VALID_METHODS)}."
        )

    @staticmethod
    def _working_names(columns: List[str]) -> Dict[str, str]:
        """Names for the assembled, scaled and array columns that no column in df uses."""
        taken = list(columns)
        names = {}
        for role, base in (
            ("assembled", "_assembled_features"),
            ("scaled", "_scaled_features"),
            ("array", "_scaled_array"),
        ):
            names[role] = unique_column_name(base, taken)
            taken.append(names[role])
        return names

    @staticmethod
    def _assemble(df: DataFrame, input_cols: List[str], output_col: str) -> DataFrame:
        return VectorAssembler(
            inputCols=input_cols,
            outputCol=output_col,
            handleInvalid="keep",
        ).transform(df)

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
            ValidationError: If numerical_cols is empty or a column is not numeric.
            ColumnNotFoundError: If any specified column is not in the DataFrame.
        """
        if not numerical_cols:
            raise ValidationError("numerical_cols cannot be empty for Scaler.")

        self._validate_columns(df, numerical_cols)
        self._validate_numeric(df, numerical_cols)
        cols = list(numerical_cols)

        names = self._working_names(df.columns)
        assembled_df = self._assemble(df, cols, names["assembled"])
        scaler = self._create_scaler(names["assembled"], names["scaled"])
        self._scaler_model = scaler.fit(assembled_df)
        self._numerical_cols = cols
        self._is_fitted = True

        return self

    def transform(self, df: DataFrame) -> DataFrame:
        """Transform the DataFrame by scaling numerical columns.

        Args:
            df: PySpark DataFrame to transform.

        Returns:
            DataFrame with scaled numerical columns, in the input's column order.

        Raises:
            ModelNotFittedError: If transform is called before fit.
            ColumnNotFoundError: If any fitted column is not in the DataFrame.
        """
        if not self._is_fitted:
            raise ModelNotFittedError("Scaler has not been fitted. Call fit() before transform().")

        # Validate that all fitted columns exist
        self._validate_columns(df, self._numerical_cols)

        names = self._working_names(df.columns)
        assembled_df = self._assemble(df, self._numerical_cols, names["assembled"])

        # The saved model carries the working names of its fit (artifacts from 0.10 and
        # earlier: "_assembled_features"); point it at this call's collision-free names.
        model = self._scaler_model
        scaled_df = model.transform(
            assembled_df,
            {model.inputCol: names["assembled"], model.outputCol: names["scaled"]},
        )
        scaled_df = scaled_df.withColumn(
            names["array"], vector_to_array(column_ref(names["scaled"]))
        )

        positions = {col: idx for idx, col in enumerate(self._numerical_cols)}
        array = column_ref(names["array"])
        return scaled_df.select(
            *[
                (
                    array[positions[col]].cast("double").alias(col)
                    if col in positions
                    else column_ref(col)
                )
                for col in df.columns
            ]
        )

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

    def save(self, path: str) -> None:
        """Save the scaler as a directory: ``metadata.json`` plus ``spark_model/``.

        IO goes through the Hadoop FileSystem layer, so the metadata lands on the same
        filesystem as the Spark model (local, ``dbfs:/``, ``s3://``, ``hdfs://``).

        Args:
            path: The directory path to save the scaler to.

        Raises:
            ModelNotFittedError: If save is called before fit.
            ValidationError: If path is empty.
        """
        if not self._is_fitted:
            raise ModelNotFittedError("Scaler has not been fitted. Call fit() before saving.")
        if not path:
            raise ValidationError("Path cannot be empty.")

        # The Spark model first: metadata.json is written last, so a directory that has
        # it holds a complete artifact.
        self._scaler_model.write().overwrite().save(_fs.join(path, SPARK_MODEL_DIR))
        _artifact.write_metadata(
            path,
            ARTIFACT_KIND,
            {"method": self._method, "numerical_cols": list(self._numerical_cols)},
        )

    @classmethod
    def load(cls, path: str) -> "Scaler":
        """Load a scaler saved by :meth:`save`, or by smallaxe 0.10 and earlier.

        Args:
            path: The directory path to load the scaler from.

        Returns:
            The loaded, fitted Scaler.

        Raises:
            ValidationError: If the path does not exist, is not a scaler artifact, was
                written by a newer smallaxe, or its metadata or Spark model is missing or
                unreadable.
        """
        metadata = read_step_metadata(path, ARTIFACT_KIND, LEGACY_METADATA_FILE)
        require_keys(metadata, ["method", "numerical_cols"], ARTIFACT_KIND, path)

        scaler = cls(method=metadata["method"])
        numerical_cols = column_list(metadata, "numerical_cols", ARTIFACT_KIND, path)
        model_path = _fs.join(path, SPARK_MODEL_DIR)
        if not _fs.exists(model_path):
            raise ValidationError(
                f"Cannot load {ARTIFACT_KIND}: {path!r} has no {SPARK_MODEL_DIR}/ directory."
            )
        model_class = _MODEL_CLASSES[scaler._method]
        try:
            scaler._scaler_model = model_class.load(model_path)
        except _fs.jvm_error_types() as exc:
            raise ValidationError(
                f"Cannot load {ARTIFACT_KIND}: the Spark model at {model_path!r} could not "
                f"be read as a {model_class.__name__} ({_artifact.jvm_error_summary(exc)})."
            ) from exc
        scaler._numerical_cols = numerical_cols
        scaler._is_fitted = True

        return scaler
