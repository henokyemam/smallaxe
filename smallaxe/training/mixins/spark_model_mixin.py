"""SparkModelMixin for Spark MLlib model operations."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from pyspark.ml.feature import VectorAssembler
from pyspark.sql import DataFrame

from smallaxe.exceptions import ColumnNotFoundError, ModelNotFittedError, ValidationError

__all__ = ["SparkModelMixin"]


class SparkModelMixin(ABC):
    """Mixin providing Spark MLlib model operations.

    This mixin provides feature assembly, prediction, feature importances, and saving
    and loading the fitted Spark model. A subclass fits the estimator
    (:meth:`_fit_spark_model`) and reads its importances
    (:meth:`_get_feature_importance`).

    The assembled feature vector and the estimator's output columns live under
    internal names while a Spark model runs, and user columns that share a name
    with one of them are set aside and restored, so ``predict`` returns every
    input column unchanged plus the requested output column.
    """

    # The assembled vector. Named so it cannot collide with a user column.
    FEATURES_COL = "__smallaxe_features__"
    # Estimator output columns. Spark ML fixes these defaults; a user column with
    # the same name is shielded for the duration of a transform.
    PREDICTION_COL = "prediction"
    PROBABILITY_COL = "probability"
    RAW_PREDICTION_COL = "rawPrediction"

    def _handle_invalid(self) -> str:
        """``VectorAssembler``'s policy for null and NaN features.

        ``"error"`` by default: a model that cannot treat NaN as missing must not
        silently lose rows. Models whose estimator accepts missing values return
        ``"keep"``, which hands nulls to the estimator as NaN.
        """
        return "error"

    @property
    def _features_col_name(self) -> str:
        """The vector column the fitted Spark model reads.

        Models saved before 0.10.0 read ``features``; everything else reads
        :attr:`FEATURES_COL`.
        """
        return getattr(self, "_features_col", None) or self.FEATURES_COL

    def _assemble_features(
        self,
        dataframe: DataFrame,
        feature_cols: List[str],
        output_col: Optional[str] = None,
        handle_invalid: Optional[str] = None,
    ) -> DataFrame:
        """Assemble feature columns into a single vector column.

        Always assembles from ``feature_cols``, even when ``dataframe`` already holds a
        column with the output name: that is a user column and is never reused.

        Args:
            dataframe: PySpark DataFrame with feature columns.
            feature_cols: List of column names to assemble.
            output_col: Name of the output vector column. Default is the
                model's internal features column.
            handle_invalid: ``VectorAssembler``'s ``handleInvalid`` policy. Defaults
                to :meth:`_handle_invalid`.

        Returns:
            DataFrame with assembled features column added.
        """
        output_col = output_col or self._features_col_name
        handle_invalid = handle_invalid or self._handle_invalid()
        if output_col in dataframe.columns:
            dataframe = dataframe.drop(output_col)

        assembler = VectorAssembler(
            inputCols=feature_cols,
            outputCol=output_col,
            handleInvalid=handle_invalid,
        )

        return assembler.transform(dataframe)

    # --- Shielding user columns from the estimator's output names ---

    def _internal_cols(self) -> List[str]:
        """Columns a Spark model reads or writes while it runs."""
        return [
            self._features_col_name,
            self.PREDICTION_COL,
            self.PROBABILITY_COL,
            self.RAW_PREDICTION_COL,
        ]

    def _shield_user_columns(self, dataframe: DataFrame) -> Tuple[DataFrame, Dict[str, str]]:
        """Rename user columns that collide with internal names; return the mapping."""
        shielded: Dict[str, str] = {}
        for name in self._internal_cols():
            if name in dataframe.columns:
                alias = f"__smallaxe_user_{name}__"
                dataframe = dataframe.withColumnRenamed(name, alias)
                shielded[alias] = name
        return dataframe, shielded

    def _unshield_user_columns(self, dataframe: DataFrame, shielded: Dict[str, str]) -> DataFrame:
        """Drop internal columns and give shielded user columns their names back."""
        for name in self._internal_cols():
            if name in dataframe.columns:
                dataframe = dataframe.drop(name)
        for alias, name in shielded.items():
            dataframe = dataframe.withColumnRenamed(alias, name)
        return dataframe

    def _transform_for_prediction(
        self, dataframe: DataFrame, feature_cols: List[str]
    ) -> Tuple[DataFrame, Dict[str, str]]:
        """Assemble features and run the fitted model, with user columns shielded.

        Returns the raw transform output and the shield mapping. Internal columns
        are still present and shielded user columns still carry their aliases;
        callers finish with :meth:`_unshield_user_columns`.
        """
        missing = [col for col in feature_cols if col not in dataframe.columns]
        if missing:
            raise ColumnNotFoundError(column=missing[0], available_columns=list(dataframe.columns))
        dataframe, shielded = self._shield_user_columns(dataframe)
        df_with_features = self._assemble_features(dataframe, feature_cols)
        return self._spark_model.transform(df_with_features), shielded

    def _check_output_col_free(self, dataframe: DataFrame, output_col: str) -> None:
        """Refuse to overwrite a user column with a prediction output."""
        if output_col in dataframe.columns:
            raise ValidationError(
                f"Column '{output_col}' already exists in the DataFrame and would be "
                f"overwritten. Pass a different output_col, or drop the column first."
            )

    @abstractmethod
    def _fit_spark_model(
        self,
        dataframe: DataFrame,
        label_col: str,
        feature_cols: List[str],
    ) -> Any:
        """Fit the Spark estimator on ``dataframe`` and keep it as ``self._spark_model``.

        Implementations also record ``self._feature_cols`` and ``self._label_col``.

        Args:
            dataframe: PySpark DataFrame with training data.
            label_col: Name of the label column.
            feature_cols: List of feature column names.

        Returns:
            The fitted Spark model.
        """

    def _predict_spark_model(
        self,
        dataframe: DataFrame,
        output_col: str = "predict_label",
    ) -> DataFrame:
        """Generate predictions using the fitted Spark MLlib model.

        Args:
            dataframe: PySpark DataFrame with feature columns.
            output_col: Name of the output prediction column.
                Default is 'predict_label'.

        Returns:
            The input columns, unchanged, plus ``output_col``. Classifiers also
            add the probability column.

        Raises:
            ModelNotFittedError: If the model has not been fitted.
            ColumnNotFoundError: If a feature column is not in ``dataframe``.
            ValidationError: If ``output_col`` is already a column of ``dataframe``.
        """
        if not hasattr(self, "_spark_model") or self._spark_model is None:
            raise ModelNotFittedError("Model has not been fitted. Call fit() before predict().")
        self._check_output_col_free(dataframe, output_col)

        predictions_df, shielded = self._transform_for_prediction(dataframe, self._feature_cols)

        keep = {self.PREDICTION_COL: output_col}
        # Classifiers also return the probability vector, unless the user owns that name.
        if (
            self.PROBABILITY_COL in predictions_df.columns
            and f"__smallaxe_user_{self.PROBABILITY_COL}__" not in shielded
        ):
            keep[self.PROBABILITY_COL] = self.PROBABILITY_COL
        return self._finish_prediction(predictions_df, shielded, keep)

    def _finish_prediction(
        self, dataframe: DataFrame, shielded: Dict[str, str], keep: Dict[str, str]
    ) -> DataFrame:
        """Drop internal columns, restore user columns, and name the kept outputs.

        Args:
            dataframe: Raw transform output.
            shielded: Mapping from :meth:`_shield_user_columns`.
            keep: Internal output column -> final name. Parked under temporary
                names first, so an output named like an internal column survives.
        """
        for internal, final in keep.items():
            dataframe = dataframe.withColumnRenamed(internal, f"__smallaxe_keep_{final}__")
        dataframe = self._unshield_user_columns(dataframe, shielded)
        for final in keep.values():
            dataframe = dataframe.withColumnRenamed(f"__smallaxe_keep_{final}__", final)
        return dataframe

    def _predict_proba_spark_model(
        self,
        dataframe: DataFrame,
        output_col: str = "probability",
    ) -> DataFrame:
        """Generate probability predictions for classification.

        Args:
            dataframe: PySpark DataFrame with feature columns.
            output_col: Name of the output probability column.
                Default is 'probability'.

        Returns:
            The input columns, unchanged, plus ``output_col``.

        Raises:
            ModelNotFittedError: If the model has not been fitted.
            ColumnNotFoundError: If a feature column is not in ``dataframe``.
            ValidationError: If ``output_col`` is already a column of ``dataframe``.
        """
        if not hasattr(self, "_spark_model") or self._spark_model is None:
            raise ModelNotFittedError(
                "Model has not been fitted. Call fit() before predict_proba()."
            )
        self._check_output_col_free(dataframe, output_col)

        predictions_df, shielded = self._transform_for_prediction(dataframe, self._feature_cols)
        keep = {}
        if self.PROBABILITY_COL in predictions_df.columns:
            keep[self.PROBABILITY_COL] = output_col
        return self._finish_prediction(predictions_df, shielded, keep)

    @abstractmethod
    def _get_feature_importance(self) -> Optional[List[float]]:
        """One importance score per feature column, in order, or None if unavailable.

        Raises:
            ModelNotFittedError: If the model has not been fitted.
        """

    @property
    def feature_importances(self) -> Optional[Dict[str, float]]:
        """Feature importance per feature column, as the algorithm defines it.

        Random Forest gives MLlib's normalised Gini importances, XGBoost the total gain,
        LightGBM split counts, CatBoost PredictionValuesChange (summing to 100).

        Returns:
            Dictionary mapping feature names to importance scores,
            or None if the fitted model cannot supply them.

        Raises:
            ModelNotFittedError: If the model has not been fitted.
        """
        importances = self._get_feature_importance()
        if importances is None:
            return None

        if not hasattr(self, "_feature_cols"):
            return None

        return dict(zip(self._feature_cols, importances))

    def _save_spark_model(self, path: str) -> None:
        """Save the Spark MLlib model to disk.

        Args:
            path: Directory path to save the model to.

        Raises:
            ModelNotFittedError: If the model has not been fitted.
        """
        import os

        if not hasattr(self, "_spark_model") or self._spark_model is None:
            raise ModelNotFittedError("Model has not been fitted. Call fit() before saving.")

        model_path = os.path.join(path, "spark_model")
        self._spark_model.write().overwrite().save(model_path)

    def _load_spark_model(self, path: str, model_class: type) -> None:
        """Load a Spark MLlib model from disk.

        Args:
            path: Directory path where the model is saved.
            model_class: The Spark MLlib model class to load.
        """
        import os

        model_path = os.path.join(path, "spark_model")
        self._spark_model = model_class.load(model_path)
