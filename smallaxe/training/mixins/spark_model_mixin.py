"""SparkModelMixin for Spark MLlib model operations."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from pyspark.ml.feature import VectorAssembler
from pyspark.sql import DataFrame

from smallaxe.exceptions import ModelNotFittedError, ValidationError


class SparkModelMixin(ABC):
    """Mixin providing Spark MLlib model operations.

    This mixin provides common functionality for working with Spark MLlib models,
    including feature assembly, fitting, and prediction.

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

    @abstractmethod
    def _create_spark_estimator(
        self,
        features_col: Optional[str] = None,
        label_col: Optional[str] = None,
        prediction_col: Optional[str] = None,
    ) -> Any:
        """Create the underlying Spark MLlib estimator.

        Subclasses must implement this to create their specific
        Spark MLlib estimator (e.g., RandomForestRegressor).

        For estimators that accept column names via constructor (XGBoost,
        LightGBM), use the provided column params. For estimators that use
        setters (native Spark ML), ignore these and let the base class set
        them via setters.

        Args:
            features_col: Features column name.
            label_col: Label column name.
            prediction_col: Prediction column name.

        Returns:
            A Spark MLlib estimator instance.
        """
        pass

    def _uses_constructor_col_params(self) -> bool:
        """Whether this estimator takes column names via constructor.

        Override and return True for estimators like XGBoost and LightGBM
        that require column names at construction time rather than via setters.
        """
        return False

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

        Models saved before 0.9.0 read ``features``; everything else reads
        :attr:`FEATURES_COL`.
        """
        return getattr(self, "_features_col", None) or self.FEATURES_COL

    def _assemble_features(
        self,
        df: DataFrame,
        feature_cols: List[str],
        output_col: Optional[str] = None,
        handle_invalid: Optional[str] = None,
    ) -> DataFrame:
        """Assemble feature columns into a single vector column.

        Always assembles from ``feature_cols``, even when ``df`` already holds a
        column with the output name: that is a user column and is never reused.

        Args:
            df: PySpark DataFrame with feature columns.
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
        if output_col in df.columns:
            df = df.drop(output_col)

        assembler = VectorAssembler(
            inputCols=feature_cols,
            outputCol=output_col,
            handleInvalid=handle_invalid,
        )

        return assembler.transform(df)

    # --- Shielding user columns from the estimator's output names ---

    def _internal_cols(self) -> List[str]:
        """Columns a Spark model reads or writes while it runs."""
        return [
            self._features_col_name,
            self.PREDICTION_COL,
            self.PROBABILITY_COL,
            self.RAW_PREDICTION_COL,
        ]

    def _shield_user_columns(self, df: DataFrame) -> Tuple[DataFrame, Dict[str, str]]:
        """Rename user columns that collide with internal names; return the mapping."""
        shielded: Dict[str, str] = {}
        for name in self._internal_cols():
            if name in df.columns:
                alias = f"__smallaxe_user_{name}__"
                df = df.withColumnRenamed(name, alias)
                shielded[alias] = name
        return df, shielded

    def _unshield_user_columns(self, df: DataFrame, shielded: Dict[str, str]) -> DataFrame:
        """Drop internal columns and give shielded user columns their names back."""
        for name in self._internal_cols():
            if name in df.columns:
                df = df.drop(name)
        for alias, name in shielded.items():
            df = df.withColumnRenamed(alias, name)
        return df

    def _transform_for_prediction(
        self, df: DataFrame, feature_cols: List[str]
    ) -> Tuple[DataFrame, Dict[str, str]]:
        """Assemble features and run the fitted model, with user columns shielded.

        Returns the raw transform output and the shield mapping. Internal columns
        are still present and shielded user columns still carry their aliases;
        callers finish with :meth:`_unshield_user_columns`.
        """
        df, shielded = self._shield_user_columns(df)
        df_with_features = self._assemble_features(df, feature_cols)
        return self._spark_model.transform(df_with_features), shielded

    def _check_output_col_free(self, df: DataFrame, output_col: str) -> None:
        """Refuse to overwrite a user column with a prediction output."""
        if output_col in df.columns:
            raise ValidationError(
                f"Column '{output_col}' already exists in the DataFrame and would be "
                f"overwritten. Pass a different output_col, or drop the column first."
            )

    def _fit_spark_model(
        self,
        df: DataFrame,
        label_col: str,
        feature_cols: List[str],
    ) -> Any:
        """Fit a Spark MLlib model.

        Args:
            df: PySpark DataFrame with training data.
            label_col: Name of the label column.
            feature_cols: List of feature column names.

        Returns:
            Fitted Spark MLlib model.
        """
        self._features_col = self.FEATURES_COL
        df, _ = self._shield_user_columns(df)
        df_with_features = self._assemble_features(df, feature_cols)

        # Create the estimator with column params
        estimator = self._create_spark_estimator(
            features_col=self.FEATURES_COL,
            label_col=label_col,
            prediction_col=self.PREDICTION_COL,
        )

        # For estimators that use setters (native Spark ML), set column names
        if not self._uses_constructor_col_params():
            estimator.setLabelCol(label_col)
            estimator.setFeaturesCol(self.FEATURES_COL)
            estimator.setPredictionCol(self.PREDICTION_COL)

        # Store feature columns for prediction
        self._feature_cols = feature_cols
        self._label_col = label_col

        # Fit the model
        self._spark_model = estimator.fit(df_with_features)

        return self._spark_model

    def _predict_spark_model(
        self,
        df: DataFrame,
        output_col: str = "predict_label",
    ) -> DataFrame:
        """Generate predictions using the fitted Spark MLlib model.

        Args:
            df: PySpark DataFrame with feature columns.
            output_col: Name of the output prediction column.
                Default is 'predict_label'.

        Returns:
            The input columns, unchanged, plus ``output_col``. Classifiers also
            add the probability column.

        Raises:
            ModelNotFittedError: If the model has not been fitted.
            ValidationError: If ``output_col`` is already a column of ``df``.
        """
        if not hasattr(self, "_spark_model") or self._spark_model is None:
            raise ModelNotFittedError("Model has not been fitted. Call fit() before predict().")
        self._check_output_col_free(df, output_col)

        predictions_df, shielded = self._transform_for_prediction(df, self._feature_cols)

        keep = {self.PREDICTION_COL: output_col}
        # Classifiers also return the probability vector, unless the user owns that name.
        if (
            self.PROBABILITY_COL in predictions_df.columns
            and f"__smallaxe_user_{self.PROBABILITY_COL}__" not in shielded
        ):
            keep[self.PROBABILITY_COL] = self.PROBABILITY_COL
        return self._finish_prediction(predictions_df, shielded, keep)

    def _finish_prediction(
        self, df: DataFrame, shielded: Dict[str, str], keep: Dict[str, str]
    ) -> DataFrame:
        """Drop internal columns, restore user columns, and name the kept outputs.

        Args:
            df: Raw transform output.
            shielded: Mapping from :meth:`_shield_user_columns`.
            keep: Internal output column -> final name. Parked under temporary
                names first, so an output named like an internal column survives.
        """
        for internal, final in keep.items():
            df = df.withColumnRenamed(internal, f"__smallaxe_keep_{final}__")
        df = self._unshield_user_columns(df, shielded)
        for final in keep.values():
            df = df.withColumnRenamed(f"__smallaxe_keep_{final}__", final)
        return df

    def _predict_proba_spark_model(
        self,
        df: DataFrame,
        output_col: str = "probability",
    ) -> DataFrame:
        """Generate probability predictions for classification.

        Args:
            df: PySpark DataFrame with feature columns.
            output_col: Name of the output probability column.
                Default is 'probability'.

        Returns:
            The input columns, unchanged, plus ``output_col``.

        Raises:
            ModelNotFittedError: If the model has not been fitted.
            ValidationError: If ``output_col`` is already a column of ``df``.
        """
        if not hasattr(self, "_spark_model") or self._spark_model is None:
            raise ModelNotFittedError(
                "Model has not been fitted. Call fit() before predict_proba()."
            )
        self._check_output_col_free(df, output_col)

        predictions_df, shielded = self._transform_for_prediction(df, self._feature_cols)
        keep = {}
        if self.PROBABILITY_COL in predictions_df.columns:
            keep[self.PROBABILITY_COL] = output_col
        return self._finish_prediction(predictions_df, shielded, keep)

    def _get_feature_importance(self) -> Optional[List[float]]:
        """Get feature importance scores from the fitted model.

        Returns:
            List of feature importance scores, or None if not available.

        Raises:
            ModelNotFittedError: If the model has not been fitted.
        """
        if not hasattr(self, "_spark_model") or self._spark_model is None:
            raise ModelNotFittedError(
                "Model has not been fitted. Call fit() before getting feature importance."
            )

        if hasattr(self._spark_model, "featureImportances"):
            return list(self._spark_model.featureImportances.toArray())
        return None

    @property
    def feature_importances(self) -> Optional[dict]:
        """Get feature importance as a dictionary mapping feature names to scores.

        Returns:
            Dictionary mapping feature names to importance scores,
            or None if not available.

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
