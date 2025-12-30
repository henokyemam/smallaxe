"""SparkModelMixin for Spark MLlib model operations."""

from abc import ABC, abstractmethod
from typing import Any, List, Optional

from pyspark.ml.feature import VectorAssembler
from pyspark.sql import DataFrame

from smallaxe.exceptions import ModelNotFittedError


class SparkModelMixin(ABC):
    """Mixin providing Spark MLlib model operations.

    This mixin provides common functionality for working with Spark MLlib models,
    including feature assembly, fitting, and prediction.
    """

    # Default column names
    FEATURES_COL = "features"
    PREDICTION_COL = "prediction"
    PROBABILITY_COL = "probability"
    RAW_PREDICTION_COL = "rawPrediction"

    @abstractmethod
    def _create_spark_estimator(self) -> Any:
        """Create the underlying Spark MLlib estimator.

        Subclasses must implement this to create their specific
        Spark MLlib estimator (e.g., RandomForestRegressor).

        Returns:
            A Spark MLlib estimator instance.
        """
        pass

    def _assemble_features(
        self,
        df: DataFrame,
        feature_cols: List[str],
        output_col: Optional[str] = None,
    ) -> DataFrame:
        """Assemble feature columns into a single vector column.

        Args:
            df: PySpark DataFrame with feature columns.
            feature_cols: List of column names to assemble.
            output_col: Name of the output vector column.
                Default is 'features'.

        Returns:
            DataFrame with assembled features column added.
        """
        output_col = output_col or self.FEATURES_COL

        # Check if features column already exists
        if output_col in df.columns:
            return df

        assembler = VectorAssembler(
            inputCols=feature_cols,
            outputCol=output_col,
            handleInvalid="skip",
        )

        return assembler.transform(df)

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
        # Assemble features
        df_with_features = self._assemble_features(df, feature_cols)

        # Create and configure the estimator
        estimator = self._create_spark_estimator()
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
            DataFrame with predictions added.

        Raises:
            ModelNotFittedError: If the model has not been fitted.
        """
        if not hasattr(self, "_spark_model") or self._spark_model is None:
            raise ModelNotFittedError(
                "Model has not been fitted. Call fit() before predict()."
            )

        # Assemble features
        df_with_features = self._assemble_features(df, self._feature_cols)

        # Generate predictions
        predictions_df = self._spark_model.transform(df_with_features)

        # Rename prediction column to output_col
        if output_col != self.PREDICTION_COL:
            predictions_df = predictions_df.withColumnRenamed(
                self.PREDICTION_COL, output_col
            )

        # Drop intermediate columns
        cols_to_drop = [self.FEATURES_COL]
        if self.RAW_PREDICTION_COL in predictions_df.columns:
            cols_to_drop.append(self.RAW_PREDICTION_COL)

        # Keep probability column if present (for classifiers)
        for col in cols_to_drop:
            if col in predictions_df.columns:
                predictions_df = predictions_df.drop(col)

        return predictions_df

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
            DataFrame with probability predictions added.

        Raises:
            ModelNotFittedError: If the model has not been fitted.
        """
        if not hasattr(self, "_spark_model") or self._spark_model is None:
            raise ModelNotFittedError(
                "Model has not been fitted. Call fit() before predict_proba()."
            )

        # Assemble features
        df_with_features = self._assemble_features(df, self._feature_cols)

        # Generate predictions (includes probability)
        predictions_df = self._spark_model.transform(df_with_features)

        # Rename probability column if needed
        if self.PROBABILITY_COL in predictions_df.columns and output_col != self.PROBABILITY_COL:
            predictions_df = predictions_df.withColumnRenamed(
                self.PROBABILITY_COL, output_col
            )

        # Drop intermediate columns
        cols_to_drop = [
            self.FEATURES_COL,
            self.PREDICTION_COL,
            self.RAW_PREDICTION_COL,
        ]
        for col in cols_to_drop:
            if col in predictions_df.columns:
                predictions_df = predictions_df.drop(col)

        return predictions_df

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
            raise ModelNotFittedError(
                "Model has not been fitted. Call fit() before saving."
            )

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
