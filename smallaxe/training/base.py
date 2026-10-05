"""Base classes for training models."""

import os
import warnings
from contextlib import nullcontext
from typing import Any, Dict, List, Literal, Mapping, Optional, Union

from pyspark.sql import DataFrame
from pyspark.sql.types import NumericType

from smallaxe import _fs
from smallaxe.exceptions import (
    ColumnNotFoundError,
    ModelNotFittedError,
    ValidationError,
)
from smallaxe.training import algorithm as _algorithm
from smallaxe.training.algorithm import Algorithm
from smallaxe.training.mixins import (
    MetadataMixin,
    ParamMixin,
    PersistenceMixin,
    SparkModelMixin,
    ValidationMixin,
)


class BaseModel(
    ParamMixin,
    PersistenceMixin,
    ValidationMixin,
    MetadataMixin,
    SparkModelMixin,
):
    """Base class for all training models.

    This class combines functionality from all mixins to provide a complete
    model interface with parameter management, persistence, validation,
    metadata tracking, and Spark MLlib integration.

    Args:
        task: The task type for this model. Options depend on the model type:
            - Regression: 'simple_regression'
            - Classification: 'binary', 'multiclass'

    Attributes:
        task: The task type string.
        task_type: The general task type ('regression' or 'classification').

    Example:
        >>> model = SomeModel(task='simple_regression')
        >>> model.fit(df, label_col='target', feature_cols=['f1', 'f2'])
        >>> predictions = model.predict(df)
    """

    # Valid task types for each task category
    REGRESSION_TASKS = {"simple_regression"}
    CLASSIFICATION_TASKS = {"binary", "multiclass"}
    VALID_TASKS = REGRESSION_TASKS | CLASSIFICATION_TASKS

    # Valid validation strategies
    VALID_VALIDATION = {"none", "train_test", "kfold"}

    # Valid cache strategies
    VALID_CACHE_STRATEGIES = {"none", "memory", "disk"}

    def __init__(self, task: str) -> None:
        """Initialize the base model.

        Args:
            task: The task type for this model.

        Raises:
            ValidationError: If task is not a valid task type.
        """
        if task not in self.VALID_TASKS:
            raise ValidationError(
                f"Invalid task '{task}'. Valid tasks are: {sorted(self.VALID_TASKS)}"
            )
        self._task = task
        self._params: Dict[str, Any] = {}
        self._spark_model = None
        self._feature_cols: List[str] = []
        self._label_col: Optional[str] = None
        self._exclude_cols: List[str] = []
        self._validation_scores: Optional[Dict[str, Any]] = None

    @property
    def task(self) -> str:
        """Get the task type string.

        Returns:
            The task type (e.g., 'simple_regression', 'binary').
        """
        return self._task

    @property
    def task_type(self) -> Literal["regression", "classification"]:
        """Get the general task type.

        Returns:
            'regression' for regression tasks, 'classification' for classification tasks.
        """
        if self._task in self.REGRESSION_TASKS:
            return "regression"
        return "classification"

    @property
    def validation_scores(self) -> Optional[Dict[str, Any]]:
        """Get validation scores from training.

        Returns:
            Dictionary containing validation scores, or None if no validation was performed.

        Raises:
            ModelNotFittedError: If accessed before the model is fitted.
        """
        if not self._is_fitted:
            raise ModelNotFittedError("No validation scores available. Model has not been fitted.")
        return self._validation_scores

    def fit(
        self,
        df: DataFrame,
        label_col: str,
        feature_cols: Optional[List[str]] = None,
        exclude_cols: Optional[List[str]] = None,
        validation: str = "none",
        test_size: float = 0.2,
        n_folds: int = 5,
        stratified: Optional[bool] = None,
        cache_strategy: str = "none",
    ) -> "BaseModel":
        """Fit the model to training data.

        Args:
            df: PySpark DataFrame containing training data.
            label_col: Name of the label/target column.
            feature_cols: List of feature column names. If None, uses all
                numeric columns except label_col and exclude_cols.
            exclude_cols: List of columns to exclude from features.
                Typically includes ID columns, timestamps, etc.
            validation: Validation strategy. Options:
                - 'none': No validation (default)
                - 'train_test': Single train/test split
                - 'kfold': K-fold cross-validation
            test_size: Proportion for test set when validation='train_test'.
                Default is 0.2.
            n_folds: Number of folds when validation='kfold'. Default is 5.
            stratified: Whether to use stratified splitting. If None, uses
                stratified for classification tasks and non-stratified for regression.
            cache_strategy: DataFrame caching strategy. Options:
                - 'none': No caching (default)
                - 'memory': Cache in memory (MEMORY_ONLY)
                - 'disk': Cache with disk spillover (MEMORY_AND_DISK)

        Returns:
            self: The fitted model instance.

        Raises:
            ValidationError: If validation strategy is invalid.
            ValidationError: If cache_strategy is invalid.
            ColumnNotFoundError: If label_col or feature_cols are not found.
        """
        # Validate parameters
        self._validate_fit_params(
            df=df,
            label_col=label_col,
            feature_cols=feature_cols,
            exclude_cols=exclude_cols,
            validation=validation,
            cache_strategy=cache_strategy,
        )

        # Store exclude_cols
        self._exclude_cols = exclude_cols or []

        # Determine feature columns
        if feature_cols is None:
            feature_cols = self._infer_feature_cols(df, label_col, self._exclude_cols)

        # Validate columns exist
        self._validate_columns(df, label_col, feature_cols)

        # Store label column
        self._label_col = label_col

        # Determine stratification
        if stratified is None:
            stratified = self.task_type == "classification"

        # Apply caching
        cached_df = self._apply_cache_strategy(df, cache_strategy)

        try:
            # Perform training with validation
            if validation == "none":
                self._fit_no_validation(cached_df, label_col, feature_cols)
            elif validation == "train_test":
                self._fit_train_test(cached_df, label_col, feature_cols, test_size, stratified)
            elif validation == "kfold":
                self._fit_kfold(cached_df, label_col, feature_cols, n_folds, stratified)

            # Capture metadata
            self._capture_training_metadata(cached_df, label_col, feature_cols)

            # Mark as fitted
            self._is_fitted = True

        finally:
            # Unpersist if we cached
            if cache_strategy != "none" and cached_df is not None:
                cached_df.unpersist()

        return self

    def _validate_fit_params(
        self,
        df: DataFrame,
        label_col: str,
        feature_cols: Optional[List[str]],
        exclude_cols: Optional[List[str]],
        validation: str,
        cache_strategy: str,
    ) -> None:
        """Validate fit parameters.

        Args:
            df: Training DataFrame.
            label_col: Name of the label column.
            feature_cols: List of feature columns.
            exclude_cols: List of columns to exclude.
            validation: Validation strategy.
            cache_strategy: Caching strategy.

        Raises:
            ValidationError: If any parameter is invalid.
        """
        if not label_col:
            raise ValidationError("label_col is required.")

        if validation not in self.VALID_VALIDATION:
            raise ValidationError(
                f"Invalid validation strategy '{validation}'. "
                f"Valid options are: {sorted(self.VALID_VALIDATION)}"
            )

        if cache_strategy not in self.VALID_CACHE_STRATEGIES:
            raise ValidationError(
                f"Invalid cache_strategy '{cache_strategy}'. "
                f"Valid options are: {sorted(self.VALID_CACHE_STRATEGIES)}"
            )

    def _infer_feature_cols(
        self,
        df: DataFrame,
        label_col: str,
        exclude_cols: List[str],
    ) -> List[str]:
        """Infer feature columns from DataFrame schema.

        Args:
            df: DataFrame to infer columns from.
            label_col: Label column to exclude.
            exclude_cols: Additional columns to exclude.

        Returns:
            List of inferred feature column names.
        """
        excluded = set(exclude_cols) | {label_col}
        feature_cols = []

        for field in df.schema.fields:
            if field.name not in excluded:
                # Only include numeric columns
                if isinstance(field.dataType, NumericType):
                    feature_cols.append(field.name)

        return feature_cols

    def _validate_columns(
        self,
        df: DataFrame,
        label_col: str,
        feature_cols: List[str],
    ) -> None:
        """Validate that required columns exist in DataFrame.

        Args:
            df: DataFrame to validate.
            label_col: Label column name.
            feature_cols: Feature column names.

        Raises:
            ColumnNotFoundError: If any column is not found.
            ValidationError: If no feature columns are provided.
        """
        available = set(df.columns)

        if label_col not in available:
            raise ColumnNotFoundError(
                column=label_col,
                available_columns=list(available),
            )

        if not feature_cols:
            raise ValidationError(
                "No feature columns provided or inferred. "
                "Please specify feature_cols or ensure DataFrame has numeric columns."
            )

        for col in feature_cols:
            if col not in available:
                raise ColumnNotFoundError(
                    column=col,
                    available_columns=list(available),
                )

    def _apply_cache_strategy(
        self,
        df: DataFrame,
        cache_strategy: str,
    ) -> DataFrame:
        """Apply caching strategy to DataFrame.

        Args:
            df: DataFrame to cache.
            cache_strategy: Caching strategy.

        Returns:
            Cached or uncached DataFrame.
        """
        if cache_strategy == "none":
            return df
        elif cache_strategy == "memory":
            from pyspark import StorageLevel

            return df.persist(StorageLevel.MEMORY_ONLY)
        elif cache_strategy == "disk":
            from pyspark import StorageLevel

            return df.persist(StorageLevel.MEMORY_AND_DISK)
        return df

    def _fit_no_validation(
        self,
        df: DataFrame,
        label_col: str,
        feature_cols: List[str],
    ) -> None:
        """Fit model without validation.

        Args:
            df: Training DataFrame.
            label_col: Label column name.
            feature_cols: Feature column names.
        """
        self._fit_spark_model(df, label_col, feature_cols)
        self._validation_scores = None

    def _fit_train_test(
        self,
        df: DataFrame,
        label_col: str,
        feature_cols: List[str],
        test_size: float,
        stratified: bool,
    ) -> None:
        """Fit model with train/test validation.

        Args:
            df: Training DataFrame.
            label_col: Label column name.
            feature_cols: Feature column names.
            test_size: Proportion for test set.
            stratified: Whether to use stratified split.
        """
        # Split data
        train_df, test_df = self._train_test_split(
            df,
            test_size=test_size,
            stratified=stratified,
            label_col=label_col if stratified else None,
        )

        # Fit on training data
        self._fit_spark_model(train_df, label_col, feature_cols)

        # Evaluate on test data
        self._validation_scores = self._evaluate(test_df, label_col)
        self._validation_scores["validation_type"] = "train_test"
        self._validation_scores["test_size"] = test_size

    def _fit_kfold(
        self,
        df: DataFrame,
        label_col: str,
        feature_cols: List[str],
        n_folds: int,
        stratified: bool,
    ) -> None:
        """Fit model with k-fold cross-validation.

        Args:
            df: Training DataFrame.
            label_col: Label column name.
            feature_cols: Feature column names.
            n_folds: Number of folds.
            stratified: Whether to use stratified split.
        """
        fold_scores: List[Dict[str, Any]] = []

        # Iterate through folds
        for fold_idx, (train_df, val_df) in enumerate(
            self._kfold_split(
                df,
                n_folds=n_folds,
                stratified=stratified,
                label_col=label_col if stratified else None,
            )
        ):
            # Fit on this fold's training data
            self._fit_spark_model(train_df, label_col, feature_cols)

            # Evaluate on validation data
            scores = self._evaluate(val_df, label_col)
            scores["fold"] = fold_idx
            fold_scores.append(scores)

        # Fit final model on all data
        self._fit_spark_model(df, label_col, feature_cols)

        # Aggregate scores
        self._validation_scores = self._aggregate_fold_scores(fold_scores, n_folds)

    def _aggregate_fold_scores(
        self,
        fold_scores: List[Dict[str, Any]],
        n_folds: int,
    ) -> Dict[str, Any]:
        """Aggregate scores across folds.

        Args:
            fold_scores: List of score dictionaries from each fold.
            n_folds: Number of folds.

        Returns:
            Dictionary with aggregated scores.
        """
        if not fold_scores:
            return {"validation_type": "kfold", "n_folds": n_folds}

        # Get metric names (exclude 'fold' key)
        metric_names = [k for k in fold_scores[0].keys() if k != "fold"]

        result: Dict[str, Any] = {
            "validation_type": "kfold",
            "n_folds": n_folds,
            "fold_scores": fold_scores,
        }

        # Calculate mean and std for each metric
        for metric in metric_names:
            values = [s[metric] for s in fold_scores if s[metric] is not None]
            if values:
                import statistics

                result[f"mean_{metric}"] = statistics.mean(values)
                if len(values) > 1:
                    result[f"std_{metric}"] = statistics.stdev(values)

        return result

    def _evaluate(
        self,
        df: DataFrame,
        label_col: str,
    ) -> Dict[str, Any]:
        """Evaluate model on a dataset.

        Args:
            df: DataFrame to evaluate on.
            label_col: Label column name.

        Returns:
            Dictionary of evaluation metrics.
        """
        if self.task_type == "regression":
            predictions_df = self._predict_spark_model(df, output_col="prediction")
            return self._compute_regression_metrics(predictions_df, label_col)
        else:
            return self._compute_classification_metrics_single_pass(df, label_col)

    def _compute_regression_metrics(
        self,
        predictions_df: DataFrame,
        label_col: str,
    ) -> Dict[str, Any]:
        """Compute regression metrics.

        Args:
            predictions_df: DataFrame with predictions.
            label_col: Label column name.

        Returns:
            Dictionary of regression metrics.
        """
        from smallaxe.metrics.regression import mae, mape, mse, r2, rmse

        return {
            "mse": mse(predictions_df, label_col, "prediction"),
            "rmse": rmse(predictions_df, label_col, "prediction"),
            "mae": mae(predictions_df, label_col, "prediction"),
            "r2": r2(predictions_df, label_col, "prediction"),
            "mape": mape(predictions_df, label_col, "prediction"),
        }

    def _compute_classification_metrics_single_pass(
        self,
        df: DataFrame,
        label_col: str,
    ) -> Dict[str, Any]:
        """Compute classification metrics in a single transform pass.

        Args:
            df: Original DataFrame to evaluate.
            label_col: Label column name.

        Returns:
            Dictionary of classification metrics.
        """
        from smallaxe.metrics.classification import (
            accuracy,
            auc_pr,
            auc_roc,
            f1_score,
            log_loss,
            precision,
            recall,
        )

        # Single transform pass produces predictions AND probabilities
        df_with_features = self._assemble_features(df, self._feature_cols)
        full_df = self._spark_model.transform(df_with_features)

        # Basic metrics using class predictions
        metrics = {
            "accuracy": accuracy(full_df, label_col, self.PREDICTION_COL),
            "precision": precision(full_df, label_col, self.PREDICTION_COL),
            "recall": recall(full_df, label_col, self.PREDICTION_COL),
            "f1_score": f1_score(full_df, label_col, self.PREDICTION_COL),
        }

        # Probability-based metrics (only for binary classification)
        if self._task == "binary":
            try:
                from pyspark.ml.functions import vector_to_array
                from pyspark.sql import functions as F

                proba_df = full_df.withColumn(
                    "prob_positive", vector_to_array(F.col(self.PROBABILITY_COL))[1]
                )

                metrics["auc_roc"] = auc_roc(proba_df, label_col, "prob_positive")
                metrics["auc_pr"] = auc_pr(proba_df, label_col, "prob_positive")
                metrics["log_loss"] = log_loss(proba_df, label_col, "prob_positive")
            except Exception:
                metrics["auc_roc"] = None
                metrics["auc_pr"] = None
                metrics["log_loss"] = None

        return metrics

    def _capture_training_metadata(
        self,
        df: DataFrame,
        label_col: str,
        feature_cols: List[str],
    ) -> None:
        """Capture metadata from training.

        Args:
            df: Training DataFrame.
            label_col: Label column name.
            feature_cols: Feature column names.
        """
        # Capture base metadata
        self._capture_metadata(df, label_col, feature_cols)

        # Capture label statistics
        label_stats = self._capture_label_stats(df, label_col, self.task_type)
        for key, value in label_stats.items():
            self._update_metadata(key, value)

        # Add task information
        self._update_metadata("task", self._task)
        self._update_metadata("task_type", self.task_type)

        # Add exclude_cols
        if self._exclude_cols:
            self._update_metadata("exclude_cols", self._exclude_cols)

    def predict(self, df: DataFrame, output_col: str = "predict_label") -> DataFrame:
        """Generate predictions for input data.

        Args:
            df: PySpark DataFrame with feature columns.
            output_col: Name for the prediction column. Default is 'predict_label'.

        Returns:
            DataFrame with predictions added.

        Raises:
            ModelNotFittedError: If the model has not been fitted.
        """
        if not self._is_fitted:
            raise ModelNotFittedError("Model has not been fitted. Call fit() before predict().")

        return self._predict_spark_model(df, output_col=output_col)

    def clone(self) -> "BaseModel":
        """Return an unfitted copy with the same task and parameter values."""
        clone = type(self)(task=self._task)
        clone._params = dict(self._params)
        return clone

    # --- PersistenceMixin abstract methods ---

    def _get_persistence_state(self) -> Dict[str, Any]:
        """Get the state dict to persist.

        Returns:
            Dictionary containing all model state.
        """
        return {
            "task": self._task,
            "params": self._params,
            "feature_cols": self._feature_cols,
            "label_col": self._label_col,
            "exclude_cols": self._exclude_cols,
            "is_fitted": self._is_fitted,
            "metadata": self._get_metadata_for_persistence(),
            "validation_scores": self._validation_scores,
        }

    def _set_persistence_state(self, state: Dict[str, Any]) -> None:
        """Restore model state from a dict.

        Args:
            state: Dictionary containing saved model state.
        """
        self._task = state.get("task", "simple_regression")
        self._params = state.get("params", {})
        self._feature_cols = state.get("feature_cols", [])
        self._label_col = state.get("label_col")
        self._exclude_cols = state.get("exclude_cols", [])
        self._is_fitted = state.get("is_fitted", False)
        self._validation_scores = state.get("validation_scores")

        # Restore metadata
        if "metadata" in state:
            self._restore_metadata_from_persistence(state["metadata"])

    def _save_artifacts(self, path: str) -> None:
        """Save additional model artifacts.

        Args:
            path: Directory path where artifacts should be saved.
        """
        if self._spark_model is not None:
            self._save_spark_model(path)

    def _load_artifacts(self, path: str) -> None:
        """Load additional model artifacts.

        Args:
            path: Directory path where artifacts are stored.
        """
        # Subclasses must implement this to load the specific Spark model type
        pass


class BaseRegressor(BaseModel):
    """Base class for regression models.

    This class provides a foundation for regression models with task type
    validation and regression-specific defaults.

    Args:
        task: The regression task type. Default is 'simple_regression'.

    Example:
        >>> model = SomeRegressor(task='simple_regression')
        >>> model.fit(df, label_col='target', feature_cols=['f1', 'f2'])
        >>> predictions = model.predict(df)
    """

    def __init__(self, task: str = "simple_regression") -> None:
        """Initialize the base regressor.

        Args:
            task: The regression task type.

        Raises:
            ValidationError: If task is not a valid regression task.
        """
        if task not in self.REGRESSION_TASKS:
            raise ValidationError(
                f"Invalid regression task '{task}'. "
                f"Valid tasks are: {sorted(self.REGRESSION_TASKS)}"
            )
        super().__init__(task)


class BaseClassifier(BaseModel):
    """Base class for classification models.

    This class provides a foundation for classification models with task type
    validation, classification-specific defaults, and probability prediction.

    Args:
        task: The classification task type. Default is 'binary'.

    Example:
        >>> model = SomeClassifier(task='binary')
        >>> model.fit(df, label_col='label', feature_cols=['f1', 'f2'])
        >>> predictions = model.predict(df)
        >>> probabilities = model.predict_proba(df)
    """

    def __init__(self, task: str = "binary") -> None:
        """Initialize the base classifier.

        Args:
            task: The classification task type.

        Raises:
            ValidationError: If task is not a valid classification task.
        """
        if task not in self.CLASSIFICATION_TASKS:
            raise ValidationError(
                f"Invalid classification task '{task}'. "
                f"Valid tasks are: {sorted(self.CLASSIFICATION_TASKS)}"
            )
        super().__init__(task)

    def predict_proba(
        self,
        df: DataFrame,
        output_col: str = "probability",
    ) -> DataFrame:
        """Generate probability predictions for input data.

        Args:
            df: PySpark DataFrame with feature columns.
            output_col: Name for the probability column. Default is 'probability'.

        Returns:
            DataFrame with probability predictions added.

        Raises:
            ModelNotFittedError: If the model has not been fitted.
        """
        if not self._is_fitted:
            raise ModelNotFittedError(
                "Model has not been fitted. Call fit() before predict_proba()."
            )

        return self._predict_proba_spark_model(df, output_col=output_col)


class Model(BaseModel):
    """A model bound to one :class:`~smallaxe.training.algorithm.Algorithm`.

    Everything algorithm-specific (parameter table, estimator and model classes,
    column wiring, task-fixed settings, importances, dependency, session setup) is
    read from the algorithm record, so fit, predict, save, load, and clone are
    written once here. Use :class:`Regressor` or :class:`Classifier`, or the
    ``Regressors`` / ``Classifiers`` factories.

    Args:
        algorithm: Algorithm name (``"xgboost"``) or record.
        task: The task to train.

    Raises:
        DependencyError: If the algorithm's optional package is not installed.
        ValidationError: If the task is invalid or unsupported by the algorithm.
    """

    def __init__(self, algorithm: Union[str, Algorithm], task: str) -> None:
        self._algorithm = _algorithm.get(algorithm) if isinstance(algorithm, str) else algorithm
        self._algorithm.require()
        super().__init__(task)
        if task not in self._algorithm.tasks:
            raise ValidationError(
                f"Algorithm '{self._algorithm.name}' does not support task '{task}'. "
                f"Supported tasks are: {sorted(self._algorithm.tasks)}"
            )

    @property
    def algorithm(self) -> Algorithm:
        """The algorithm record this model is bound to."""
        return self._algorithm

    # --- ParamMixin ---

    @property
    def params(self) -> Dict[str, str]:
        """Parameter descriptions for this algorithm and task."""
        return {p.name: p.description for p in self._algorithm.visible_params(self._task)}

    @property
    def default_params(self) -> Dict[str, Any]:
        """Default parameter values for this algorithm and task."""
        return {p.name: p.default for p in self._algorithm.visible_params(self._task)}

    # --- SparkModelMixin ---

    def _column_names(self, label_col: str) -> Dict[str, str]:
        """Column role -> name handed to the estimator; classification adds probabilities."""
        columns = {
            "features": self.FEATURES_COL,
            "label": label_col,
            "prediction": self.PREDICTION_COL,
        }
        if self.task_type == "classification":
            columns["probability"] = self.PROBABILITY_COL
            columns["raw_prediction"] = self.RAW_PREDICTION_COL
        return columns

    def _build_estimator(self, label_col: str, extra: Mapping[str, Any]) -> Any:
        """Construct the Spark estimator from the algorithm record and current params."""
        algorithm = self._algorithm
        ctor, setters = _algorithm.estimator_kwargs(
            algorithm, self._task, self.get_params(), self._column_names(label_col)
        )
        ctor.update(extra)
        estimator = algorithm.estimator_class(self._task)(**ctor)
        for setter, value in setters:
            getattr(estimator, setter)(value)
        if algorithm.cols is None:
            estimator.setLabelCol(label_col)
            estimator.setFeaturesCol(self.FEATURES_COL)
            estimator.setPredictionCol(self.PREDICTION_COL)
        return estimator

    def _create_spark_estimator(
        self,
        features_col: Optional[str] = None,
        label_col: Optional[str] = None,
        prediction_col: Optional[str] = None,
    ) -> Any:
        return self._build_estimator(label_col or "label", {})

    def _fit_spark_model(self, df: DataFrame, label_col: str, feature_cols: List[str]) -> Any:
        algorithm = self._algorithm
        if algorithm.prepare is not None:
            algorithm.prepare()
        df_with_features = self._assemble_features(df, feature_cols)
        context = (
            algorithm.fit_context(self.get_params())
            if algorithm.fit_context is not None
            else nullcontext({})
        )
        with context as extra:
            estimator = self._build_estimator(label_col, extra)
            self._feature_cols = feature_cols
            self._label_col = label_col
            self._spark_model = estimator.fit(df_with_features)
        return self._spark_model

    def _get_feature_importance(self) -> Optional[List[float]]:
        if self._spark_model is None:
            raise ModelNotFittedError(
                "Model has not been fitted. Call fit() before getting feature importance."
            )
        if self._algorithm.importances is None:
            return None
        return self._algorithm.importances(self._spark_model, len(self._feature_cols))

    def clone(self) -> "Model":
        """Return an unfitted copy with the same algorithm, task, and parameter values."""
        kind = Classifier if self.task_type == "classification" else Regressor
        clone = kind(self._algorithm, self._task)
        clone._params = dict(self._params)
        return clone

    # --- PersistenceMixin ---

    def _get_persistence_state(self) -> Dict[str, Any]:
        state = super()._get_persistence_state()
        state["algorithm"] = self._algorithm.name
        return state

    def _load_artifacts(self, path: str) -> None:
        if self._algorithm.prepare is not None:
            self._algorithm.prepare()
        self._load_spark_model(path, self._algorithm.model_class(self._task))

    @classmethod
    def load(cls, path: str) -> "Model":
        """Load a saved model, returning a :class:`Regressor` or :class:`Classifier`.

        Reads ``algorithm`` from ``metadata.json``; artifacts written by smallaxe 0.8.x
        record the old class name instead and are mapped through
        :data:`~smallaxe.training.algorithm.LEGACY_CLASS_NAMES`.

        Raises:
            ValidationError: If the path or metadata is invalid, or the saved model is
                not the kind (or algorithm) this class expects.
            DependencyError: If the saved model's algorithm is not installed.
        """
        if not path or not _fs.exists(path):
            raise ValidationError(f"Model path does not exist: {path}")
        metadata_path = os.path.join(path, "metadata.json")
        if not _fs.exists(metadata_path):
            raise ValidationError(f"Invalid model directory: metadata.json not found at {path}")

        state = _fs.read_json(metadata_path)
        saved_class = state.pop("__class__", None)
        state.pop("__module__", None)

        name = state.get("algorithm")
        if name is None:
            if saved_class not in _algorithm.LEGACY_CLASS_NAMES:
                raise ValidationError(
                    "Model metadata does not contain 'algorithm'. "
                    "This may be an older model format or corrupted metadata."
                )
            name = _algorithm.LEGACY_CLASS_NAMES[saved_class][0]
        algorithm = _algorithm.get(name)
        algorithm.require()

        task = state.get("task")
        if task in BaseModel.CLASSIFICATION_TASKS:
            kind, task_type = Classifier, "classification"
        else:
            kind, task_type = Regressor, "regression"
        saved_name = _algorithm.legacy_class_name(name, task_type)
        expected_kind = None
        if issubclass(cls, Regressor):
            expected_kind = Regressor
        elif issubclass(cls, Classifier):
            expected_kind = Classifier
        if expected_kind is not None and expected_kind is not kind:
            raise ValidationError(
                f"Model type mismatch: expected {cls.__name__}, got {saved_name}."
            )
        alias_algorithm = getattr(cls, "_alias_algorithm", None)
        if alias_algorithm is not None and alias_algorithm != name:
            raise ValidationError(
                f"Model type mismatch: expected {cls.__name__}, got {saved_name}."
            )

        instance = kind.__new__(kind)
        instance._algorithm = algorithm
        instance._spark_model = None
        instance._set_persistence_state(state)
        instance._load_artifacts(path)
        return instance


class Regressor(Model, BaseRegressor):
    """A regression model for any algorithm. Prefer ``Regressors.<algorithm>()``."""

    def __init__(self, algorithm: Union[str, Algorithm], task: str = "simple_regression") -> None:
        super().__init__(algorithm, task)


class Classifier(Model, BaseClassifier):
    """A classification model for any algorithm. Prefer ``Classifiers.<algorithm>()``."""

    def __init__(self, algorithm: Union[str, Algorithm], task: str = "binary") -> None:
        super().__init__(algorithm, task)


def _warn_deprecated_alias(name: str, factory: str) -> None:
    """Emit the deprecation warning for a 0.8.x model class alias."""
    warnings.warn(
        f"{name} is deprecated and will be removed in smallaxe 1.0. Use {factory} instead.",
        DeprecationWarning,
        stacklevel=3,
    )
