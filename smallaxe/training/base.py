"""The model classes: :class:`Model` and its two kinds, :class:`Regressor` and :class:`Classifier`.

:class:`BaseModel` composes the five mixins into the fit / validate / predict / evaluate
/ save machinery, written once for every algorithm; :class:`BaseRegressor` and
:class:`BaseClassifier` restrict it to one kind of task. :class:`Model` binds it to one
:class:`~smallaxe.training.algorithm.Algorithm` record, and :class:`Regressor` /
:class:`Classifier` are what the factory functions return. The ``Base*`` classes are
implementation classes: use ``Model``, ``Regressor`` and ``Classifier``.
"""

import math
import warnings
from contextlib import nullcontext
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple, Type, TypeVar, Union, cast

from pyspark.ml.linalg import VectorUDT
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import BooleanType, DoubleType, FloatType, NumericType

import smallaxe
from smallaxe import _checks, _warn
from smallaxe.exceptions import (
    ColumnNotFoundError,
    ModelNotFittedError,
    ValidationError,
)
from smallaxe.training import algorithm as _algorithm
from smallaxe.training.algorithm import ALL_TASKS, CLASSIFICATION_TASKS, REGRESSION_TASKS, Algorithm
from smallaxe.training.mixins import (
    MetadataMixin,
    ParamMixin,
    PersistenceMixin,
    SparkModelMixin,
    ValidationMixin,
)
from smallaxe.training.mixins.validation_mixin import _resolve_split_seed

__all__ = ["BaseModel", "BaseRegressor", "BaseClassifier", "Model", "Regressor", "Classifier"]

# The type of the model a method is called on, so ``regressor.fit(...)`` is a Regressor.
_ModelT = TypeVar("_ModelT", bound="BaseModel")

# fit's validation strategies and cache strategies; None caches only when fit reads the
# data more than once.
_VALIDATIONS = ("none", "train_test", "kfold")
_CACHE_STRATEGIES = (None, "none", "memory", "disk")


def _infer_feature_cols(
    dataframe: DataFrame, label_col: str, exclude_cols: List[str]
) -> Tuple[List[str], Dict[str, str]]:
    """Numeric and boolean columns as features; everything else is reported.

    Returns:
        ``(feature_cols, ignored)`` where ``ignored`` maps each column that is not
        a feature, other than the label and ``exclude_cols``, to its type name.
    """
    excluded = set(exclude_cols) | {label_col}
    feature_cols: List[str] = []
    ignored: Dict[str, str] = {}
    for field in dataframe.schema.fields:
        if field.name in excluded:
            continue
        if isinstance(field.dataType, (NumericType, BooleanType)):
            feature_cols.append(field.name)
        else:
            ignored[field.name] = field.dataType.simpleString()
    return feature_cols, ignored


def _warn_ignored_cols(
    feature_cols: List[str], ignored: Dict[str, str], stacklevel: Optional[int] = None
) -> None:
    """One ``UserWarning`` naming the columns feature inference left out.

    Args:
        feature_cols: The inferred feature columns.
        ignored: Column name -> type name of each column inference left out.
        stacklevel: Passed to ``warnings.warn``. By default the warning points at the
            first caller outside smallaxe.
    """
    if not ignored:
        return
    listed = ", ".join(f"{name} ({kind})" for name, kind in ignored.items())
    warnings.warn(
        f"Feature inference used {feature_cols} and ignored non-numeric columns: {listed}. "
        "Encode them, or list them in exclude_cols to silence this warning.",
        UserWarning,
        stacklevel=_warn.caller_stacklevel() if stacklevel is None else stacklevel,
    )


def _check_training_data(
    dataframe: DataFrame,
    label_col: str,
    feature_cols: List[str],
    accepts_missing: bool,
) -> Dict[str, Any]:
    """Count rows, missing features, and null labels in one pass, and reject bad data.

    Args:
        dataframe: Training DataFrame.
        label_col: Label column; null labels are always an error.
        feature_cols: Feature columns to inspect.
        accepts_missing: Whether the model treats NaN as missing. If not, null, NaN,
            and infinite features are errors and the user is told to add an Imputer.

    Returns:
        ``{"n_samples": int, "missing_counts": {col: int}}``. ``missing_counts``
        counts null and NaN per feature column.

    Raises:
        ValidationError: On null labels, or on missing or infinite features the
            model cannot accept.
    """
    types = {field.name: field.dataType for field in dataframe.schema.fields}

    def is_float(col: str) -> bool:
        return isinstance(types.get(col), (FloatType, DoubleType))

    aggregates = [F.count(F.lit(1)).alias("__n__")]
    aggregates.append(F.sum(F.col(label_col).isNull().cast("long")).alias("__label_null__"))
    for col in feature_cols:
        missing = F.col(col).isNull()
        if is_float(col):
            missing = missing | F.isnan(F.col(col))
        aggregates.append(F.sum(missing.cast("long")).alias(f"__missing__{col}"))
        if is_float(col) and not accepts_missing:
            infinite = F.abs(F.col(col)) == float("inf")
            aggregates.append(F.sum(infinite.cast("long")).alias(f"__inf__{col}"))

    row = dataframe.select(*aggregates).first()
    assert row is not None  # an aggregate without GROUP BY always yields one row
    n_samples = int(row["__n__"])
    missing_counts = {col: int(row[f"__missing__{col}"] or 0) for col in feature_cols}

    if row["__label_null__"]:
        raise ValidationError(
            f"Label column '{label_col}' has {int(row['__label_null__'])} null values. "
            "Drop or fill those rows before fitting."
        )

    if not accepts_missing:
        bad = [col for col, count in missing_counts.items() if count]
        if bad:
            detail = ", ".join(f"{col} ({missing_counts[col]})" for col in bad)
            raise ValidationError(
                f"Feature columns have null or NaN values: {detail}. This algorithm "
                "cannot treat them as missing. Fill them first, for example with "
                "smallaxe.preprocessing.Imputer."
            )
        infinite_cols = [col for col in feature_cols if is_float(col) and row[f"__inf__{col}"]]
        if infinite_cols:
            detail = ", ".join(f"{col} ({int(row[f'__inf__{col}'])})" for col in infinite_cols)
            raise ValidationError(
                f"Feature columns have infinite values: {detail}. Replace them before fitting."
            )

    return {"n_samples": n_samples, "missing_counts": missing_counts}


class BaseModel(
    ParamMixin,
    PersistenceMixin,
    ValidationMixin,
    MetadataMixin,
    SparkModelMixin,
):
    """Implementation class behind every model: fit, validation, predict, evaluate, save.

    Combines the parameter, persistence, validation, metadata and Spark mixins. It is
    abstract: :class:`Model` supplies the parameter table and the Spark estimator from
    an algorithm record. Use :class:`Regressor` / :class:`Classifier`.

    Args:
        task: The task: 'simple_regression', 'binary' or 'multiclass'.

    Raises:
        ValidationError: If ``task`` is not one of those.
    """

    # Split seed used when neither the model's ``seed`` nor the global seed is set.
    # ``optimize.run`` sets it on every trial so all trials score on one split; it
    # never seeds the estimator. Private: not a fit argument.
    _split_seed_fallback: Optional[int] = None

    def __init__(self, task: str) -> None:
        """Initialize the base model.

        Args:
            task: The task type for this model.

        Raises:
            ValidationError: If task is not a valid task type.
        """
        if task not in ALL_TASKS:
            raise ValidationError(f"Invalid task {task!r}. Valid tasks are: {sorted(ALL_TASKS)}")
        self._task = task
        self._params: Dict[str, Any] = {}
        self._spark_model = None
        self._features_col: str = self.FEATURES_COL
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
        if self._task in REGRESSION_TASKS:
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
        self: _ModelT,
        dataframe: DataFrame,
        label_col: str,
        feature_cols: Optional[List[str]] = None,
        exclude_cols: Optional[List[str]] = None,
        validation: str = "none",
        test_size: float = 0.2,
        n_folds: int = 5,
        cache_strategy: Optional[str] = None,
        fold_key: Optional[str] = None,
        refit: bool = True,
    ) -> _ModelT:
        """Fit the model to training data.

        Every split hashes the row content (or ``fold_key``) with the seed, which is
        independent of the label, so each class is split in the same proportion in
        expectation. The seed is the model's ``seed`` parameter, else the global
        ``smallaxe.set_seed`` value, else a random one (recorded in
        ``metadata["split_seed"]``).

        Args:
            dataframe: PySpark DataFrame containing training data.
            label_col: Name of the label/target column.
            feature_cols: List of feature column names. If None, uses all
                numeric and boolean columns except label_col and exclude_cols,
                and warns once about the columns it leaves out.
            exclude_cols: List of columns to exclude from features.
                Typically includes ID columns, timestamps, etc.
            validation: Validation strategy. Options:
                - 'none': No validation (default)
                - 'train_test': Single train/test split
                - 'kfold': K-fold cross-validation
            test_size: Proportion for test set when validation='train_test'.
                Default is 0.2.
            n_folds: Number of folds when validation='kfold'. Default is 5.
            cache_strategy: DataFrame caching for the duration of fit. Options:
                - None (default): persist with MEMORY_AND_DISK when fit reads the
                  data more than once (k-fold, or train_test with refit) and the
                  DataFrame is not already cached
                - 'none': Never cache
                - 'memory': Cache in memory (MEMORY_ONLY)
                - 'disk': Cache with disk spillover (MEMORY_AND_DISK)
                A DataFrame the caller already cached is left as it is.
            fold_key: Column whose value alone decides a row's fold or split, so all
                rows of one entity (a customer, a session) validate together. By
                default the whole row's content is hashed.
            refit: If True (default), the kept model is trained on all rows after
                validation scored it. If False, the kept model is the one from the
                last training split: the train split for 'train_test', the last
                fold for 'kfold'. Use False to save a training pass in a search.

        Returns:
            The fitted model itself, for chaining.

        Raises:
            ValidationError: If ``label_col`` is empty, ``validation``, ``cache_strategy``,
                ``test_size`` or ``n_folds`` is invalid, no feature column is given or
                inferred, a feature has missing values the algorithm cannot accept, the
                label has nulls, or a validation split or fold is empty.
            ColumnNotFoundError: If ``label_col``, a feature column or ``fold_key`` is not
                in ``dataframe``.
        """
        # Validate parameters
        self._validate_fit_params(
            dataframe=dataframe,
            label_col=label_col,
            feature_cols=feature_cols,
            exclude_cols=exclude_cols,
            validation=validation,
            cache_strategy=cache_strategy,
            test_size=test_size,
            n_folds=n_folds,
            fold_key=fold_key,
            refit=refit,
        )

        # Store exclude_cols
        self._exclude_cols = exclude_cols or []

        # Determine feature columns
        inferred = feature_cols is None
        ignored: Dict[str, str] = {}
        if feature_cols is None:
            feature_cols, ignored = _infer_feature_cols(dataframe, label_col, self._exclude_cols)
            _warn_ignored_cols(feature_cols, ignored)

        # Validate columns exist
        self._validate_columns(dataframe, label_col, feature_cols)

        # Store label column
        self._label_col = label_col

        # Seeds: the model's seed parameter, then the global seed, then the search's
        # shared split seed, then random.
        split_seed = _resolve_split_seed(self._seed_param(), self._split_seed_fallback)

        # Apply caching
        cached_df, persisted = self._apply_cache_strategy(
            dataframe, cache_strategy, validation, refit
        )

        try:
            # One pass over the data: row count, missing features, null labels.
            data_stats = _check_training_data(
                cached_df, label_col, feature_cols, self._accepts_missing()
            )
            data_stats.update(
                {
                    "feature_cols_inferred": inferred,
                    "ignored_cols": ignored,
                    "split_seed": split_seed if validation != "none" else None,
                    "refit": refit,
                }
            )

            # Perform training with validation
            if validation == "none":
                self._fit_no_validation(cached_df, label_col, feature_cols)
                data_stats["kept_model"] = "full"
                data_stats["n_training_rows"] = data_stats["n_samples"]
            elif validation == "train_test":
                self._fit_train_test(
                    cached_df,
                    label_col,
                    feature_cols,
                    test_size,
                    fold_key,
                    split_seed,
                    refit,
                    data_stats,
                )
            elif validation == "kfold":
                self._fit_kfold(
                    cached_df,
                    label_col,
                    feature_cols,
                    n_folds,
                    fold_key,
                    split_seed,
                    refit,
                    data_stats,
                )

            # Capture metadata
            self._capture_training_metadata(cached_df, label_col, feature_cols, data_stats)

            # Mark as fitted
            self._is_fitted = True

        finally:
            # Unpersist only what this call persisted
            if persisted:
                cached_df.unpersist()

        return self

    def _seed_param(self) -> Optional[int]:
        """The model's own ``seed`` parameter, if it has one and it is set.

        A model without a ``seed`` parameter has none. Errors reading the parameters
        propagate rather than silently unseeding the model.
        """
        return self.get_params().get("seed")

    def _validate_fit_params(
        self,
        dataframe: DataFrame,
        label_col: str,
        feature_cols: Optional[List[str]],
        exclude_cols: Optional[List[str]],
        validation: str,
        cache_strategy: Optional[str],
        test_size: float = 0.2,
        n_folds: int = 5,
        fold_key: Optional[str] = None,
        refit: bool = True,
    ) -> None:
        """Validate fit parameters before any Spark job runs.

        Args:
            dataframe: Training DataFrame.
            label_col: Name of the label column.
            feature_cols: List of feature columns.
            exclude_cols: List of columns to exclude.
            validation: Validation strategy.
            cache_strategy: Caching strategy.
            test_size: Test proportion, checked for 'train_test'.
            n_folds: Number of folds, checked for 'kfold'.
            fold_key: Column to hash instead of the whole row.
            refit: Whether to refit on all rows.

        Raises:
            ValidationError: If any parameter is invalid.
        """
        _checks.require_dataframe(dataframe)
        if not label_col or not isinstance(label_col, str):
            raise ValidationError(f"label_col must be a non-empty column name, got {label_col!r}.")
        _checks.column_names("feature_cols", feature_cols)
        _checks.column_names("exclude_cols", exclude_cols)
        if fold_key is not None and not isinstance(fold_key, str):
            raise ValidationError(f"fold_key must be a column name or None, got {fold_key!r}.")
        if not isinstance(refit, bool):
            raise ValidationError(f"refit must be True or False, got {refit!r}.")
        if validation == "train_test" and (
            isinstance(test_size, bool)
            or not isinstance(test_size, (int, float))
            or not 0 < test_size < 1
        ):
            raise ValidationError(f"test_size must be between 0 and 1, got {test_size!r}.")
        if validation == "kfold" and (
            isinstance(n_folds, bool) or not isinstance(n_folds, int) or n_folds < 2
        ):
            raise ValidationError(f"n_folds must be an integer of at least 2, got {n_folds!r}.")

        if validation not in _VALIDATIONS:
            raise ValidationError(
                f"Invalid validation strategy {validation!r}. "
                f"Valid options are: {sorted(_VALIDATIONS)}"
            )

        if cache_strategy not in _CACHE_STRATEGIES:
            raise ValidationError(
                f"Invalid cache_strategy '{cache_strategy}'. "
                "Valid options are: None, 'disk', 'memory', 'none'"
            )

    def _validate_columns(
        self,
        dataframe: DataFrame,
        label_col: str,
        feature_cols: List[str],
    ) -> None:
        """Validate that required columns exist in DataFrame.

        Args:
            dataframe: DataFrame to validate.
            label_col: Label column name.
            feature_cols: Feature column names.

        Raises:
            ColumnNotFoundError: If any column is not found.
            ValidationError: If no feature columns are provided.
        """
        available = set(dataframe.columns)

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
        types = {field.name: field.dataType for field in dataframe.schema.fields}
        for col in feature_cols:
            data_type = types.get(col)
            if data_type is not None and not isinstance(
                data_type, (NumericType, BooleanType, VectorUDT)
            ):
                raise ValidationError(
                    f"Feature column '{col}' has type {data_type.simpleString()}; features "
                    "must be numeric, boolean or ML vectors. Encode it first, for example "
                    "with smallaxe.preprocessing.Encoder."
                )

    def _accepts_missing(self) -> bool:
        """Whether the estimator treats NaN features as missing values."""
        return False

    def _effective_seed(self) -> Optional[int]:
        """The seed the estimator trains with: the model's parameter, else the global seed."""
        seed = self._seed_param()
        return smallaxe.get_seed() if seed is None else seed

    def _apply_cache_strategy(
        self,
        dataframe: DataFrame,
        cache_strategy: Optional[str],
        validation: str = "none",
        refit: bool = True,
    ) -> Tuple[DataFrame, bool]:
        """Persist ``dataframe`` for the duration of fit when that helps.

        Args:
            dataframe: DataFrame to cache.
            cache_strategy: See :meth:`fit`.
            validation: The validation strategy, which decides how often fit reads
                the data when ``cache_strategy`` is None.
            refit: Whether a final training pass follows validation.

        Returns:
            ``(dataframe, persisted)``: ``persisted`` is True only when this call persisted
            the DataFrame, so the caller unpersists exactly that.
        """
        from pyspark import StorageLevel

        if cache_strategy == "none":
            return dataframe, False
        current = dataframe.storageLevel
        # Compared by field: ``StorageLevel.NONE`` and ``StorageLevel.__eq__`` only exist
        # from Spark 3.4.
        if current.useMemory or current.useDisk or current.useOffHeap:
            # The caller cached it; leave it alone.
            return dataframe, False

        if cache_strategy is None:
            reads_twice = validation == "kfold" or (validation == "train_test" and refit)
            if not reads_twice:
                return dataframe, False
            level = StorageLevel.MEMORY_AND_DISK
        elif cache_strategy == "memory":
            level = StorageLevel.MEMORY_ONLY
        else:
            level = StorageLevel.MEMORY_AND_DISK

        try:
            return dataframe.persist(level), True
        except Exception as exc:  # noqa: BLE001 - caching is an optimisation only
            warnings.warn(
                f"Could not cache the training DataFrame ({type(exc).__name__}: {exc}). "
                "Continuing without a cache.",
                UserWarning,
                stacklevel=_warn.caller_stacklevel(),
            )
            return dataframe, False

    def _fit_no_validation(
        self,
        dataframe: DataFrame,
        label_col: str,
        feature_cols: List[str],
    ) -> None:
        """Fit model without validation.

        Args:
            dataframe: Training DataFrame.
            label_col: Label column name.
            feature_cols: Feature column names.
        """
        self._fit_spark_model(dataframe, label_col, feature_cols)
        self._validation_scores = None

    def _fit_train_test(
        self,
        dataframe: DataFrame,
        label_col: str,
        feature_cols: List[str],
        test_size: float,
        fold_key: Optional[str] = None,
        seed: Optional[int] = None,
        refit: bool = True,
        data_stats: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Fit model with train/test validation.

        Args:
            dataframe: Training DataFrame.
            label_col: Label column name.
            feature_cols: Feature column names.
            test_size: Proportion for test set.
            fold_key: Column to hash instead of the whole row.
            seed: Seed for the split.
            refit: Train the kept model on all rows after scoring.
            data_stats: Receives ``kept_model`` and ``n_training_rows``.
        """
        # Split data
        train_df, test_df = self._train_test_split(
            dataframe, test_size=test_size, seed=seed, fold_key=fold_key
        )

        # Fit on training data
        self._fit_spark_model(train_df, label_col, feature_cols)

        # Evaluate on test data
        self._validation_scores = _undefined_to_none(self._evaluate(test_df, label_col))
        self._validation_scores["validation_type"] = "train_test"
        self._validation_scores["test_size"] = test_size

        self._keep_model(dataframe, train_df, "train_split", refit, data_stats)

    def _keep_model(
        self,
        dataframe: DataFrame,
        split_df: DataFrame,
        split_name: str,
        refit: bool,
        data_stats: Optional[Dict[str, Any]],
    ) -> None:
        """Refit on all rows, or keep the last split's model, and record which."""
        if refit:
            self._fit_spark_model(dataframe, self._label_col or "", self._feature_cols)
        if data_stats is not None:
            data_stats["kept_model"] = "full" if refit else split_name
            data_stats["n_training_rows"] = (
                data_stats.get("n_samples") if refit else split_df.count()
            )

    def _fit_kfold(
        self,
        dataframe: DataFrame,
        label_col: str,
        feature_cols: List[str],
        n_folds: int,
        fold_key: Optional[str] = None,
        seed: Optional[int] = None,
        refit: bool = True,
        data_stats: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Fit model with k-fold cross-validation.

        Args:
            dataframe: Training DataFrame.
            label_col: Label column name.
            feature_cols: Feature column names.
            n_folds: Number of folds.
            fold_key: Column to hash instead of the whole row.
            seed: Seed for the fold assignment.
            refit: Train the kept model on all rows after scoring.
            data_stats: Receives ``kept_model`` and ``n_training_rows``.
        """
        fold_scores: List[Dict[str, Any]] = []
        train_df = dataframe

        # Iterate through folds
        for fold_idx, (train_df, val_df) in enumerate(
            self._kfold_split(dataframe, n_folds=n_folds, seed=seed, fold_key=fold_key)
        ):
            # Fit on this fold's training data
            self._fit_spark_model(train_df, label_col, feature_cols)

            # Evaluate on validation data
            scores = self._evaluate(val_df, label_col)
            scores["fold"] = fold_idx
            fold_scores.append(scores)

        self._keep_model(dataframe, train_df, "last_fold", refit, data_stats)

        # Aggregate scores
        self._validation_scores = self._aggregate_fold_scores(fold_scores, n_folds)

    def _aggregate_fold_scores(
        self,
        fold_scores: List[Dict[str, Any]],
        n_folds: int,
    ) -> Dict[str, Any]:
        """Aggregate scores across folds.

        A fold value is either a number, NaN (the metric is undefined on that fold,
        such as AUC-ROC on a fold with one class), or None (the model cannot produce
        the metric). Means and standard deviations use the numeric folds only. Folds
        where a metric was undefined are reported in one ``UserWarning``; a metric
        undefined on every fold gets a ``mean_`` of None. In the stored
        ``fold_scores`` undefined values are None.

        Args:
            fold_scores: List of score dictionaries from each fold.
            n_folds: Number of folds.

        Returns:
            Dictionary with aggregated scores.
        """
        if not fold_scores:
            return {"validation_type": "kfold", "n_folds": n_folds}

        import statistics

        # Get metric names (exclude 'fold' key)
        metric_names = [k for k in fold_scores[0].keys() if k != "fold"]

        result: Dict[str, Any] = {
            "validation_type": "kfold",
            "n_folds": n_folds,
            "fold_scores": [_undefined_to_none(s) for s in fold_scores],
        }

        undefined_counts: Dict[str, int] = {}
        for metric in metric_names:
            present = [s[metric] for s in fold_scores if s[metric] is not None]
            values = [v for v in present if not _is_undefined(v)]
            if len(values) < len(present):
                undefined_counts[metric] = len(present) - len(values)
            if values:
                result[f"mean_{metric}"] = statistics.mean(values)
                if len(values) > 1:
                    result[f"std_{metric}"] = statistics.stdev(values)
            elif present:
                result[f"mean_{metric}"] = None

        if undefined_counts:
            detail = ", ".join(
                f"{metric} in {count} of {len(fold_scores)} folds"
                for metric, count in undefined_counts.items()
            )
            warnings.warn(
                f"k-fold validation: a metric was undefined on some folds ({detail}), for "
                "example AUC on a fold with one class. Their means use the remaining folds; "
                "a metric undefined on every fold has mean None.",
                UserWarning,
                stacklevel=_warn.caller_stacklevel(),
            )

        return result

    def _evaluate(
        self,
        dataframe: DataFrame,
        label_col: str,
    ) -> Dict[str, Any]:
        """Evaluate model on a dataset.

        Args:
            dataframe: DataFrame to evaluate on.
            label_col: Label column name.

        Returns:
            Dictionary of evaluation metrics.
        """
        if self.task_type == "regression":
            predictions_df = self._predict_spark_model(dataframe, output_col="prediction")
            return self._compute_regression_metrics(predictions_df, label_col)
        else:
            return self._compute_classification_metrics_single_pass(dataframe, label_col)

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
        dataframe: DataFrame,
        label_col: str,
    ) -> Dict[str, Any]:
        """Compute classification metrics in a single transform pass.

        Args:
            dataframe: Original DataFrame to evaluate.
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
            multiclass_scores,
            precision,
            recall,
        )

        # Single transform pass produces predictions AND probabilities
        full_df, _ = self._transform_for_prediction(dataframe, self._feature_cols)

        if self._task == "multiclass":
            return multiclass_scores(full_df, label_col, self.PREDICTION_COL)

        # Basic metrics using class predictions
        metrics = {
            "accuracy": accuracy(full_df, label_col, self.PREDICTION_COL),
            "precision": precision(full_df, label_col, self.PREDICTION_COL),
            "recall": recall(full_df, label_col, self.PREDICTION_COL),
            "f1_score": f1_score(full_df, label_col, self.PREDICTION_COL),
        }

        # Probability-based metrics (only for binary classification). Two conditions
        # are handled here; any other failure propagates:
        # - a fitted model that writes no probability column cannot be scored on
        #   them, so they are unavailable (None);
        # - a model fitted on a training split holding one class (label 0 only) writes
        #   a 1-element probability vector (Spark's tree models size it by the classes
        #   seen). It has no positive-class probability, so the metrics are undefined
        #   (NaN, stored as None) rather than handed a null score the evaluator rejects.
        if self._task == "binary":
            if self.PROBABILITY_COL not in full_df.columns:
                metrics["auc_roc"] = None
                metrics["auc_pr"] = None
                metrics["log_loss"] = None
                return metrics

            from pyspark.ml.functions import vector_to_array

            probabilities = vector_to_array(F.col(self.PROBABILITY_COL))
            width = full_df.select(
                F.min(F.when(probabilities.isNotNull(), F.size(probabilities)))
            ).first()[0]
            if width is not None and width < 2:
                metrics["auc_roc"] = float("nan")
                metrics["auc_pr"] = float("nan")
                metrics["log_loss"] = float("nan")
                return metrics

            proba_df = full_df.withColumn("prob_positive", probabilities[1])

            metrics["auc_roc"] = auc_roc(proba_df, label_col, "prob_positive")
            metrics["auc_pr"] = auc_pr(proba_df, label_col, "prob_positive")
            metrics["log_loss"] = log_loss(proba_df, label_col, "prob_positive")

        return metrics

    def _capture_training_metadata(
        self,
        dataframe: DataFrame,
        label_col: str,
        feature_cols: List[str],
        data_stats: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Capture metadata from training.

        Args:
            dataframe: Training DataFrame.
            label_col: Label column name.
            feature_cols: Feature column names.
            data_stats: Output of :func:`_check_training_data`, so the row count
                is not computed again.
        """
        # Capture base metadata
        data_stats = data_stats or {}
        self._capture_metadata(
            dataframe, label_col, feature_cols, n_samples=data_stats.get("n_samples")
        )
        for key in (
            "missing_counts",
            "feature_cols_inferred",
            "ignored_cols",
            "split_seed",
            "refit",
            "kept_model",
            "n_training_rows",
        ):
            if key in data_stats:
                self._update_metadata(key, data_stats[key])
        self._update_metadata("seed", self._effective_seed())

        # Capture label statistics
        label_stats = self._capture_label_stats(dataframe, label_col, self.task_type)
        for key, value in label_stats.items():
            self._update_metadata(key, value)

        # Add task information
        self._update_metadata("task", self._task)
        self._update_metadata("task_type", self.task_type)

        # Add exclude_cols
        if self._exclude_cols:
            self._update_metadata("exclude_cols", self._exclude_cols)

    def predict(self, dataframe: DataFrame, output_col: str = "predict_label") -> DataFrame:
        """Generate predictions for input data.

        Args:
            dataframe: PySpark DataFrame with the feature columns the model was fitted on.
            output_col: Name for the prediction column. Default is 'predict_label'.

        Returns:
            The input columns, unchanged, plus ``output_col``; a classifier also adds
            its ``probability`` vector column unless ``dataframe`` already has one.

        Raises:
            ModelNotFittedError: If the model has not been fitted.
            ColumnNotFoundError: If a feature column is not in ``dataframe``.
            ValidationError: If ``output_col`` is already a column of ``dataframe``.
        """
        if not self._is_fitted:
            raise ModelNotFittedError("Model has not been fitted. Call fit() before predict().")
        _checks.require_dataframe(dataframe)

        return self._predict_spark_model(dataframe, output_col=output_col)

    def evaluate(self, dataframe: DataFrame, label_col: Optional[str] = None) -> Dict[str, Any]:
        """Score the fitted model on ``dataframe`` with the metrics validation computes.

        The keys are those of ``validation_scores`` for this task without the
        validation bookkeeping: ``mse``, ``rmse``, ``mae``, ``r2``, ``mape`` for
        regression; ``accuracy``, ``precision``, ``recall``, ``f1_score``, ``auc_roc``,
        ``auc_pr``, ``log_loss`` for binary classification; accuracy and the macro and
        weighted scores for multiclass. A metric that is undefined on ``dataframe`` (for example
        AUC-ROC when only one class is present) is None.

        Args:
            dataframe: PySpark DataFrame holding the feature columns and the label. Other
                columns are ignored.
            label_col: The label column. Defaults to the label the model was fitted with.

        Returns:
            Dictionary mapping metric names to values.

        Raises:
            ModelNotFittedError: If the model has not been fitted.
            ColumnNotFoundError: If the label or a feature column is not in ``dataframe``.
        """
        if not self._is_fitted:
            raise ModelNotFittedError("Model has not been fitted. Call fit() before evaluate().")
        _checks.require_dataframe(dataframe)
        label = label_col or self._label_col or ""
        available = list(dataframe.columns)
        for col in [label] + list(self._feature_cols):
            if col not in available:
                raise ColumnNotFoundError(column=col, available_columns=available)
        # Score only the columns the model needs, so a column of dataframe that shares a name
        # with one the model writes (``prediction``, ``probability``) cannot collide.
        needed = [label] + [col for col in self._feature_cols if col != label]
        return _undefined_to_none(self._evaluate(dataframe.select(*needed), label))

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
            "features_col": self._features_col,
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
        self._task = state["task"]
        self._params = dict(state.get("params") or {})
        self._feature_cols = state.get("feature_cols", [])
        # Artifacts saved before 0.10.0 read the vector from a column named "features".
        self._features_col = state.get("features_col") or "features"
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


class BaseRegressor(BaseModel):
    """Implementation class for regression models: accepts only regression tasks.

    Args:
        task: The regression task. Default is 'simple_regression'.

    Raises:
        ValidationError: If ``task`` is not a regression task.
    """

    def __init__(self, task: str = "simple_regression") -> None:
        """Initialize the base regressor.

        Args:
            task: The regression task type.

        Raises:
            ValidationError: If task is not a valid regression task.
        """
        if task not in REGRESSION_TASKS:
            raise ValidationError(
                f"Invalid regression task {task!r}. Valid tasks are: {sorted(REGRESSION_TASKS)}"
            )
        super().__init__(task)


class BaseClassifier(BaseModel):
    """Implementation class for classification models: classification tasks and probabilities.

    Args:
        task: The classification task. Default is 'binary'.

    Raises:
        ValidationError: If ``task`` is not 'binary' or 'multiclass'.
    """

    def __init__(self, task: str = "binary") -> None:
        """Initialize the base classifier.

        Args:
            task: The classification task type.

        Raises:
            ValidationError: If task is not a valid classification task.
        """
        if task not in CLASSIFICATION_TASKS:
            raise ValidationError(
                f"Invalid classification task {task!r}. "
                f"Valid tasks are: {sorted(CLASSIFICATION_TASKS)}"
            )
        super().__init__(task)

    def predict_proba(
        self,
        dataframe: DataFrame,
        output_col: str = "probability",
    ) -> DataFrame:
        """Generate class probabilities for input data.

        Args:
            dataframe: PySpark DataFrame with the feature columns the model was fitted on.
            output_col: Name for the probability vector column. Default is 'probability'.

        Returns:
            The input columns, unchanged, plus ``output_col``: a vector with one
            probability per class, in class-index order.

        Raises:
            ModelNotFittedError: If the model has not been fitted.
            ColumnNotFoundError: If a feature column is not in ``dataframe``.
            ValidationError: If ``output_col`` is already a column of ``dataframe``.
        """
        if not self._is_fitted:
            raise ModelNotFittedError(
                "Model has not been fitted. Call fit() before predict_proba()."
            )
        _checks.require_dataframe(dataframe)

        return self._predict_proba_spark_model(dataframe, output_col=output_col)


class Model(BaseModel):
    """A model bound to one :class:`~smallaxe.training.algorithm.Algorithm`.

    Everything algorithm-specific (parameter table, estimator and model classes,
    column wiring, task-fixed settings, importances, dependency, session setup) is
    read from the algorithm record, so fit, predict, save, load, and clone are
    written once here. Use its two kinds, :class:`Regressor` and :class:`Classifier`,
    usually through the ``regressors.<algorithm>()`` / ``classifiers.<algorithm>()``
    functions.

    Args:
        algorithm: Algorithm name (``"xgboost"``) or record.
        task: The task to train.

    Raises:
        DependencyError: If the algorithm's optional package is not installed.
        ValidationError: If ``algorithm`` is unknown or not a name or record, or the
            task is invalid or unsupported by the algorithm.

    Example:
        >>> model = Regressor("random_forest").set_params(n_estimators=50)
        >>> model.fit(dataframe, label_col="target").validation_scores
    """

    def __init__(self, algorithm: Union[str, Algorithm], task: str) -> None:
        if isinstance(algorithm, str):
            self._algorithm = _algorithm.get(algorithm)
        elif isinstance(algorithm, Algorithm):
            self._algorithm = algorithm
        else:
            raise ValidationError(
                "algorithm must be an algorithm name or an Algorithm record, got "
                f"{type(algorithm).__name__}. Known algorithms are: {_algorithm.names()}"
            )
        self._algorithm.require()
        super().__init__(task)
        if task not in self._algorithm.tasks:
            raise ValidationError(
                f"Algorithm '{self._algorithm.name}' does not support task '{task}'. "
                f"Supported tasks are: {sorted(self._algorithm.tasks)}"
            )

    @property
    def algorithm(self) -> Algorithm:
        """The algorithm record this model is bound to.

        Returns:
            The :class:`~smallaxe.training.algorithm.Algorithm` record.
        """
        return self._algorithm

    def _accepts_missing(self) -> bool:
        return self._algorithm.accepts_missing

    def _handle_invalid(self) -> str:
        """Hand nulls to an estimator that treats NaN as missing; reject them otherwise."""
        return "keep" if self._accepts_missing() else "error"

    # --- ParamMixin ---

    @property
    def params(self) -> Dict[str, str]:
        """The parameters this algorithm has for this task, with their descriptions.

        Returns:
            Parameter name -> description, in the algorithm's table order.
        """
        return {p.name: p.description for p in self._algorithm.visible_params(self._task)}

    @property
    def default_params(self) -> Dict[str, Any]:
        """The default value of each parameter this algorithm has for this task.

        Returns:
            Parameter name -> default value (None lets the estimator decide).
        """
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
        params = dict(self.get_params())
        if "seed" in params and params["seed"] is None:
            params["seed"] = smallaxe.get_seed()
        ctor, setters = _algorithm._estimator_kwargs(
            algorithm, self._task, params, self._column_names(label_col)
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

    def _fit_spark_model(
        self, dataframe: DataFrame, label_col: str, feature_cols: List[str]
    ) -> Any:
        algorithm = self._algorithm
        if algorithm.prepare is not None:
            algorithm.prepare()
        self._features_col = self.FEATURES_COL
        dataframe, _ = self._shield_user_columns(dataframe)
        df_with_features = self._assemble_features(dataframe, feature_cols)
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

    def clone(self: _ModelT) -> _ModelT:
        """Return an unfitted copy with the same algorithm, task, and parameter values.

        Returns:
            A new, unfitted model of the same class.
        """
        model = cast(Model, self)
        kind = Classifier if model.task_type == "classification" else Regressor
        clone = kind(model._algorithm, model._task)
        clone._params = dict(model._params)
        return cast(_ModelT, clone)

    # --- PersistenceMixin ---

    def _capture_training_metadata(
        self,
        dataframe: DataFrame,
        label_col: str,
        feature_cols: List[str],
        data_stats: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Capture training metadata, adding the algorithm package and its version.

        ``algorithm_package`` / ``algorithm_package_version`` sit next to
        ``smallaxe_version`` and ``spark_version``; both are None for an algorithm
        shipped with PySpark. Loading ignores them; they are there for diagnosis.
        """
        super()._capture_training_metadata(dataframe, label_col, feature_cols, data_stats)
        dependency = self._algorithm.dependency
        package = {
            "algorithm_package": dependency.package if dependency is not None else None,
            "algorithm_package_version": _algorithm._package_version(self._algorithm),
        }
        metadata: Dict[str, Any] = {}
        for key, value in self._metadata.items():
            metadata[key] = value
            if key == "smallaxe_version":
                metadata.update(package)
        metadata.update({key: value for key, value in package.items() if key not in metadata})
        self._metadata = metadata

    def _get_persistence_state(self) -> Dict[str, Any]:
        state = super()._get_persistence_state()
        state["algorithm"] = self._algorithm.name
        return state

    def _load_artifacts(self, path: str) -> None:
        if self._algorithm.prepare is not None:
            self._algorithm.prepare()
        self._load_spark_model(path, self._algorithm.model_class(self._task))
        # The loaded Spark model decides which vector column it reads. catboost-spark does
        # not persist featuresCol, so a reloaded model falls back to "features".
        get_features_col = getattr(self._spark_model, "getFeaturesCol", None)
        if callable(get_features_col):
            try:
                self._features_col = get_features_col() or self._features_col
            except Exception:  # noqa: BLE001 - keep the recorded name if the model cannot say
                pass

    @classmethod
    def load(cls, path: str) -> "Model":
        """Load a saved model, returning a :class:`Regressor` or :class:`Classifier`.

        This is the one load rule: :func:`smallaxe.load_model`, ``regressors.load`` and
        ``classifiers.load`` all come here. ``metadata.json`` is checked before anything
        is built: an artifact with the header must be a ``"model"`` one in a format this
        smallaxe reads; an artifact without it (smallaxe 0.10 and earlier) is accepted.
        ``algorithm`` names the algorithm; artifacts written by smallaxe 0.8.x record
        their class name instead, which is mapped to the algorithm. ``task`` is required
        and every saved parameter must be in the algorithm's table for that task;
        parameters the artifact leaves out take their defaults. ``Regressor.load`` and
        ``Classifier.load`` also check the kind.

        Args:
            path: The artifact directory (local, ``dbfs:/``, ``s3://``, ``hdfs://``, ...).

        Returns:
            The fitted model.

        Raises:
            ValidationError: If the path or its ``metadata.json`` does not exist, it holds
                another kind of artifact or was written by a newer smallaxe, its
                ``algorithm``, ``task`` or ``params`` are missing or unknown, or the saved
                model is not the kind this class loads.
            DependencyError: If the saved model's algorithm is not installed.
        """
        state = cls._read_saved_state(path)
        saved_class = state.pop("__class__", None)
        state.pop("__module__", None)

        name = state.get("algorithm")
        if name is None:
            if saved_class not in _algorithm._LEGACY_CLASS_NAMES:
                raise ValidationError(
                    "Model metadata does not contain 'algorithm'. "
                    "This may be an older model format or corrupted metadata."
                )
            name = _algorithm._LEGACY_CLASS_NAMES[saved_class]
        algorithm = _algorithm.get(name)
        task = _saved_task(path, state, algorithm)
        state["params"] = _saved_params(path, state, algorithm, task)

        kind: Type[Model] = Classifier if task in CLASSIFICATION_TASKS else Regressor
        for expected in (Regressor, Classifier):
            if issubclass(cls, expected) and kind is not expected:
                saved = "classifier" if kind is Classifier else "regressor"
                raise ValidationError(
                    f"Cannot load model: {cls.__name__}.load expects a "
                    f"{expected.__name__.lower()}, but {path!r} holds a {name!r} {saved} "
                    f"(task {task!r}). Use smallaxe.load_model(), which loads either."
                )
        algorithm.require()

        instance: Model = kind.__new__(kind)
        instance._algorithm = algorithm
        instance._spark_model = None
        instance._set_persistence_state(state)
        instance._load_artifacts(path)
        if task == "multiclass" and instance._validation_scores:
            _warn_if_class1_scores(instance._validation_scores)
        return instance


def _saved_task(path: str, state: Dict[str, Any], algorithm: Algorithm) -> str:
    """The saved ``task``, checked against what ``algorithm`` trains.

    Every smallaxe release has written ``task``, so a missing one means the artifact
    is corrupted.

    Raises:
        ValidationError: If ``task`` is missing or not a task of ``algorithm``.
    """
    task = state.get("task")
    if task is None:
        raise ValidationError(
            f"Cannot load model: {path!r} has no 'task' in its metadata.json; the artifact "
            "is corrupted."
        )
    if not isinstance(task, str) or task not in algorithm.tasks:
        raise ValidationError(
            f"Cannot load model: {path!r} records task {task!r}, which {algorithm.name!r} "
            f"does not train (its tasks are {sorted(algorithm.tasks)})."
        )
    return task


def _saved_params(
    path: str, state: Dict[str, Any], algorithm: Algorithm, task: str
) -> Dict[str, Any]:
    """The saved ``params``, checked against ``algorithm``'s table for ``task``.

    Raises:
        ValidationError: If ``params`` is not a mapping or names a parameter the table
            does not have for ``task``.
    """
    params = state.get("params")
    if params is None:
        return {}
    if not isinstance(params, dict):
        raise ValidationError(
            f"Cannot load model: 'params' in {path!r} must be a JSON object, got "
            f"{type(params).__name__}; the artifact is corrupted."
        )
    known = {param.name for param in algorithm.visible_params(task)}
    unknown = sorted(set(params) - known)
    if unknown:
        raise ValidationError(
            f"Cannot load model: {path!r} sets parameter(s) {unknown}, which "
            f"{algorithm.name!r} does not have for task {task!r} (it has {sorted(known)}). "
            "The artifact was written by a newer smallaxe or is corrupted."
        )
    return dict(params)


class Regressor(Model, BaseRegressor):
    """A regression model for any algorithm. Prefer ``regressors.<algorithm>()``.

    Args:
        algorithm: Algorithm name (``"xgboost"``) or record.
        task: The regression task. Default is 'simple_regression'.

    Raises:
        DependencyError: If the algorithm's optional package is not installed.
        ValidationError: If ``algorithm`` is unknown, or ``task`` is not a regression
            task.

    Example:
        >>> model = Regressor("xgboost").set_params(max_depth=4)
    """

    def __init__(self, algorithm: Union[str, Algorithm], task: str = "simple_regression") -> None:
        super().__init__(algorithm, task)


class Classifier(Model, BaseClassifier):
    """A classification model for any algorithm. Prefer ``classifiers.<algorithm>()``.

    Args:
        algorithm: Algorithm name (``"xgboost"``) or record.
        task: The classification task: 'binary' (default) or 'multiclass'.

    Raises:
        DependencyError: If the algorithm's optional package is not installed.
        ValidationError: If ``algorithm`` is unknown, or ``task`` is not 'binary' or
            'multiclass'.

    Example:
        >>> model = Classifier("random_forest", task="multiclass")
    """

    def __init__(self, algorithm: Union[str, Algorithm], task: str = "binary") -> None:
        super().__init__(algorithm, task)


def _is_undefined(value: Any) -> bool:
    """Whether a metric value is NaN, the metrics' marker for "undefined on this data"."""
    return isinstance(value, float) and math.isnan(value)


def _undefined_to_none(scores: Dict[str, Any]) -> Dict[str, Any]:
    """A copy of ``scores`` with undefined (NaN) values as None.

    Stored scores use None for any value that is not a number: NaN is not valid JSON,
    and None already marks metrics a model cannot produce.
    """
    return {key: None if _is_undefined(value) else value for key, value in scores.items()}


def _warn_if_class1_scores(validation_scores: Dict[str, Any]) -> None:
    """Warn when multiclass scores were saved before macro averaging (smallaxe <= 0.8.x)."""
    if any(key.endswith("precision_macro") for key in validation_scores):
        return
    warnings.warn(
        "This multiclass model was saved by a smallaxe version that scored validation "
        "precision, recall, and f1_score for class 1 only (one-vs-rest), not as macro "
        "averages. Refit it to get macro and weighted scores.",
        UserWarning,
        stacklevel=_warn.caller_stacklevel(),
    )
