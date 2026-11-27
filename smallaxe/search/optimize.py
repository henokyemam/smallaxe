"""Hyperparameter optimization for smallaxe models.

This module provides :func:`run`, a small, predictable hyperparameter search
built on `hyperopt <https://github.com/hyperopt/hyperopt>`_. It optimizes the
hyperparameters of any smallaxe model (Random Forest, XGBoost, LightGBM,
CatBoost) using the model's own ``fit`` / ``validation_scores`` machinery, so
the search honors the same validation strategy, folds, and seed as ordinary training.

Example:
    >>> from hyperopt import hp
    >>> from smallaxe.search import optimize
    >>> from smallaxe.training import regressors
    >>>
    >>> model = regressors.random_forest(seed=42)
    >>> result = optimize.run(
    ...     model,
    ...     dataframe,
    ...     label_col="price",
    ...     param_space={
    ...         "n_estimators": hp.quniform("n_estimators", 20, 200, 20),
    ...         "max_depth": hp.quniform("max_depth", 3, 12, 1),
    ...     },
    ...     metric="rmse",
    ...     validation="train_test",
    ...     max_evals=25,
    ... )
    >>> result.best_params
    {'n_estimators': 120, 'max_depth': 8, ...}
    >>> predictions = result.best_model.predict(dataframe)
"""

import math
import numbers
from typing import Any, Dict, List, Optional

from hyperopt import STATUS_FAIL, STATUS_OK, Trials, fmin, hp, space_eval, tpe
from hyperopt.early_stop import no_progress_loss
from hyperopt.exceptions import AllTrialsFailed
from hyperopt.pyll import Apply
from pyspark.sql import DataFrame

import smallaxe
from smallaxe import _checks
from smallaxe.exceptions import ValidationError
from smallaxe.metrics import TASK_METRICS
from smallaxe.metrics.classification import _MULTICLASS_AVERAGED_METRICS
from smallaxe.training.base import (
    Model,
    _check_training_data,
    _infer_feature_cols,
    _warn_ignored_cols,
)
from smallaxe.training.mixins.validation_mixin import _resolve_split_seed

__all__ = ["SearchResult", "run"]

# Metric direction. Search always minimizes loss internally; for metrics where
# a higher value is better we minimize the negated score.
_MAXIMIZE_METRICS = {
    "r2",
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "auc_roc",
    "auc_pr",
} | _MULTICLASS_AVERAGED_METRICS
_MINIMIZE_METRICS = {
    "mse",
    "rmse",
    "mae",
    "mape",
    "log_loss",
}
_VALID_METRICS = _MAXIMIZE_METRICS | _MINIMIZE_METRICS

# The metric optimised when none is given, per task. AutomatedTraining uses the same.
_DEFAULT_METRICS: Dict[str, str] = {
    "simple_regression": "rmse",
    "binary": "auc_roc",
    "multiclass": "accuracy",
}


class SearchResult:
    """Container for the outcome of a hyperparameter search.

    :func:`run` builds it; each argument is kept as the attribute of the same name.

    Args:
        best_params: The best hyperparameters found.
        best_score: The validation score of ``best_params``.
        best_model: The model refitted with ``best_params``, or None.
        trials_history: One dict per evaluation.
        metric: The optimized metric name.
        validation: The validation strategy used during the search.
        max_evals: The number of evaluations requested.

    Attributes:
        best_params: The best hyperparameters found, cast to the model's
            native parameter types (ints stay ints, floats stay floats).
        best_score: The validation score of ``best_params`` for the chosen
            metric (in the metric's natural orientation, not the internal loss).
        best_model: A fitted model configured with ``best_params``. ``None`` when
            ``refit=False`` was passed to :func:`run`. Ready for ``predict``.
        trials_history: One entry per evaluation, each a dict with ``params``,
            ``score``, ``loss``, and ``status`` keys. Failed trials carry an
            ``error`` message and a ``None`` score.
        metric: The optimized metric name.
        validation: The validation strategy used during the search.
        max_evals: The number of evaluations requested.
    """

    def __init__(
        self,
        best_params: Dict[str, Any],
        best_score: Optional[float],
        best_model: Any,
        trials_history: List[Dict[str, Any]],
        metric: str,
        validation: str,
        max_evals: int,
    ) -> None:
        self.best_params = best_params
        self.best_score = best_score
        self.best_model = best_model
        self.trials_history = trials_history
        self.metric = metric
        self.validation = validation
        self.max_evals = max_evals

    @property
    def n_trials(self) -> int:
        """Number of trials actually evaluated.

        Returns:
            The trial count, failed trials included.
        """
        return len(self.trials_history)

    @property
    def n_successful_trials(self) -> int:
        """Number of trials that completed without error.

        Returns:
            The count of trials whose status is ``"ok"``.
        """
        return sum(1 for t in self.trials_history if t.get("status") == "ok")

    def __repr__(self) -> str:
        score = "None" if self.best_score is None else f"{self.best_score:.6g}"
        return (
            f"SearchResult(metric='{self.metric}', best_score={score}, "
            f"best_params={self.best_params}, "
            f"trials={self.n_successful_trials}/{self.n_trials})"
        )


def _default_metric(model: Any) -> str:
    """Pick RMSE for regression, ROC AUC for binary, and accuracy for multiclass."""
    return _DEFAULT_METRICS[model.task]


def _resolve_metric(metric: Optional[str], task: str) -> str:
    """``metric``, or the task's default when it is None, checked against the task.

    Shared by :func:`run` and ``AutomatedTraining``, so both accept the same metrics.

    Args:
        metric: The metric asked for, or None.
        task: The task being trained.

    Returns:
        The metric name.

    Raises:
        ValidationError: If ``metric`` is not a metric a search can optimise, or not one
            the task's validation produces.
    """
    if metric is None:
        metric = _DEFAULT_METRICS[task]
    if not isinstance(metric, str) or metric not in _VALID_METRICS:
        raise ValidationError(
            f"Invalid metric '{metric}'. Supported metrics are: {sorted(_VALID_METRICS)}."
        )
    # A metric the task's validation never produces would fail every trial only after fitting it.
    task_metrics = TASK_METRICS[task]
    if metric not in task_metrics:
        raise ValidationError(
            f"Metric '{metric}' is not available for task '{task}'. "
            f"Supported metrics for this task are: {sorted(task_metrics)}."
        )
    return metric


def _higher_is_better(metric: str) -> bool:
    """Whether a larger value of ``metric`` is better (``r2``, ``auc_roc``) or worse (``rmse``)."""
    return metric in _MAXIMIZE_METRICS


def _normalize_space(param_space: Dict[str, Any], type_reference: Dict[str, Any]) -> Dict[str, Any]:
    """Convert convenience shorthands into hyperopt expressions, checking fixed values.

    A list/tuple value is treated as a discrete set of choices and converted to
    ``hp.choice``. A hyperopt expression (``hp.quniform(...)``) is passed through
    unchanged. Any other value is a constant: hyperopt hands it to every trial as is.
    Each choice and constant must cast to the parameter's type (see :func:`_cast_params`).

    Raises:
        ValidationError: If ``param_space`` is not a non-empty dict with string keys, a
            choice list is empty, or a choice or constant does not fit its parameter.
    """
    if not isinstance(param_space, dict) or not param_space:
        raise ValidationError(
            "param_space must be a non-empty dict mapping parameter names to "
            "hyperopt expressions (e.g. hp.quniform(...)) or lists of choices."
        )

    normalized: Dict[str, Any] = {}
    for name, spec in param_space.items():
        if not isinstance(name, str):
            raise ValidationError(f"param_space keys must be parameter names, got {name!r}.")
        if isinstance(spec, (list, tuple)):
            if len(spec) == 0:
                raise ValidationError(f"Choice list for parameter '{name}' is empty.")
            for choice in spec:
                _cast_params({name: choice}, type_reference)
            normalized[name] = hp.choice(name, list(spec))
        else:
            if not isinstance(spec, Apply):
                _cast_params({name: spec}, type_reference)
            normalized[name] = spec
    return normalized


def _cast_params(sampled: Dict[str, Any], type_reference: Dict[str, Any]) -> Dict[str, Any]:
    """Cast sampled values to the model's native parameter types.

    hyperopt's quantized distributions (``hp.quniform``) yield floats even for
    integer-valued parameters, so cast each sampled value to match the type of
    the model's default for that parameter.

    Raises:
        ValidationError: If a value for an int or float parameter is not a number.
    """
    cast: Dict[str, Any] = {}
    for name, value in sampled.items():
        default = type_reference.get(name)
        if value is None or default is None:
            cast[name] = value
        elif isinstance(default, bool):
            # bool is a subclass of int; handle before int to avoid misfire.
            cast[name] = bool(value)
        elif isinstance(default, (int, float)):
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise ValidationError(
                    f"param_space value {value!r} for '{name}' must be a number, like the "
                    f"parameter's default {default!r}."
                )
            cast[name] = int(round(value)) if isinstance(default, int) else float(value)
        else:
            cast[name] = value
    return cast


def _score_key(metric: str, validation: str) -> str:
    """The ``validation_scores`` key of ``metric``: k-fold stores it as ``mean_<metric>``."""
    return f"mean_{metric}" if validation == "kfold" else metric


def _lookup_metric(
    validation_scores: Optional[Dict[str, Any]],
    metric: str,
    validation: str,
) -> Optional[float]:
    """A single scalar metric of a model's ``validation_scores``, or None when undefined.

    Undefined means present but None or NaN: the metric could not be computed on the
    validation rows, for example AUC on a split with a single class.

    Raises:
        ValidationError: If there are no scores, or the metric is missing from them.
    """
    if not validation_scores:
        raise ValidationError(
            "No validation scores were produced. Use validation='train_test' or "
            "'kfold' so the search has a score to optimize."
        )

    key = _score_key(metric, validation)
    if key not in validation_scores:
        available = sorted(k for k, v in validation_scores.items() if isinstance(v, (int, float)))
        raise ValidationError(
            f"Metric '{metric}' (looked up as '{key}') is not available in "
            f"validation scores. Available numeric scores: {available}. "
            "Note: auc_roc/auc_pr/log_loss are only produced for binary classification."
        )
    value = validation_scores[key]
    if value is None or math.isnan(float(value)):
        return None
    return float(value)


def _extract_metric(
    validation_scores: Optional[Dict[str, Any]],
    metric: str,
    validation: str,
) -> float:
    """Pull a single scalar metric out of a model's ``validation_scores``.

    Raises:
        ValidationError: If the metric is missing, or present but undefined (None or
            NaN) on this trial's validation data, so the trial fails rather than
            scoring.
    """
    score = _lookup_metric(validation_scores, metric, validation)
    if score is None:
        where = "every fold" if validation == "kfold" else "the test split"
        raise ValidationError(
            f"Metric '{metric}' is undefined on {where} of this trial (looked up as "
            f"'{_score_key(metric, validation)}'), for example AUC on validation data with "
            "a single class. The trial is not scored."
        )
    return score


def run(
    model: Any,
    dataframe: DataFrame,
    label_col: str,
    param_space: Dict[str, Any],
    metric: Optional[str] = None,
    max_evals: int = 20,
    validation: str = "train_test",
    n_folds: int = 5,
    test_size: float = 0.2,
    feature_cols: Optional[List[str]] = None,
    exclude_cols: Optional[List[str]] = None,
    fold_key: Optional[str] = None,
    seed: Optional[int] = None,
    refit: bool = True,
    verbose: Optional[bool] = None,
    early_stopping: bool = False,
    early_stopping_rounds: int = 10,
) -> SearchResult:
    """Search for the best hyperparameters of a smallaxe model.

    For each evaluation hyperopt samples a point from ``param_space``, a fresh
    model of the same type is configured with the base model's fixed parameters
    overridden by the sampled values, fit with the requested ``validation``
    strategy, and scored on ``metric``. The search minimizes loss internally,
    negating the score for metrics where higher is better.

    Args:
        model: A smallaxe model instance (e.g. from ``regressors.xgboost()``).
            Its current parameters are used as the fixed baseline; only the
            parameters present in ``param_space`` are searched. Used as a
            template only — it is not mutated.
        dataframe: PySpark DataFrame with features and the label column.
        label_col: Name of the label/target column.
        param_space: Dict mapping parameter names to hyperopt expressions
            (e.g. ``hp.quniform``, ``hp.loguniform``) or lists of discrete
            choices (auto-wrapped in ``hp.choice``). Parameter names must be
            valid for the model.
        metric: Metric to optimize. Defaults to ``'rmse'`` for regression,
            ``'auc_roc'`` for binary classification, and ``'accuracy'`` for
            multiclass. Multiclass precision/recall/f1_score use macro averages;
            explicit ``*_macro`` and ``*_weighted`` names are also supported.
            Must be a metric of the model's task in
            :data:`smallaxe.metrics.TASK_METRICS` (e.g. ``auc_roc`` is binary-only,
            ``*_macro`` multiclass-only).
        max_evals: Maximum number of hyperparameter evaluations. Default 20.
        validation: Validation strategy used to score each trial. One of
            ``'train_test'`` (default) or ``'kfold'``. ``'none'`` is rejected
            because it produces no score.
        n_folds: Number of folds when ``validation='kfold'``. Default 5.
        test_size: Test proportion when ``validation='train_test'``. Default 0.2.
        feature_cols: Feature columns. If None, every numeric and boolean column
            except ``label_col`` and ``exclude_cols``; the columns left out are
            reported in one ``UserWarning``.
        exclude_cols: Columns to exclude from inferred features.
        fold_key: Column whose value alone decides a row's fold or split, as in
            ``fit``. Every trial scores on the same folds either way.
        seed: Seed for hyperopt's sampler and for the data splits. Defaults to
            the global ``smallaxe.set_seed`` value; with neither, the sampler and
            the estimators are unseeded and one random split seed is drawn for the
            whole search. Every trial scores on the same folds either way.
        refit: If True (default), refit a final model on the full dataset with
            the best params and expose it as ``best_model``.
        verbose: If True, show hyperopt's progress bar. Defaults to the global
            verbosity: quiet hides it.
        early_stopping: If True, stop early when no improvement is seen for
            ``early_stopping_rounds`` consecutive trials.
        early_stopping_rounds: Patience (in trials) for early stopping.

    Returns:
        SearchResult: best_params, best_score, best_model, and trials_history.

    Raises:
        ValidationError: If arguments are invalid: ``model`` is not a smallaxe model,
            ``dataframe`` is not a DataFrame, a bad metric or one the model's task does
            not produce, an empty or malformed space (a key that is not a parameter
            name, a choice or constant that is not a number for a numeric parameter),
            validation='none', a ``test_size``, ``n_folds`` or ``fold_key`` that ``fit``
            would reject, a ``max_evals`` or ``early_stopping_rounds`` below 1, a ``seed``
            that is not an int, a ``refit``, ``verbose`` or ``early_stopping`` that is not
            a bool, a label that is not numeric, or every trial failed.
        ColumnNotFoundError: If ``label_col`` or a feature column is missing.
    """
    # --- Validate arguments up front for clear, early failures. ---
    if not isinstance(model, Model):
        raise ValidationError(
            "model must be a smallaxe model (from regressors.<algorithm>() or "
            f"classifiers.<algorithm>()), got {type(model).__name__}."
        )
    _checks.require_dataframe(dataframe)
    feature_cols = _checks.column_names("feature_cols", feature_cols)
    exclude_cols = _checks.column_names("exclude_cols", exclude_cols)
    if not label_col or not isinstance(label_col, str):
        raise ValidationError(f"label_col must be a non-empty column name, got {label_col!r}.")

    if validation not in ("train_test", "kfold"):
        raise ValidationError(
            f"validation must be 'train_test' or 'kfold' for optimization, got "
            f"{validation!r}. 'none' produces no score to optimize."
        )
    # test_size, n_folds, fold_key and refit are checked as fit checks them, once.
    model._validate_fit_params(
        dataframe=dataframe,
        label_col=label_col,
        feature_cols=feature_cols,
        exclude_cols=exclude_cols,
        validation=validation,
        cache_strategy=None,
        test_size=test_size,
        n_folds=n_folds,
        fold_key=fold_key,
        refit=refit,
    )
    for name, flag in (("verbose", verbose), ("early_stopping", early_stopping)):
        if flag is not None and not isinstance(flag, bool):
            raise ValidationError(f"{name} must be True or False, got {flag!r}.")

    if isinstance(max_evals, bool) or not isinstance(max_evals, int) or max_evals < 1:
        raise ValidationError(f"max_evals must be an integer of at least 1, got {max_evals!r}.")
    if (
        isinstance(early_stopping_rounds, bool)
        or not isinstance(early_stopping_rounds, int)
        or early_stopping_rounds < 1
    ):
        raise ValidationError(
            f"early_stopping_rounds must be an integer of at least 1, got {early_stopping_rounds!r}."
        )
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise ValidationError(f"seed must be an integer or None, got {type(seed).__name__}.")

    metric = _resolve_metric(metric, model.task)

    type_reference = model.default_params
    space = _normalize_space(param_space, type_reference)
    unknown = sorted(set(param_space) - set(model.params))
    if unknown:
        raise ValidationError(
            f"param_space names {unknown}, which {model.algorithm.name!r} does not have for "
            f"task {model.task!r}. Its parameters are: {sorted(model.params)}."
        )
    maximize = _higher_is_better(metric)

    # Infer features once and check the data once, so bad data fails here rather
    # than in every trial, and the inference warning is given at most once.
    if feature_cols is None:
        feature_cols, ignored = _infer_feature_cols(dataframe, label_col, exclude_cols or [])
        _warn_ignored_cols(feature_cols, ignored)
    model._validate_columns(dataframe, label_col, feature_cols)
    _check_training_data(dataframe, label_col, feature_cols, model._accepts_missing())

    if seed is None:
        seed = smallaxe.get_seed()
    if verbose is None:
        verbose = smallaxe.get_verbosity() != "quiet"
    # Every trial must score on the same split. Unseeded, each fit would draw its own
    # random split seed, so draw one here for all trials (or take the template's private
    # fallback: AutomatedTraining sets it so every algorithm of a run shares one split).
    # It only decides the split: unseeded estimators stay unseeded.
    split_seed = _resolve_split_seed(seed, model._split_seed_fallback)

    def _build_model(overrides: Dict[str, Any], trial: bool = True) -> Any:
        """Create a fresh model with base params overridden by ``overrides``.

        Only trial models get the search's split seed; ``best_model`` is handed to the
        caller and must not carry it into later fits.
        """
        candidate = model.clone()
        if trial:
            candidate._split_seed_fallback = split_seed
        candidate.set_params(**overrides)
        if (
            seed is not None
            and "seed" in type_reference
            and candidate.get_params().get("seed") is None
        ):
            candidate.set_params(seed=seed)
        return candidate

    def objective(sampled: Dict[str, Any]) -> Dict[str, Any]:
        params = _cast_params(sampled, type_reference)
        try:
            trial_model = _build_model(params)
            trial_model.fit(
                dataframe,
                label_col=label_col,
                feature_cols=feature_cols,
                exclude_cols=exclude_cols,
                validation=validation,
                n_folds=n_folds,
                test_size=test_size,
                fold_key=fold_key,
                cache_strategy="none",
                refit=False,
            )
            score = _extract_metric(trial_model.validation_scores, metric, validation)
        except Exception as exc:  # noqa: BLE001 - a bad config must not kill the search
            return {
                "status": STATUS_FAIL,
                "loss": float("inf"),
                "score": None,
                "params": params,
                "error": f"{type(exc).__name__}: {exc}",
            }

        loss = -score if maximize else score
        return {"status": STATUS_OK, "loss": loss, "score": score, "params": params}

    trials = Trials()
    rstate = None
    if seed is not None:
        import numpy as np

        rstate = np.random.default_rng(seed)

    fmin_kwargs: Dict[str, Any] = {
        "fn": objective,
        "space": space,
        "algo": tpe.suggest,
        "max_evals": max_evals,
        "trials": trials,
        "show_progressbar": verbose,
        "rstate": rstate,
    }
    if early_stopping:
        fmin_kwargs["early_stop_fn"] = no_progress_loss(early_stopping_rounds)

    # Persist once for the whole search; each trial runs with cache_strategy="none".
    # "kfold" here is deliberate, not the search's validation: the two arguments say
    # whether one fit reads the data more than once, and a search reads it once per
    # trial whatever the validation, so it always qualifies (unless already cached).
    dataframe, persisted = model._apply_cache_strategy(dataframe, None, "kfold", True)
    try:
        fmin(**fmin_kwargs)
    except AllTrialsFailed:
        # Every trial errored; the friendly "all trials failed" guard below
        # surfaces the per-trial error messages from trials_history.
        pass
    finally:
        if persisted:
            dataframe.unpersist()

    # --- Collect trial history in submission order. ---
    trials_history: List[Dict[str, Any]] = []
    for trial in trials.trials:
        result = trial.get("result", {})
        trials_history.append(
            {
                "params": result.get("params"),
                "score": result.get("score"),
                "loss": result.get("loss"),
                "status": "ok" if result.get("status") == STATUS_OK else "fail",
                "error": result.get("error"),
            }
        )

    # --- Resolve the best point. ---
    successful = [t for t in trials_history if t["status"] == "ok"]
    if not successful:
        raise ValidationError(
            "All trials failed; no best parameters could be determined. "
            "Inspect trials_history for per-trial error messages. "
            f"First error: {trials_history[0].get('error') if trials_history else 'n/a'}"
        )

    best_trial = trials.best_trial
    best_score = best_trial["result"]["score"]
    # space_eval resolves hp.choice indices back to concrete values.
    best_params = _cast_params(space_eval(space, trials.argmin), type_reference)

    best_model = None
    if refit:
        best_model = _build_model(best_params, trial=False)
        best_model.fit(
            dataframe,
            label_col=label_col,
            feature_cols=feature_cols,
            exclude_cols=exclude_cols,
            validation="none",
        )

    return SearchResult(
        best_params=best_params,
        best_score=best_score,
        best_model=best_model,
        trials_history=trials_history,
        metric=metric,
        validation=validation,
        max_evals=max_evals,
    )
