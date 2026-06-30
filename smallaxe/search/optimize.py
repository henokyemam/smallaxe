"""Hyperparameter optimization for smallaxe models.

This module provides :func:`run`, a small, predictable hyperparameter search
built on `hyperopt <https://github.com/hyperopt/hyperopt>`_. It optimizes the
hyperparameters of any smallaxe model (Random Forest, XGBoost, LightGBM,
CatBoost) using the model's own ``fit`` / ``validation_scores`` machinery, so
the search honors the same validation strategy, stratification, and seed as
ordinary training.

hyperopt is an optional dependency. Calling :func:`run` without it raises a
:class:`~smallaxe.exceptions.DependencyError` with an install hint.

Example:
    >>> from hyperopt import hp
    >>> from smallaxe.search import optimize
    >>> from smallaxe.training import Regressors
    >>>
    >>> model = Regressors.random_forest(seed=42)
    >>> result = optimize.run(
    ...     model,
    ...     df,
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
    >>> predictions = result.best_model.predict(df)
"""

from typing import Any, Dict, List, Optional

from pyspark.sql import DataFrame

from smallaxe.exceptions import DependencyError, ValidationError

# hyperopt is an optional dependency. Guard the import so the module remains
# importable (and the rest of smallaxe keeps working) when it is absent.
try:
    from hyperopt import STATUS_FAIL, STATUS_OK, Trials, fmin, hp, space_eval, tpe
    from hyperopt.early_stop import no_progress_loss
    from hyperopt.exceptions import AllTrialsFailed

    HYPEROPT_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised in environments without hyperopt
    HYPEROPT_AVAILABLE = False

HYPEROPT_INSTALL_HINT = "pip install hyperopt"

# Metric direction. Search always minimizes loss internally; for metrics where
# a higher value is better we minimize the negated score.
MAXIMIZE_METRICS = {
    "r2",
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "auc_roc",
    "auc_pr",
}
MINIMIZE_METRICS = {
    "mse",
    "rmse",
    "mae",
    "mape",
    "log_loss",
}
VALID_METRICS = MAXIMIZE_METRICS | MINIMIZE_METRICS


def _check_hyperopt_available() -> None:
    """Raise an actionable error when hyperopt is not installed."""
    if not HYPEROPT_AVAILABLE:
        raise DependencyError(package="hyperopt", install_command=HYPEROPT_INSTALL_HINT)


class SearchResult:
    """Container for the outcome of a hyperparameter search.

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
        """Number of trials actually evaluated (including failed ones)."""
        return len(self.trials_history)

    @property
    def n_successful_trials(self) -> int:
        """Number of trials that completed without error."""
        return sum(1 for t in self.trials_history if t.get("status") == "ok")

    def __repr__(self) -> str:
        score = "None" if self.best_score is None else f"{self.best_score:.6g}"
        return (
            f"SearchResult(metric='{self.metric}', best_score={score}, "
            f"best_params={self.best_params}, "
            f"trials={self.n_successful_trials}/{self.n_trials})"
        )


def _default_metric(model: Any) -> str:
    """Pick a sensible default metric for the model's task type.

    Note: precision/recall/f1_score are computed with binary semantics (they
    score only the positive class), so multiclass defaults to ``accuracy``,
    which is the only task-agnostic classification metric available.
    """
    if model.task_type == "regression":
        return "rmse"
    if getattr(model, "task", None) == "binary":
        return "auc_roc"
    return "accuracy"


def _normalize_space(param_space: Dict[str, Any]) -> Dict[str, Any]:
    """Convert convenience shorthands into hyperopt expressions.

    A list/tuple value is treated as a discrete set of choices and converted to
    ``hp.choice``. Any other value is assumed to already be a hyperopt
    expression and is passed through unchanged.
    """
    if not isinstance(param_space, dict) or not param_space:
        raise ValidationError(
            "param_space must be a non-empty dict mapping parameter names to "
            "hyperopt expressions (e.g. hp.quniform(...)) or lists of choices."
        )

    normalized: Dict[str, Any] = {}
    for name, spec in param_space.items():
        if isinstance(spec, (list, tuple)):
            if len(spec) == 0:
                raise ValidationError(f"Choice list for parameter '{name}' is empty.")
            normalized[name] = hp.choice(name, list(spec))
        else:
            normalized[name] = spec
    return normalized


def _cast_params(sampled: Dict[str, Any], type_reference: Dict[str, Any]) -> Dict[str, Any]:
    """Cast sampled values to the model's native parameter types.

    hyperopt's quantized distributions (``hp.quniform``) yield floats even for
    integer-valued parameters, so cast each sampled value to match the type of
    the model's default for that parameter.
    """
    cast: Dict[str, Any] = {}
    for name, value in sampled.items():
        default = type_reference.get(name)
        if value is None or default is None:
            cast[name] = value
        elif isinstance(default, bool):
            # bool is a subclass of int; handle before int to avoid misfire.
            cast[name] = bool(value)
        elif isinstance(default, int):
            cast[name] = int(round(value))
        elif isinstance(default, float):
            cast[name] = float(value)
        else:
            cast[name] = value
    return cast


def _extract_metric(
    validation_scores: Optional[Dict[str, Any]],
    metric: str,
    validation: str,
) -> float:
    """Pull a single scalar metric out of a model's ``validation_scores``."""
    if not validation_scores:
        raise ValidationError(
            "No validation scores were produced. Use validation='train_test' or "
            "'kfold' so the search has a score to optimize."
        )

    # k-fold aggregates each metric under a 'mean_<metric>' key.
    key = f"mean_{metric}" if validation == "kfold" else metric
    value = validation_scores.get(key)
    if value is None:
        available = sorted(k for k, v in validation_scores.items() if isinstance(v, (int, float)))
        raise ValidationError(
            f"Metric '{metric}' (looked up as '{key}') is not available in "
            f"validation scores. Available numeric scores: {available}. "
            "Note: auc_roc/auc_pr/log_loss are only produced for binary classification."
        )
    return float(value)


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
    stratified: Optional[bool] = None,
    seed: Optional[int] = None,
    refit: bool = True,
    verbose: bool = True,
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
        model: A smallaxe model instance (e.g. from ``Regressors.xgboost()``).
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
            ``'auc_roc'`` for binary classification, and ``'f1_score'`` for
            multiclass. Must be one of :data:`VALID_METRICS`.
        max_evals: Maximum number of hyperparameter evaluations. Default 20.
        validation: Validation strategy used to score each trial. One of
            ``'train_test'`` (default) or ``'kfold'``. ``'none'`` is rejected
            because it produces no score.
        n_folds: Number of folds when ``validation='kfold'``. Default 5.
        test_size: Test proportion when ``validation='train_test'``. Default 0.2.
        feature_cols: Feature columns. If None, inferred from numeric columns.
        exclude_cols: Columns to exclude from inferred features.
        stratified: Whether to stratify splits. If None, defaults to True for
            classification and False for regression (matching ``fit``).
        seed: Seed for hyperopt's sampler, for reproducible searches. Note that
            data-split reproducibility is governed by ``smallaxe.set_seed``.
        refit: If True (default), refit a final model on the full dataset with
            the best params and expose it as ``best_model``.
        verbose: If True, show hyperopt's progress bar.
        early_stopping: If True, stop early when no improvement is seen for
            ``early_stopping_rounds`` consecutive trials.
        early_stopping_rounds: Patience (in trials) for early stopping.

    Returns:
        SearchResult: best_params, best_score, best_model, and trials_history.

    Raises:
        DependencyError: If hyperopt is not installed.
        ValidationError: If arguments are invalid (bad metric, empty space,
            validation='none', etc.).
    """
    _check_hyperopt_available()

    # --- Validate arguments up front for clear, early failures. ---
    if not label_col:
        raise ValidationError("label_col is required.")

    if validation not in {"train_test", "kfold"}:
        raise ValidationError(
            f"validation must be 'train_test' or 'kfold' for optimization, got "
            f"'{validation}'. 'none' produces no score to optimize."
        )

    if max_evals < 1:
        raise ValidationError(f"max_evals must be at least 1, got {max_evals}.")

    metric = metric or _default_metric(model)
    if metric not in VALID_METRICS:
        raise ValidationError(
            f"Invalid metric '{metric}'. Supported metrics are: {sorted(VALID_METRICS)}."
        )

    space = _normalize_space(param_space)
    base_params = model.get_params()
    type_reference = model.default_params
    maximize = metric in MAXIMIZE_METRICS
    model_cls = type(model)
    task = model.task

    def _build_model(overrides: Dict[str, Any]) -> Any:
        """Create a fresh model with base params overridden by ``overrides``."""
        candidate = model_cls(task=task)
        candidate.set_param({**base_params, **overrides})
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
                stratified=stratified,
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

    try:
        fmin(**fmin_kwargs)
    except AllTrialsFailed:
        # Every trial errored; the friendly "all trials failed" guard below
        # surfaces the per-trial error messages from trials_history.
        pass

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
        best_model = _build_model(best_params)
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
