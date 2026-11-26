"""AutomatedTraining: train every available algorithm on one split and keep the best.

Built only on the public training API: the factories in ``smallaxe.training.regressors``
/ ``classifiers``, ``Model.fit`` / ``validation_scores`` / ``predict`` / ``evaluate``, and
``smallaxe.search.optimize.run``. It adds no training logic of its own: the metric
orientation, the score extraction, the shared split seed, the argument checks and the
caching rule are those of ``optimize.run`` and ``Model.fit``.
"""

import copy
import math
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from pyspark.sql import DataFrame

import smallaxe
from smallaxe import _artifact, _checks, _fs, _warn
from smallaxe.exceptions import (
    ColumnNotFoundError,
    DependencyError,
    ModelNotFittedError,
    SmallaxeError,
    ValidationError,
)
from smallaxe.metrics import TASK_METRICS
from smallaxe.search import optimize
from smallaxe.search.optimize import (
    _higher_is_better,
    _lookup_metric,
    _normalize_space,
    _resolve_metric,
    _score_key,
)
from smallaxe.training import algorithm as _algorithm
from smallaxe.training import classifiers, regressors
from smallaxe.training.base import (
    Classifier,
    Model,
    _check_training_data,
    _infer_feature_cols,
    _warn_ignored_cols,
)
from smallaxe.training.mixins.validation_mixin import _resolve_split_seed

__all__ = ["AutomatedTraining"]

# The tasks AutomatedTraining supports in this release.
_SUPPORTED_TASKS: Tuple[str, ...] = ("simple_regression", "binary")

# fit's validation strategies: "none" gives no score to compare.
_SCORED_VALIDATIONS: Tuple[str, ...] = ("train_test", "kfold")

# Row order of the comparison: successful fits first (best first), then failures,
# then algorithms whose package is not installed.
_OK, _FAILED, _SKIPPED = "ok", "failed", "skipped"
_STATUS_ORDER: Dict[str, int] = {_OK: 0, _FAILED: 1, _SKIPPED: 2}


def _check_task(task: Any) -> str:
    """``task`` if AutomatedTraining supports it.

    Raises:
        ValidationError: If ``task`` is 'multiclass' or not one of the supported tasks.
    """
    if task == "multiclass":
        raise ValidationError(
            "AutomatedTraining supports the tasks 'simple_regression' and 'binary' in this "
            "release; 'multiclass' is not supported yet. Train a multiclass model with "
            "classifiers.<algorithm>(task='multiclass') and optimize.run."
        )
    if not isinstance(task, str) or task not in _SUPPORTED_TASKS:
        raise ValidationError(
            f"Invalid task {task!r}. AutomatedTraining supports: {list(_SUPPORTED_TASKS)}."
        )
    return task


def _known_algorithms(task: str) -> List[str]:
    """Every algorithm name of the task's factory module, installed or not."""
    names = regressors.algorithms() if task == "simple_regression" else classifiers.algorithms()
    return [name for name in names if task in _algorithm.get(name).tasks]


def _check_algorithms(task: str, algorithms: Any) -> List[str]:
    """The requested algorithm names, or every algorithm of ``task`` when None.

    Raises:
        ValidationError: If ``algorithms`` is not a list of known, distinct names, or is
            empty.
    """
    known = _known_algorithms(task)
    if algorithms is None:
        return known
    if not isinstance(algorithms, (list, tuple)) or not all(
        isinstance(name, str) for name in algorithms
    ):
        raise ValidationError(
            f"algorithms must be a list of algorithm names, got {algorithms!r}. "
            f"Known algorithms are: {known}."
        )
    if not algorithms:
        raise ValidationError(
            f"algorithms must name at least one algorithm. Known algorithms are: {known}."
        )
    unknown = [name for name in algorithms if name not in known]
    if unknown:
        raise ValidationError(f"Unknown algorithm(s) {unknown}. Known algorithms are: {known}.")
    duplicates = sorted({name for name in algorithms if list(algorithms).count(name) > 1})
    if duplicates:
        raise ValidationError(f"algorithms lists {duplicates} more than once.")
    return list(algorithms)


def _check_per_algorithm(
    name: str, value: Any, task: str, algorithms: List[str]
) -> Dict[str, Dict[str, Any]]:
    """``params`` / ``param_spaces``: a dict of parameter dicts keyed by requested algorithm.

    Parameter names are checked against each algorithm's table for ``task`` without
    importing any optional package. ``seed`` is rejected: every algorithm of a run must
    split on the same seed, which is AutomatedTraining's own ``seed``.

    Raises:
        ValidationError: If ``value`` is malformed, names an algorithm that is not
            requested, or a parameter the algorithm does not have.
    """
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValidationError(
            f"{name} must be a dict mapping algorithm names to parameter dicts, got "
            f"{type(value).__name__}."
        )
    checked: Dict[str, Dict[str, Any]] = {}
    for algorithm_name, entries in value.items():
        if algorithm_name not in algorithms:
            raise ValidationError(
                f"{name} has an entry for {algorithm_name!r}, which is not one of the "
                f"requested algorithms {algorithms}."
            )
        if not isinstance(entries, dict) or not all(isinstance(key, str) for key in entries):
            raise ValidationError(
                f"{name}[{algorithm_name!r}] must be a dict mapping parameter names to "
                f"values, got {entries!r}."
            )
        if name == "param_spaces" and not entries:
            raise ValidationError(
                f"param_spaces[{algorithm_name!r}] is empty; leave the algorithm out of "
                "param_spaces to fit it once with its params."
            )
        if "seed" in entries:
            raise ValidationError(
                f"{name}[{algorithm_name!r}] sets 'seed'. Pass seed= to AutomatedTraining "
                "instead: every algorithm of a run is scored on the split that seed gives."
            )
        table = sorted(param.name for param in _algorithm.get(algorithm_name).visible_params(task))
        unknown = sorted(set(entries) - set(table))
        if unknown:
            raise ValidationError(
                f"{name}[{algorithm_name!r}] names {unknown}, which {algorithm_name!r} does "
                f"not have for task {task!r}. Its parameters are: {table}."
            )
        checked[algorithm_name] = dict(entries)
    return checked


def _is_nan(value: Any) -> bool:
    return isinstance(value, float) and math.isnan(value)


def _progress(message: str) -> None:
    """Print a progress line unless the global verbosity is 'quiet'.

    The same rule decides whether ``optimize.run`` shows hyperopt's progress bar.
    """
    if smallaxe.get_verbosity() != "quiet":
        print(message, flush=True)


class AutomatedTraining:
    """Train every available algorithm on the same rows and keep the best one.

    ``fit`` builds one model per algorithm from its factory, validates each on exactly
    the same validation rows (one split seed for the whole run), compares them on one
    metric and keeps the best, refitted on every row. An algorithm listed in
    ``param_spaces`` is tuned with :func:`smallaxe.search.optimize.run`; the others are
    fitted once with their ``params``. An algorithm whose optional package is not
    installed is skipped with a warning; one that fails while training is reported and
    the run continues.

    AutomatedTraining has no ``save`` / ``load``: persist the winner with
    ``auto.best_model.save(path)`` and the table with ``auto.comparison.to_csv(...)``.
    Like ``Model.fit`` it trains on numeric and boolean features: encode categorical
    columns first (for example with :class:`smallaxe.preprocessing.Encoder`).

    Args:
        task: ``"simple_regression"`` or ``"binary"``.
        algorithms: Algorithm names to train, from ``regressors.algorithms()`` /
            ``classifiers.algorithms()``. Defaults to every algorithm of the task.
        metric: The metric to compare on. Defaults to ``"rmse"`` for regression and
            ``"auc_roc"`` for binary, as in ``optimize.run``. It must be a metric of the
            task in :data:`smallaxe.metrics.TASK_METRICS`. ``rmse``, ``mse``, ``mae``,
            ``mape`` and ``log_loss`` are minimised; the others are maximised.
        params: Fixed parameters per algorithm name, such as
            ``{"xgboost": {"max_depth": 4}}``. With ``param_spaces`` they are the fixed
            baseline of the search.
        param_spaces: A hyperopt search space per algorithm name, as ``optimize.run``
            takes it. Algorithms without one are fitted once.
        max_evals: Evaluations per algorithm that has a search space. Default 20.
        seed: Seeds the shared split, every estimator and every search. Defaults to the
            global ``smallaxe.set_seed`` value; with neither, one random split seed is
            drawn for the whole run and the estimators are unseeded.

    Raises:
        ValidationError: If ``task`` is not supported, an algorithm is unknown or listed
            twice, ``algorithms`` is empty, ``metric`` is not a metric of the task,
            ``params`` / ``param_spaces`` name an algorithm that is not requested, a
            parameter the algorithm does not have, or ``seed``, ``max_evals`` is not an
            integer of at least 1, or ``seed`` is not an integer or None.

    Example:
        >>> from smallaxe.auto import AutomatedTraining
        >>> auto = AutomatedTraining(task="simple_regression", seed=42)
        >>> auto.fit(dataframe, label_col="price")
        >>> auto.comparison[["algorithm", "status", "rmse"]]
        >>> auto.best_algorithm, auto.best_score
        >>> predictions = auto.predict(new_dataframe)
        >>> auto.best_model.save("/models/price")
    """

    def __init__(
        self,
        task: str,
        algorithms: Optional[List[str]] = None,
        metric: Optional[str] = None,
        params: Optional[Dict[str, Dict[str, Any]]] = None,
        param_spaces: Optional[Dict[str, Dict[str, Any]]] = None,
        max_evals: int = 20,
        seed: Optional[int] = None,
    ) -> None:
        self._task = _check_task(task)
        self._algorithms = _check_algorithms(self._task, algorithms)
        self._metric = _resolve_metric(metric, self._task)
        self._params = _check_per_algorithm("params", params, self._task, self._algorithms)
        self._param_spaces = _check_per_algorithm(
            "param_spaces", param_spaces, self._task, self._algorithms
        )
        if isinstance(max_evals, bool) or not isinstance(max_evals, int) or max_evals < 1:
            raise ValidationError(f"max_evals must be an integer of at least 1, got {max_evals!r}.")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise ValidationError(f"seed must be an integer or None, got {type(seed).__name__}.")
        self._max_evals = max_evals
        self._seed = seed

        self._models: Dict[str, Model] = {}
        self._rows: List[Dict[str, Any]] = []
        self._best_algorithm: Optional[str] = None
        self._best_score: Optional[float] = None
        self._split_seed: Optional[int] = None

    # --- Always available ---------------------------------------------------------

    @property
    def task(self) -> str:
        """The task: ``"simple_regression"`` or ``"binary"``.

        Returns:
            The task name.
        """
        return self._task

    @property
    def metric(self) -> str:
        """The metric the algorithms are compared on.

        Returns:
            The metric name.
        """
        return self._metric

    @property
    def algorithms(self) -> List[str]:
        """The algorithms this run trains, in order (the requested list or the default).

        Returns:
            A new list of algorithm names.
        """
        return list(self._algorithms)

    # --- After fit ----------------------------------------------------------------

    def _require_fitted(self, what: str) -> None:
        if self._best_algorithm is None:
            raise ModelNotFittedError(
                f"AutomatedTraining has not been fitted. Call fit() before {what}."
            )

    @property
    def best_model(self) -> Model:
        """The winning model, fitted on every row. ``save`` it to keep it.

        Returns:
            A fitted :class:`~smallaxe.training.Regressor` or
            :class:`~smallaxe.training.Classifier`.

        Raises:
            ModelNotFittedError: Before ``fit``.
        """
        self._require_fitted("best_model")
        assert self._best_algorithm is not None
        return self._models[self._best_algorithm]

    @property
    def best_algorithm(self) -> str:
        """The name of the winning algorithm.

        Returns:
            An algorithm name, such as ``"xgboost"``.

        Raises:
            ModelNotFittedError: Before ``fit``.
        """
        self._require_fitted("best_algorithm")
        assert self._best_algorithm is not None
        return self._best_algorithm

    @property
    def best_score(self) -> float:
        """The winner's validation score on :attr:`metric`, in the metric's own orientation.

        Returns:
            The score (for k-fold, the mean over folds).

        Raises:
            ModelNotFittedError: Before ``fit``.
        """
        self._require_fitted("best_score")
        assert self._best_score is not None
        return self._best_score

    @property
    def models(self) -> Dict[str, Model]:
        """Every successfully fitted model, by algorithm name, each refitted on every row.

        Returns:
            A new dict; the models themselves are not copied.

        Raises:
            ModelNotFittedError: Before ``fit``.
        """
        self._require_fitted("models")
        return dict(self._models)

    @property
    def comparison(self) -> "pd.DataFrame":
        """One row per requested algorithm, best first, as a pandas DataFrame.

        Columns, in order: ``algorithm``; ``status`` (``"ok"``, ``"failed"`` or
        ``"skipped"``); the compared metric; every other validation metric of the task,
        sorted by name; ``fit_seconds``; ``params`` (the parameters used: the fixed
        ``params`` merged with the values a search found, plus ``seed`` when the run is
        seeded); ``error`` (None when ok). Rows that are ok come first, best first, then
        failed, then skipped rows. A score that is undefined or was not computed is NaN:
        failed and skipped rows have none, a row whose metric was undefined on the
        validation rows (AUC on one class) ranks last among the ok rows, and a searched
        algorithm has only the compared metric (``optimize.run`` scores trials on it
        alone). Save it with ``comparison.to_csv(path)``.

        Returns:
            A new DataFrame on each access.

        Raises:
            ModelNotFittedError: Before ``fit``.
        """
        self._require_fitted("comparison")
        columns = ["algorithm", "status", self._metric]
        columns += sorted(TASK_METRICS[self._task] - {self._metric})
        columns += ["fit_seconds", "params", "error"]
        table = pd.DataFrame(copy.deepcopy(self._rows), columns=columns)
        # An object column keeps None for "no error"; a string dtype (pandas >= 3 infers
        # one) would turn it into NaN.
        table["error"] = pd.Series([row["error"] for row in self._rows], dtype=object)
        return table

    # --- Training -----------------------------------------------------------------

    def fit(
        self,
        dataframe: DataFrame,
        label_col: str,
        feature_cols: Optional[List[str]] = None,
        exclude_cols: Optional[List[str]] = None,
        *,
        validation: str = "train_test",
        test_size: float = 0.2,
        n_folds: int = 5,
        fold_key: Optional[str] = None,
        cache_strategy: Optional[str] = None,
    ) -> "AutomatedTraining":
        """Train every requested algorithm, compare them on :attr:`metric`, keep the best.

        The arguments are checked once, before any algorithm trains, with the checks
        ``Model.fit`` uses. Every algorithm is validated on the same rows: one split seed
        is resolved for the run (``seed``, else the global seed, else one random draw)
        and every model and search uses it. Each model is then refitted on every row
        (``refit=True``).

        Args:
            dataframe: PySpark DataFrame with the feature columns and the label.
            label_col: Name of the label column; it must be numeric (0/1 for binary).
            feature_cols: Feature columns. If None, every numeric and boolean column
                except ``label_col`` and ``exclude_cols``; the columns left out are
                reported in one ``UserWarning``.
            exclude_cols: Columns to leave out of the inferred features.
            validation: ``"train_test"`` (default) or ``"kfold"``. ``"none"`` is
                rejected: it gives no score to compare.
            test_size: Test proportion for ``"train_test"``. Default 0.2.
            n_folds: Number of folds for ``"kfold"``. Default 5.
            fold_key: Column whose value alone decides a row's split or fold, as in
                ``Model.fit``.
            cache_strategy: As in ``Model.fit``, applied once for the whole run: None
                (default) persists the data for the run unless it is already cached,
                ``"none"`` never caches, ``"memory"`` / ``"disk"`` choose the level.

        Returns:
            This AutomatedTraining, fitted.

        Raises:
            ValidationError: If ``dataframe`` is not a DataFrame, ``validation`` is not
                ``"train_test"`` or ``"kfold"``, another argument is one ``Model.fit``
                rejects, the label is not numeric or has nulls, a ``params`` value or
                ``param_spaces`` entry is invalid for its algorithm, every algorithm
                failed or was skipped (the message carries each error), or the metric
                was undefined for every algorithm that trained.
            ColumnNotFoundError: If ``label_col``, a feature column or ``fold_key`` is
                missing.
            DependencyError: If none of the requested algorithms is installed.
        """
        # A fit that raises must not leave an earlier run's results looking current.
        self._rows = []
        self._models = {}
        self._best_algorithm = None
        self._best_score = None
        self._split_seed = None

        _checks.require_dataframe(dataframe)
        if not isinstance(validation, str) or validation not in _SCORED_VALIDATIONS:
            raise ValidationError(
                f"validation must be 'train_test' or 'kfold' for AutomatedTraining, got "
                f"{validation!r}. 'none' gives no score to compare the algorithms on."
            )
        feature_cols = _checks.column_names("feature_cols", feature_cols)
        exclude_cols = _checks.column_names("exclude_cols", exclude_cols)

        available, skipped = self._probe()
        if not available:
            self._warn_skipped(skipped)
            raise DependencyError(
                f"None of the requested algorithms {self._algorithms} is installed. "
                + " ".join(f"{name}: {reason}." for name, reason in skipped.items())
            )
        split_seed = _resolve_split_seed(self._seed)
        models = {name: self._build(name, split_seed) for name in available}
        template = models[available[0]]

        # The checks Model.fit runs, once for the whole run and before anything trains.
        template._validate_fit_params(
            dataframe=dataframe,
            label_col=label_col,
            feature_cols=feature_cols,
            exclude_cols=exclude_cols,
            validation=validation,
            cache_strategy=cache_strategy,
            test_size=test_size,
            n_folds=n_folds,
            fold_key=fold_key,
            refit=True,
        )
        if fold_key is not None and fold_key not in dataframe.columns:
            raise ColumnNotFoundError(column=fold_key, available_columns=list(dataframe.columns))
        if feature_cols is None:
            feature_cols, ignored = _infer_feature_cols(dataframe, label_col, exclude_cols or [])
            _warn_ignored_cols(feature_cols, ignored)
        template._validate_columns(dataframe, label_col, feature_cols)
        for name, space in self._param_spaces.items():
            if name in models:
                _normalize_space(space, models[name].default_params)

        self._warn_skipped(skipped)
        rows: Dict[str, Dict[str, Any]] = {}
        fitted: Dict[str, Model] = {}
        for name in skipped:
            rows[name] = self._row(name, _SKIPPED, error=skipped[name])

        # Persist once for the whole run: it reads the data at least twice per algorithm
        # (validation, then the refit on every row), so the rule fit applies always holds.
        cached, persisted = template._apply_cache_strategy(
            dataframe, cache_strategy, validation, True
        )
        try:
            # Null labels fail here, once, rather than in every algorithm. Missing feature
            # values are left to each algorithm: some accept them.
            _check_training_data(cached, label_col, feature_cols, True)
            for index, name in enumerate(available, 1):
                _progress(f"AutomatedTraining: training {name} ({index}/{len(available)})...")
                rows[name], model = self._train(
                    name,
                    models[name],
                    cached,
                    label_col,
                    feature_cols,
                    exclude_cols,
                    validation,
                    test_size,
                    n_folds,
                    fold_key,
                )
                if model is not None:
                    fitted[name] = model
        finally:
            for model in models.values():
                model._split_seed_fallback = None
            if persisted:
                cached.unpersist()

        ordered = self._rank([rows[name] for name in self._algorithms])
        best = ordered[0]
        if best["status"] != _OK:
            detail = "; ".join(f"{row['algorithm']}: {row['error']}" for row in ordered)
            raise ValidationError(
                f"AutomatedTraining could not train any of {self._algorithms}. {detail}"
            )
        if _is_nan(best[self._metric]):
            raise ValidationError(
                f"Metric '{self._metric}' was undefined on the validation rows for every "
                "algorithm that trained (for example AUC when the validation rows hold a "
                "single class), so no algorithm can be chosen. Use more data, another "
                "test_size or seed, or another metric."
            )

        self._rows = ordered
        self._models = fitted
        self._best_algorithm = best["algorithm"]
        self._best_score = float(best[self._metric])
        self._split_seed = split_seed
        _progress(
            f"AutomatedTraining: best is {self._best_algorithm} "
            f"({self._metric}={self._best_score:.6g})"
        )
        return self

    def _probe(self) -> Tuple[List[str], Dict[str, str]]:
        """Split the requested algorithms into installed ones and skipped ones.

        Uses each record's availability check, the one ``available_models()`` reports.

        Returns:
            ``(available names, {skipped name: reason naming the install hint})``, in
            request order.
        """
        available: List[str] = []
        skipped: Dict[str, str] = {}
        for name in self._algorithms:
            record = _algorithm.get(name)
            if record.available():
                available.append(name)
                continue
            dependency = record.dependency
            package = dependency.package if dependency is not None else name
            reason = f"{package} is not installed or cannot be imported"
            if dependency is not None:
                reason += f"; install it with: {dependency.install_hint}"
            skipped[name] = reason
        return available, skipped

    def _warn_skipped(self, skipped: Dict[str, str]) -> None:
        """One ``UserWarning`` per skipped algorithm, attributed to the caller's line."""
        for name, reason in skipped.items():
            warnings.warn(
                f"AutomatedTraining skips {name!r}: {reason}.",
                UserWarning,
                stacklevel=_warn.caller_stacklevel(),
            )

    def _build(self, name: str, split_seed: int) -> Model:
        """A fresh model of ``name`` from its factory, with the fixed params and run seed.

        Raises:
            ValidationError: If a fixed parameter value has the wrong type.
        """
        fixed = self._params.get(name, {})
        if self._task == "simple_regression":
            model: Model = getattr(regressors, name)(**fixed)
        else:
            model = getattr(classifiers, name)(task=self._task, **fixed)
        if self._seed is not None:
            model.set_params(seed=self._seed)
        # Split on the run's seed even when nothing else seeds the split (private, as in
        # optimize.run); reset after the run so the model does not carry it.
        model._split_seed_fallback = split_seed
        return model

    def _used_params(self, name: str, searched: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        used = dict(self._params.get(name, {}))
        if self._seed is not None:
            used["seed"] = self._seed
        used.update(searched or {})
        return used

    def _row(
        self,
        name: str,
        status: str,
        score: Optional[float] = None,
        scores: Optional[Dict[str, Any]] = None,
        validation: str = "train_test",
        seconds: Optional[float] = None,
        params: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        """One comparison row; a score that is undefined or not computed is NaN."""
        row: Dict[str, Any] = {"algorithm": name, "status": status}
        row[self._metric] = float("nan") if score is None else float(score)
        for other in sorted(TASK_METRICS[self._task] - {self._metric}):
            value = (scores or {}).get(_score_key(other, validation))
            row[other] = float("nan") if value is None else float(value)
        row["fit_seconds"] = float("nan") if seconds is None else seconds
        row["params"] = self._used_params(name) if params is None else params
        row["error"] = error
        return row

    def _train(
        self,
        name: str,
        model: Model,
        dataframe: DataFrame,
        label_col: str,
        feature_cols: List[str],
        exclude_cols: Optional[List[str]],
        validation: str,
        test_size: float,
        n_folds: int,
        fold_key: Optional[str],
    ) -> Tuple[Dict[str, Any], Optional[Model]]:
        """Fit (or search) one algorithm; a training failure becomes a ``failed`` row.

        Returns:
            ``(comparison row, fitted model or None)``.
        """
        failures = (SmallaxeError,) + _fs.jvm_error_types()
        start = time.perf_counter()
        try:
            space = self._param_spaces.get(name)
            if space is not None:
                result = optimize.run(
                    model,
                    dataframe,
                    label_col,
                    space,
                    metric=self._metric,
                    max_evals=self._max_evals,
                    validation=validation,
                    n_folds=n_folds,
                    test_size=test_size,
                    feature_cols=feature_cols,
                    exclude_cols=exclude_cols,
                    fold_key=fold_key,
                    seed=self._seed,
                    refit=True,
                )
                fitted: Model = result.best_model
                score: Optional[float] = result.best_score
                scores: Dict[str, Any] = {}
                used = self._used_params(name, result.best_params)
            else:
                model.fit(
                    dataframe,
                    label_col,
                    feature_cols=feature_cols,
                    exclude_cols=exclude_cols,
                    validation=validation,
                    test_size=test_size,
                    n_folds=n_folds,
                    cache_strategy="none",
                    fold_key=fold_key,
                    refit=True,
                )
                fitted = model
                scores = model.validation_scores or {}
                score = _lookup_metric(scores, self._metric, validation)
                used = self._used_params(name)
        except failures as exc:
            seconds = time.perf_counter() - start
            if isinstance(exc, SmallaxeError):
                summary = f"{type(exc).__name__}: {exc}"
            else:
                summary = f"{type(exc).__name__}: {_artifact.jvm_error_summary(exc)}"
            warnings.warn(
                f"AutomatedTraining: {name!r} failed and is left out of the comparison: "
                f"{summary}",
                UserWarning,
                stacklevel=_warn.caller_stacklevel(),
            )
            _progress(f"AutomatedTraining: {name} failed ({summary})")
            return self._row(name, _FAILED, seconds=seconds, error=summary), None

        seconds = time.perf_counter() - start
        shown = "undefined" if score is None else f"{score:.6g}"
        _progress(f"AutomatedTraining: {name} {self._metric}={shown} ({seconds:.1f}s)")
        row = self._row(
            name,
            _OK,
            score=score,
            scores=scores,
            validation=validation,
            seconds=seconds,
            params=used,
        )
        return row, fitted

    def _rank(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rows best first: ok by score (undefined last), then failed, then skipped."""
        maximize = _higher_is_better(self._metric)

        def key(row: Dict[str, Any]) -> Tuple[int, int, float]:
            score = row[self._metric]
            if row["status"] != _OK or _is_nan(score):
                return (_STATUS_ORDER[row["status"]], 1, 0.0)
            return (_STATUS_ORDER[row["status"]], 0, -score if maximize else score)

        return sorted(rows, key=key)

    # --- Using the best model -----------------------------------------------------

    def predict(self, dataframe: DataFrame, output_col: str = "predict_label") -> DataFrame:
        """Predict with :attr:`best_model`.

        Args:
            dataframe: PySpark DataFrame with the feature columns.
            output_col: Name for the prediction column. Default 'predict_label'.

        Returns:
            ``dataframe`` plus ``output_col`` (see ``Model.predict``).

        Raises:
            ModelNotFittedError: Before ``fit``.
            ValidationError: As ``Model.predict`` raises it.
            ColumnNotFoundError: If a feature column is missing.
        """
        self._require_fitted("predict()")
        return self.best_model.predict(dataframe, output_col=output_col)

    def predict_proba(self, dataframe: DataFrame, output_col: str = "probability") -> DataFrame:
        """Class probabilities from :attr:`best_model` (binary only).

        Args:
            dataframe: PySpark DataFrame with the feature columns.
            output_col: Name for the probability vector column. Default 'probability'.

        Returns:
            ``dataframe`` plus ``output_col`` (see ``Classifier.predict_proba``).

        Raises:
            ValidationError: If the task is not 'binary', or as
                ``Classifier.predict_proba`` raises it.
            ModelNotFittedError: Before ``fit``.
            ColumnNotFoundError: If a feature column is missing.
        """
        if self._task != "binary":
            raise ValidationError(
                f"predict_proba is only available for the 'binary' task, not {self._task!r}."
            )
        self._require_fitted("predict_proba()")
        model = self.best_model
        assert isinstance(model, Classifier)
        return model.predict_proba(dataframe, output_col=output_col)

    def evaluate(self, dataframe: DataFrame, label_col: Optional[str] = None) -> Dict[str, Any]:
        """Score :attr:`best_model` on ``dataframe`` (see ``Model.evaluate``).

        Args:
            dataframe: PySpark DataFrame with the feature columns and the label.
            label_col: The label column. Defaults to the one ``fit`` used.

        Returns:
            Metric name -> value; an undefined metric is None.

        Raises:
            ModelNotFittedError: Before ``fit``.
            ValidationError: As ``Model.evaluate`` raises it.
            ColumnNotFoundError: If the label or a feature column is missing.
        """
        self._require_fitted("evaluate()")
        return self.best_model.evaluate(dataframe, label_col=label_col)

    def __repr__(self) -> str:
        text = f"AutomatedTraining(task={self._task!r}, metric={self._metric!r}"
        if self._best_algorithm is not None and self._best_score is not None:
            text += f", best_algorithm={self._best_algorithm!r}, best_score={self._best_score:.6g}"
        return text + ")"
