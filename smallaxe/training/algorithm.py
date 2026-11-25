"""The algorithm seam.

An :class:`Algorithm` is a declarative record describing one learning method
(Random Forest, XGBoost, LightGBM, CatBoost): its parameter table, the Spark
estimator and fitted-model classes per task, how column names and task-fixed
settings reach the estimator, how to read feature importances, and which optional
dependency it needs. :func:`_estimator_kwargs` is the single translation rule that
turns a record plus the user's parameter values into an estimator.

The four records live in ``smallaxe.training.random_forest`` / ``xgboost`` /
``lightgbm`` / ``catboost`` (one ``ALGORITHM`` each) and are reached through
:func:`get`. Nothing here imports those modules, or any optional package, at import
time.

This module also defines the task names, once: :data:`REGRESSION_TASKS`,
:data:`CLASSIFICATION_TASKS` and :data:`ALL_TASKS`.
"""

import importlib
import importlib.metadata
import sys
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    ContextManager,
    Dict,
    FrozenSet,
    List,
    Mapping,
    Optional,
    Tuple,
)

from smallaxe.exceptions import DependencyError, ValidationError

__all__ = [
    "REGRESSION_TASKS",
    "CLASSIFICATION_TASKS",
    "ALL_TASKS",
    "Importances",
    "FitContext",
    "Param",
    "Dependency",
    "Algorithm",
    "names",
    "get",
]

# The tasks a model can train, by kind. Every other module imports these.
REGRESSION_TASKS: FrozenSet[str] = frozenset({"simple_regression"})
CLASSIFICATION_TASKS: FrozenSet[str] = frozenset({"binary", "multiclass"})
ALL_TASKS: FrozenSet[str] = REGRESSION_TASKS | CLASSIFICATION_TASKS

# Column roles a model hands to the estimator. Regression uses the first three;
# classification adds probability and raw_prediction.
_COLUMN_ROLES: Tuple[str, ...] = (
    "features",
    "label",
    "prediction",
    "probability",
    "raw_prediction",
)

# ``importances(spark_model, n_features)`` returns one score per assembled feature, in
# feature order, or None when the fitted model cannot supply them.
Importances = Callable[[Any, int], Optional[List[float]]]

# ``fit_context(param_values)`` is entered around ``estimator.fit`` and yields extra
# constructor kwargs; its exit runs cleanup even when fit fails.
FitContext = Callable[[Dict[str, Any]], ContextManager[Dict[str, Any]]]


@dataclass(frozen=True)
class Param:
    """One row of an algorithm's strict parameter table.

    Args:
        name: smallaxe parameter name, as users set it.
        description: Shown by ``model.params``.
        default: Default value; its type drives validation. ``None`` means "let the
            estimator decide" and is never forwarded.
        spark: Estimator constructor kwarg. ``None`` means the same as ``name``.
        setter: Estimator setter to call after construction instead of a kwarg
            (``"setSeed"``), for estimators that reject the value in the constructor.
        tasks: Tasks the parameter applies to. Defaults to all.
    """

    name: str
    description: str
    default: Any
    spark: Optional[str] = None
    setter: Optional[str] = None
    tasks: FrozenSet[str] = ALL_TASKS


@dataclass(frozen=True)
class Dependency:
    """The optional package an algorithm needs, and how to install it.

    Args:
        package: The package name (``"xgboost"``).
        install_hint: How to install it (``"pip install smallaxe[xgboost]"``).
    """

    package: str
    install_hint: str


@dataclass(frozen=True)
class Algorithm:
    """Declarative description of one learning method.

    Args:
        name: Registry key and the value persisted in ``metadata.json`` (``"xgboost"``).
        params: The strict parameter table.
        estimators: Task -> dotted path of the Spark estimator class, resolved lazily.
        models: Task -> dotted path of the fitted-model class used by ``load``.
        cols: Column role -> estimator constructor kwarg. ``None`` means the estimator
            takes columns through the standard pyspark.ml setters instead.
        task_params: Task -> estimator kwargs fixed by the task (LightGBM ``objective``,
            CatBoost ``lossFunction``). Override user params of the same name.
        importances: Reads feature importances from a fitted model; see :data:`Importances`.
        dependency: Optional package; ``None`` for algorithms shipped with PySpark.
        accepts_raw_categoricals: Whether the estimator handles string columns itself,
            so a Pipeline need not add an Encoder.
        accepts_missing: Whether the estimator treats NaN as a missing value. If so,
            null and NaN features are kept and handed to it as NaN; otherwise ``fit``
            rejects them with a ``ValidationError``.
        prepare: Runs before every fit and load (CatBoost's JVM compatibility shim).
        fit_context: See :data:`FitContext` (CatBoost's temporary training directory).
    """

    name: str
    params: Tuple[Param, ...]
    estimators: Mapping[str, str]
    models: Mapping[str, str]
    cols: Optional[Mapping[str, str]] = None
    task_params: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    importances: Optional[Importances] = None
    dependency: Optional[Dependency] = None
    accepts_raw_categoricals: bool = False
    accepts_missing: bool = False
    prepare: Optional[Callable[[], None]] = None
    fit_context: Optional[FitContext] = None

    @property
    def tasks(self) -> FrozenSet[str]:
        """Tasks this algorithm can train.

        Returns:
            The tasks it has an estimator for.
        """
        return frozenset(self.estimators)

    def visible_params(self, task: str) -> Tuple[Param, ...]:
        """The parameter rows that apply to ``task``.

        Args:
            task: A task name.

        Returns:
            The rows whose ``tasks`` include ``task``, in table order (empty for a task
            the algorithm does not know).
        """
        return tuple(p for p in self.params if task in p.tasks)

    @property
    def module(self) -> str:
        """The Python module the estimator classes live in.

        Returns:
            Its dotted name (``"xgboost.spark"``).
        """
        return next(iter(self.estimators.values())).rpartition(".")[0]

    def _import_error(self) -> Optional[ImportError]:
        """Import the estimator module; return the ImportError if that fails."""
        try:
            _import_estimator_module(self, self.module)
        except ImportError as exc:
            return exc
        return None

    def available(self) -> bool:
        """Whether the estimator module imports right now.

        Re-probed on every call rather than cached, because catboost_spark only
        becomes importable once the JVM package is on the Spark classpath.

        Returns:
            True if the algorithm can be used in this process.
        """
        return self._import_error() is None

    def require(self) -> None:
        """Raise ``DependencyError`` with an install hint if the algorithm is unavailable.

        A plain "module not found" for the estimator module (or a parent package) says
        the package is not installed. Any other ImportError, such as a missing
        dependency of the package, is a broken install, and its text is included.

        Raises:
            DependencyError: If the estimator module cannot be imported.
        """
        error = self._import_error()
        if error is None:
            return
        package = self.dependency.package if self.dependency is not None else self.name
        hint = self.dependency.install_hint if self.dependency is not None else None
        if _is_missing_module(error, self.module):
            raise DependencyError(package=package, install_command=hint) from error
        message = f"{package} could not be imported: {type(error).__name__}: {error}."
        if hint:
            message += f" Install or repair it with: {hint}"
        broken = DependencyError(message)
        broken.package = package
        broken.install_command = hint
        raise broken from error

    def estimator_class(self, task: str) -> Any:
        """Resolve the Spark estimator class for ``task``, importing its module.

        Args:
            task: One of :attr:`tasks`.

        Returns:
            The estimator class.

        Raises:
            ValidationError: If the algorithm has no estimator for ``task``.
            ImportError: If the estimator module is not installed (use :meth:`require`
                first for a ``DependencyError`` with an install hint).
        """
        return self._resolve(self._path(self.estimators, task))

    def model_class(self, task: str) -> Any:
        """Resolve the fitted Spark model class for ``task``, importing its module.

        Args:
            task: One of :attr:`tasks`.

        Returns:
            The fitted-model class ``load`` reads a saved Spark model with.

        Raises:
            ValidationError: If the algorithm has no model class for ``task``.
            ImportError: If the model module is not installed (use :meth:`require`
                first for a ``DependencyError`` with an install hint).
        """
        return self._resolve(self._path(self.models, task))

    def _path(self, paths: Mapping[str, str], task: str) -> str:
        if task not in paths:
            raise ValidationError(
                f"Algorithm '{self.name}' has no task {task!r}; its tasks are {sorted(paths)}."
            )
        return paths[task]

    def _resolve(self, path: str) -> Any:
        module_name, _, attr = path.rpartition(".")
        return getattr(_import_estimator_module(self, module_name), attr)


def _package_version(record: Algorithm) -> Optional[str]:
    """The installed version of ``record``'s optional package, for diagnosis.

    Read from the package's distribution metadata, or else from the ``__version__`` of
    its module when that is already imported (catboost_spark ships inside the Spark
    package jar, so pip may not know it). Never imports anything.

    Returns:
        The version string, or None for an algorithm shipped with PySpark or when the
        version cannot be read.
    """
    if record.dependency is None:
        return None
    try:
        return importlib.metadata.version(record.dependency.package)
    except importlib.metadata.PackageNotFoundError:
        pass
    module = sys.modules.get(record.module.partition(".")[0])
    value = getattr(module, "__version__", None)
    return None if value is None else str(value)


def _resolve(path: str) -> Any:
    """Import a dotted ``module.Attribute`` path."""
    module_name, _, attr = path.rpartition(".")
    return getattr(importlib.import_module(module_name), attr)


def _is_missing_module(error: ImportError, module_name: str) -> bool:
    """Whether ``error`` only says ``module_name`` (or a parent package) does not exist."""
    missing = getattr(error, "name", None)
    if not isinstance(error, ModuleNotFoundError) or not missing:
        return False
    return module_name == missing or module_name.startswith(missing + ".")


def _estimator_kwargs(
    algorithm: Algorithm,
    task: str,
    values: Mapping[str, Any],
    columns: Mapping[str, str],
) -> Tuple[Dict[str, Any], List[Tuple[str, Any]]]:
    """Translate smallaxe parameter values into estimator kwargs and setter calls.

    Order, later keys winning: the task's visible params (``None`` is never
    forwarded), then ``task_params[task]``, then column names. Parameters with a
    ``setter`` are returned separately as ``(setter_name, value)`` pairs.

    Args:
        algorithm: The algorithm record.
        task: The task being trained.
        values: Current parameter values by smallaxe name; missing names use defaults.
        columns: Column role -> column name (see :data:`_COLUMN_ROLES`). Ignored when
            ``algorithm.cols`` is ``None``.

    Returns:
        ``(constructor_kwargs, setters)``.
    """
    ctor: Dict[str, Any] = {}
    setters: List[Tuple[str, Any]] = []

    for param in algorithm.visible_params(task):
        value = values.get(param.name, param.default)
        if value is None:
            continue
        if param.setter is not None:
            setters.append((param.setter, value))
        else:
            ctor[param.spark or param.name] = value

    ctor.update(algorithm.task_params.get(task, {}))

    if algorithm.cols is not None:
        for role, column in columns.items():
            kwarg = algorithm.cols.get(role)
            if kwarg is not None:
                ctor[kwarg] = column

    return ctor, setters


# --- Registry -------------------------------------------------------------------

_MODULES: Dict[str, str] = {
    "random_forest": "smallaxe.training.random_forest",
    "xgboost": "smallaxe.training.xgboost",
    "lightgbm": "smallaxe.training.lightgbm",
    "catboost": "smallaxe.training.catboost",
}


# Algorithms whose estimator module patches ``pyspark.ml.wrapper`` process-wide when
# imported, in the order those modules must first be imported. CatBoost's
# compatibility shim (``catboost._ensure_synapseml_compat``) assumes SynapseML's
# patches were applied last, so catboost_spark is imported before synapse.ml.lightgbm.
_PATCHES_PYSPARK: Tuple[str, ...] = ("catboost", "lightgbm")


def _import_estimator_module(record: Algorithm, module_name: str) -> Any:
    """Import one of ``record``'s estimator or model modules, keeping the JVM import order.

    The first time any :data:`_PATCHES_PYSPARK` algorithm's module is about to be
    imported, every one of those modules is imported first, in that order; each is
    skipped if it is not installed. Everything else is a plain import.

    Raises:
        ImportError: If ``module_name`` itself cannot be imported.
    """
    if record.name in _PATCHES_PYSPARK and module_name not in sys.modules:
        for name in _PATCHES_PYSPARK:
            other = record.module if name == record.name else get(name).module
            if other in sys.modules:
                continue
            try:
                importlib.import_module(other)
            except ImportError:
                pass  # Not installed (or not on the classpath yet); the next may be.
    return importlib.import_module(module_name)


def names() -> List[str]:
    """Registered algorithm names, in factory order.

    Returns:
        ``["random_forest", "xgboost", "lightgbm", "catboost"]``.
    """
    return list(_MODULES)


def get(name: str) -> Algorithm:
    """Look up an algorithm record by name. Imports no optional package.

    Args:
        name: An algorithm name, one of :func:`names`.

    Returns:
        The algorithm's record.

    Raises:
        ValidationError: If ``name`` is not registered.
    """
    if not isinstance(name, str):
        raise ValidationError(f"Algorithm name must be a string, got {type(name).__name__}.")
    if name not in _MODULES:
        raise ValidationError(f"Unknown algorithm '{name}'. Known algorithms are: {names()}")
    return importlib.import_module(_MODULES[name]).ALGORITHM


def _availability() -> Dict[str, Dict[str, Any]]:
    """What ``regressors.available_models()`` / ``classifiers.available_models()`` report.

    Returns:
        Algorithm name -> ``{"available", "dependency", "install_hint"}``; the hint is
        None when the algorithm is installed or needs no optional package.
    """
    report: Dict[str, Dict[str, Any]] = {}
    for name in names():
        record = get(name)
        available = record.available()
        dependency = record.dependency
        report[name] = {
            "available": available,
            "dependency": dependency.package if dependency is not None else None,
            "install_hint": None if available or dependency is None else dependency.install_hint,
        }
    return report


# ``metadata.json`` written by smallaxe <= 0.8.x records the model class name instead
# of ``algorithm``; this map reads those artifacts and is used nowhere else. The task is
# read from the ``task`` field either way.
_LEGACY_CLASS_NAMES: Dict[str, str] = {
    "RandomForestRegressor": "random_forest",
    "RandomForestClassifier": "random_forest",
    "XGBoostRegressor": "xgboost",
    "XGBoostClassifier": "xgboost",
    "LightGBMRegressor": "lightgbm",
    "LightGBMClassifier": "lightgbm",
    "CatBoostRegressor": "catboost",
    "CatBoostClassifier": "catboost",
}
