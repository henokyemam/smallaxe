"""The algorithm seam.

An :class:`Algorithm` is a declarative record describing one learning method
(Random Forest, XGBoost, LightGBM, CatBoost): its parameter table, the Spark
estimator and fitted-model classes per task, how column names and task-fixed
settings reach the estimator, how to read feature importances, and which optional
dependency it needs. :func:`estimator_kwargs` is the single translation rule that
turns a record plus the user's parameter values into an estimator.

The four records live next to their deprecated class aliases in
``smallaxe.training.random_forest`` / ``xgboost`` / ``lightgbm`` / ``catboost``
and are reached through :func:`get`. Nothing here imports those modules, or any
optional package, at import time.
"""

import importlib
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

REGRESSION_TASKS: FrozenSet[str] = frozenset({"simple_regression"})
CLASSIFICATION_TASKS: FrozenSet[str] = frozenset({"binary", "multiclass"})
ALL_TASKS: FrozenSet[str] = REGRESSION_TASKS | CLASSIFICATION_TASKS

# Column roles a model hands to the estimator. Regression uses the first three;
# classification adds probability and raw_prediction.
COLUMN_ROLES: Tuple[str, ...] = ("features", "label", "prediction", "probability", "raw_prediction")

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
    """The optional package an algorithm needs, and how to install it."""

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
        """Tasks this algorithm can train."""
        return frozenset(self.estimators)

    def visible_params(self, task: str) -> Tuple[Param, ...]:
        """The parameter rows that apply to ``task``, in table order."""
        return tuple(p for p in self.params if task in p.tasks)

    @property
    def module(self) -> str:
        """The Python module the estimator classes live in (``"xgboost.spark"``)."""
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
        """
        return self._import_error() is None

    def require(self) -> None:
        """Raise ``DependencyError`` with an install hint if the algorithm is unavailable.

        A plain "module not found" for the estimator module (or a parent package) says
        the package is not installed. Any other ImportError, such as a missing
        dependency of the package, is a broken install, and its text is included.
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
        """Resolve the Spark estimator class for ``task``."""
        return self._resolve(self.estimators[task])

    def model_class(self, task: str) -> Any:
        """Resolve the fitted Spark model class for ``task``."""
        return self._resolve(self.models[task])

    def _resolve(self, path: str) -> Any:
        module_name, _, attr = path.rpartition(".")
        return getattr(_import_estimator_module(self, module_name), attr)


def resolve(path: str) -> Any:
    """Import a dotted ``module.Attribute`` path."""
    module_name, _, attr = path.rpartition(".")
    return getattr(importlib.import_module(module_name), attr)


def _is_missing_module(error: ImportError, module_name: str) -> bool:
    """Whether ``error`` only says ``module_name`` (or a parent package) does not exist."""
    missing = getattr(error, "name", None)
    if not isinstance(error, ModuleNotFoundError) or not missing:
        return False
    return module_name == missing or module_name.startswith(missing + ".")


def estimator_kwargs(
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
        columns: Column role -> column name (see :data:`COLUMN_ROLES`). Ignored when
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
    """Registered algorithm names, in factory order."""
    return list(_MODULES)


def get(name: str) -> Algorithm:
    """Look up an algorithm record by name.

    Raises:
        ValidationError: If ``name`` is not registered.
    """
    if name not in _MODULES:
        raise ValidationError(f"Unknown algorithm '{name}'. Known algorithms are: {names()}")
    return importlib.import_module(_MODULES[name]).ALGORITHM


# ``metadata.json`` written by smallaxe <= 0.8.x records the model class name instead
# of ``algorithm``. The concrete task is read from the ``task`` field either way.
LEGACY_CLASS_NAMES: Dict[str, Tuple[str, str]] = {
    "RandomForestRegressor": ("random_forest", "regression"),
    "RandomForestClassifier": ("random_forest", "classification"),
    "XGBoostRegressor": ("xgboost", "regression"),
    "XGBoostClassifier": ("xgboost", "classification"),
    "LightGBMRegressor": ("lightgbm", "regression"),
    "LightGBMClassifier": ("lightgbm", "classification"),
    "CatBoostRegressor": ("catboost", "regression"),
    "CatBoostClassifier": ("catboost", "classification"),
}

_LEGACY_BY_ALGORITHM: Dict[Tuple[str, str], str] = {
    value: key for key, value in LEGACY_CLASS_NAMES.items()
}


def legacy_class_name(name: str, task_type: str) -> str:
    """The 0.8.x class name for an algorithm and task type (``"XGBoostRegressor"``)."""
    return _LEGACY_BY_ALGORITHM[(name, task_type)]
