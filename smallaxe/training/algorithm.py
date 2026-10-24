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
    prepare: Optional[Callable[[], None]] = None
    fit_context: Optional[FitContext] = None

    @property
    def tasks(self) -> FrozenSet[str]:
        """Tasks this algorithm can train."""
        return frozenset(self.estimators)

    def visible_params(self, task: str) -> Tuple[Param, ...]:
        """The parameter rows that apply to ``task``, in table order."""
        return tuple(p for p in self.params if task in p.tasks)

    def available(self) -> bool:
        """Whether the estimator module imports right now.

        Re-probed on every call rather than cached, because catboost_spark only
        becomes importable once the JVM package is on the Spark classpath.
        """
        module_name = next(iter(self.estimators.values())).rpartition(".")[0]
        try:
            importlib.import_module(module_name)
        except ImportError:
            return False
        return True

    def require(self) -> None:
        """Raise ``DependencyError`` with an install hint if the algorithm is unavailable."""
        if self.available():
            return
        if self.dependency is None:
            raise DependencyError(package=self.name)
        raise DependencyError(
            package=self.dependency.package,
            install_command=self.dependency.install_hint,
        )

    def estimator_class(self, task: str) -> Any:
        """Resolve the Spark estimator class for ``task``."""
        return resolve(self.estimators[task])

    def model_class(self, task: str) -> Any:
        """Resolve the fitted Spark model class for ``task``."""
        return resolve(self.models[task])


def resolve(path: str) -> Any:
    """Import a dotted ``module.Attribute`` path."""
    module_name, _, attr = path.rpartition(".")
    return getattr(importlib.import_module(module_name), attr)


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
