"""The public API of smallaxe, pinned. This file is the tripwire for API changes after 1.0.

The public API is every name in the ``__all__`` of a public module (one whose dotted
path has no segment starting with ``_``), the public methods and attributes of every
public class, and their parameters. A function or class defined in a private module and
exported by a public one (``smallaxe.datasets.load_sample_regression``) is pinned under
the public module. The snapshots below record it; they also pin the implementation
classes (``BaseModel``, the mixins), which the README leaves out of the public API but
whose members ``Model`` exposes. A failing test here means the API changed. If that was
deliberate, update the snapshot in the same change and add a CHANGELOG.md entry; under
semantic versioning a removal or an incompatible change needs a new major version, an
addition a new minor version.

Signatures are recorded without annotations (parameter names, kinds and defaults):
annotation reprs differ across the supported Pythons (3.8 spells ``Optional[X]`` as
``Union[X, NoneType]``), and annotations do not change how a call binds.
"""

import ast
import importlib
import inspect
import pkgutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import smallaxe

CHANGE_HINT = (
    "The public API of smallaxe changed. If this is deliberate, update the snapshot in "
    "tests/test_public_api.py in the same change: changing the snapshot is a deliberate "
    "API change and needs a CHANGELOG.md entry (and, for a removal or an incompatible "
    "change, a new major version)."
)


def _all_modules() -> List[str]:
    names = ["smallaxe"]
    names += [info.name for info in pkgutil.walk_packages(smallaxe.__path__, "smallaxe.")]
    return sorted(names)


def _is_public(module_name: str) -> bool:
    return not any(part.startswith("_") for part in module_name.split("."))


def public_modules() -> List[str]:
    """Every public smallaxe module, sorted."""
    return [name for name in _all_modules() if _is_public(name)]


class _Literal:
    """A default value with a repr that does not depend on hash randomisation."""

    def __init__(self, text: str) -> None:
        self.text = text

    def __repr__(self) -> str:
        return self.text


def _stable(default: Any) -> Any:
    """``default``, with sets rendered in sorted order."""
    if isinstance(default, (set, frozenset)):
        items = ", ".join(repr(item) for item in sorted(default, key=repr))
        return _Literal(f"{type(default).__name__}({{{items}}})")
    return default


def _strip(signature: inspect.Signature) -> str:
    """The signature without annotations, as a string."""
    parameters = [
        p.replace(annotation=inspect.Parameter.empty, default=_stable(p.default))
        for p in signature.parameters.values()
    ]
    return str(signature.replace(parameters=parameters, return_annotation=inspect.Signature.empty))


def _describe(member: Any) -> str:
    """How a class member is pinned: its signature, ``property`` or ``attribute``."""
    if isinstance(member, property):
        return "property"
    if isinstance(member, (classmethod, staticmethod)):
        kind = type(member).__name__
        return f"{kind} {_strip(inspect.signature(member.__func__))}"
    if inspect.isfunction(member):
        return _strip(inspect.signature(member))
    return "attribute"


def _class_members(cls: type) -> Dict[str, str]:
    """Public members of ``cls`` defined by smallaxe classes in its MRO, plus ``__init__``."""
    members: Dict[str, str] = {}
    for klass in reversed(cls.__mro__):
        if not klass.__module__.startswith("smallaxe"):
            continue
        for name, member in vars(klass).items():
            if name.startswith("_") and name != "__init__":
                continue
            if name == "__init__" and not inspect.isfunction(member):
                continue
            members[name] = _describe(member)
    return dict(sorted(members.items()))


def _pinned_here(obj: Any, module_name: str) -> bool:
    """Whether ``obj``, exported by ``module_name``, is pinned under that module.

    An object is pinned where it is defined when that module is public and exports it,
    and otherwise under every public module that exports it: a function defined in a
    private module (``smallaxe.datasets._data``) and re-exported by a public one
    (``smallaxe.datasets``) is public API all the same.
    """
    if not (inspect.isfunction(obj) or inspect.isclass(obj)):
        return False  # a submodule or a constant: pinned by __all__ alone
    home = obj.__module__
    if not home.startswith("smallaxe"):
        return False
    if home == module_name:
        return True
    pinned_at_home = _is_public(home) and obj.__name__ in importlib.import_module(home).__all__
    return not pinned_at_home


def public_api() -> Tuple[Dict[str, List[str]], Dict[str, str], Dict[str, Dict[str, str]]]:
    """``(__all__ per module, function signatures, class members)`` of public smallaxe."""
    exported: Dict[str, List[str]] = {}
    functions: Dict[str, str] = {}
    classes: Dict[str, Dict[str, str]] = {}
    for module_name in public_modules():
        module = importlib.import_module(module_name)
        exported[module_name] = sorted(module.__all__)
        for name in module.__all__:
            obj = getattr(module, name)
            if not _pinned_here(obj, module_name):
                continue  # pinned where it is defined
            if inspect.isclass(obj):
                classes[f"{module_name}.{name}"] = _class_members(obj)
            elif inspect.isfunction(obj):
                functions[f"{module_name}.{name}"] = _strip(inspect.signature(obj))
    return exported, functions, classes


def public_callables() -> List[Tuple[str, Callable[..., Any]]]:
    """``(qualified name, function)`` for every public function and method of smallaxe."""
    found: List[Tuple[str, Callable[..., Any]]] = []
    for module_name in public_modules():
        module = importlib.import_module(module_name)
        for name in module.__all__:
            obj = getattr(module, name)
            if not _pinned_here(obj, module_name):
                continue
            if inspect.isfunction(obj):
                found.append((f"{module_name}.{name}", obj))
            elif inspect.isclass(obj):
                for attr, member in vars(obj).items():
                    if attr.startswith("_") and attr != "__init__":
                        continue
                    if isinstance(member, (classmethod, staticmethod)):
                        member = member.__func__
                    if inspect.isfunction(member):
                        found.append((f"{module_name}.{name}.{attr}", member))
    return found


def _top_level_names(path: Path) -> List[str]:
    """Names a module's source defines at top level: functions, classes, assignments."""
    tree = ast.parse(path.read_text())
    names: List[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.append(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.append(node.target.id)
    return names


def test_every_module_defines_all_and_lists_its_public_names():
    """No module of smallaxe, public or private, defines a public name outside ``__all__``."""
    problems = []
    for module_name in _all_modules():
        module = importlib.import_module(module_name)
        if not isinstance(getattr(module, "__all__", None), list):
            problems.append(f"{module_name} has no __all__ list")
            continue
        for name in module.__all__:
            if not hasattr(module, name):
                problems.append(f"{module_name}.__all__ lists {name!r}, which it lacks")
        defined = _top_level_names(Path(module.__file__))
        for name in defined:
            if not name.startswith("_") and name not in module.__all__:
                problems.append(f"{module_name}.{name} is public but not in __all__")
    assert problems == []


def test_task_constants_are_defined_once():
    """The task sets are defined in smallaxe.training.algorithm and imported elsewhere."""
    root = Path(smallaxe.__file__).parent
    definitions = []
    for path in sorted(root.rglob("*.py")):
        for node in ast.parse(path.read_text()).body:
            targets = []
            if isinstance(node, ast.Assign):
                targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                targets = [node.target.id]
            for name in targets:
                if name.lstrip("_") in {"REGRESSION_TASKS", "CLASSIFICATION_TASKS", "ALL_TASKS"}:
                    definitions.append((str(path.relative_to(root)), name))
    assert sorted(definitions) == [
        ("training/algorithm.py", "ALL_TASKS"),
        ("training/algorithm.py", "CLASSIFICATION_TASKS"),
        ("training/algorithm.py", "REGRESSION_TASKS"),
    ]


def test_module_all_snapshot():
    exported, _, _ = public_api()
    assert exported == EXPECTED_ALL, CHANGE_HINT


def test_function_signature_snapshot():
    _, functions, _ = public_api()
    assert functions == EXPECTED_FUNCTIONS, CHANGE_HINT


def test_class_member_snapshot():
    _, _, classes = public_api()
    assert classes == EXPECTED_CLASSES, CHANGE_HINT


def test_every_exported_function_and_class_is_pinned():
    """Each function or class a public module exports has a snapshot entry somewhere.

    Pinned under its defining module when that module is public and exports it, else
    under the exporting module, so a public name implemented in a private module
    (``smallaxe.datasets._data``) cannot escape the tripwire.
    """
    _, functions, classes = public_api()
    unpinned = []
    for module_name in public_modules():
        module = importlib.import_module(module_name)
        for name in module.__all__:
            obj = getattr(module, name)
            if not (inspect.isfunction(obj) or inspect.isclass(obj)):
                continue
            keys = {f"{module_name}.{name}", f"{obj.__module__}.{obj.__name__}"}
            if not keys & (set(functions) | set(classes)):
                unpinned.append(f"{module_name}.{name}")
    assert unpinned == []
    assert "smallaxe.datasets.load_sample_regression" in functions


def _documented_args(docstring: str) -> List[str]:
    """The parameter names listed in the ``Args:`` section of a Google-style docstring."""
    names: List[str] = []
    in_args = False
    indent = None
    for line in inspect.cleandoc(docstring).splitlines():
        stripped = line.strip()
        if stripped == "Args:":
            in_args, indent = True, None
            continue
        if not in_args or not stripped:
            continue
        current = len(line) - len(line.lstrip())
        if current == 0:
            in_args = False  # the next section
            continue
        if indent is None:
            indent = current
        if current == indent and ":" in stripped:
            names.append(stripped.split(":", 1)[0].split(" ", 1)[0].lstrip("*"))
    return names


def test_every_public_callable_documents_its_arguments():
    """Each parameter of a public function or method is listed under ``Args:``.

    A class may document its constructor's arguments in the class docstring instead.
    """
    problems = []
    for name, func in public_callables():
        params = [
            p.name
            for p in inspect.signature(func).parameters.values()
            if p.name not in ("self", "cls")
        ]
        if not params:
            continue
        documented = _documented_args(func.__doc__ or "")
        if func.__name__ == "__init__":
            # ``name`` is "<module>.<Class>.__init__". A dataclass's generated __init__ has
            # no usable __qualname__ on Python 3.8, so the class is resolved from the name.
            module_name, class_name, _ = name.rsplit(".", 2)
            cls = getattr(importlib.import_module(module_name), class_name)
            documented += _documented_args(cls.__doc__ or "")
        missing = [param for param in params if param not in documented]
        if missing:
            problems.append(f"{name}: {missing} not under Args:")
    assert problems == []


def test_training_exports_the_supported_names_only():
    import smallaxe.training

    assert sorted(smallaxe.training.__all__) == [
        "Classifier",
        "Model",
        "Regressor",
        "algorithm",
        "classifiers",
        "regressors",
    ]


def test_fit_set_params_and_clone_are_typed_as_the_model_class():
    """``Regressor.fit(...)`` is typed as a Regressor: the return is the class's TypeVar."""
    from smallaxe.training import base
    from smallaxe.training.mixins import param_mixin

    for method in (base.BaseModel.fit, base.Model.clone, param_mixin.ParamMixin.set_params):
        hints = method.__annotations__
        self_type = hints["self"]
        assert hints["return"] is self_type, method.__qualname__
        assert getattr(self_type, "__bound__", None) is not None, method.__qualname__


# --- Snapshots ------------------------------------------------------------------------
# Regenerate with ``python -c "from tests.test_public_api import public_api; ..."`` only
# for a deliberate API change, together with a CHANGELOG.md entry.

EXPECTED_ALL: Dict[str, List[str]] = {
    "smallaxe": [
        "__version__",
        "get_seed",
        "get_verbosity",
        "load",
        "load_model",
        "load_pipeline",
        "set_seed",
        "set_verbosity",
        "verbosity",
    ],
    "smallaxe.auto": [
        "AutomatedTraining",
    ],
    "smallaxe.datasets": [
        "dataset_info",
        "load_sample_classification",
        "load_sample_regression",
    ],
    "smallaxe.exceptions": [
        "ColumnNotFoundError",
        "ConfigurationError",
        "DependencyError",
        "ModelNotFittedError",
        "PreprocessingError",
        "SmallaxeError",
        "ValidationError",
    ],
    "smallaxe.metrics": [
        "TASK_METRICS",
        "accuracy",
        "auc_pr",
        "auc_roc",
        "f1_score",
        "log_loss",
        "mae",
        "mape",
        "mse",
        "multiclass_scores",
        "precision",
        "r2",
        "recall",
        "rmse",
    ],
    "smallaxe.metrics.classification": [
        "accuracy",
        "auc_pr",
        "auc_roc",
        "f1_score",
        "log_loss",
        "multiclass_scores",
        "precision",
        "recall",
    ],
    "smallaxe.metrics.regression": [
        "mae",
        "mape",
        "mse",
        "r2",
        "rmse",
    ],
    "smallaxe.pipeline": [
        "Pipeline",
    ],
    "smallaxe.pipeline.pipeline": [
        "Pipeline",
    ],
    "smallaxe.preprocessing": [
        "Encoder",
        "Imputer",
        "Scaler",
    ],
    "smallaxe.preprocessing.encoder": [
        "Encoder",
    ],
    "smallaxe.preprocessing.imputer": [
        "Imputer",
    ],
    "smallaxe.preprocessing.scaler": [
        "Scaler",
    ],
    "smallaxe.search": [
        "SearchResult",
        "optimize",
    ],
    "smallaxe.search.optimize": [
        "SearchResult",
        "run",
    ],
    "smallaxe.training": [
        "Classifier",
        "Model",
        "Regressor",
        "algorithm",
        "classifiers",
        "regressors",
    ],
    "smallaxe.training.algorithm": [
        "ALL_TASKS",
        "Algorithm",
        "CLASSIFICATION_TASKS",
        "Dependency",
        "FitContext",
        "Importances",
        "Param",
        "REGRESSION_TASKS",
        "get",
        "names",
    ],
    "smallaxe.training.base": [
        "BaseClassifier",
        "BaseModel",
        "BaseRegressor",
        "Classifier",
        "Model",
        "Regressor",
    ],
    "smallaxe.training.catboost": [
        "ALGORITHM",
    ],
    "smallaxe.training.classifiers": [
        "algorithms",
        "available_models",
        "catboost",
        "lightgbm",
        "load",
        "random_forest",
        "xgboost",
    ],
    "smallaxe.training.lightgbm": [
        "ALGORITHM",
    ],
    "smallaxe.training.mixins": [
        "MetadataMixin",
        "ParamMixin",
        "PersistenceMixin",
        "SparkModelMixin",
        "ValidationMixin",
    ],
    "smallaxe.training.mixins.metadata_mixin": [
        "MetadataMixin",
    ],
    "smallaxe.training.mixins.param_mixin": [
        "ParamMixin",
    ],
    "smallaxe.training.mixins.persistence_mixin": [
        "PersistenceMixin",
    ],
    "smallaxe.training.mixins.spark_model_mixin": [
        "SparkModelMixin",
    ],
    "smallaxe.training.mixins.validation_mixin": [
        "ValidationMixin",
    ],
    "smallaxe.training.random_forest": [
        "ALGORITHM",
    ],
    "smallaxe.training.regressors": [
        "algorithms",
        "available_models",
        "catboost",
        "lightgbm",
        "load",
        "random_forest",
        "xgboost",
    ],
    "smallaxe.training.xgboost": [
        "ALGORITHM",
    ],
    "smallaxe.viz": [],
}

EXPECTED_FUNCTIONS: Dict[str, str] = {
    "smallaxe.datasets.dataset_info": "(dataset_name, *, verbose=False)",
    "smallaxe.datasets.load_sample_classification": "(spark=None, n_rows=10000, seed=42)",
    "smallaxe.datasets.load_sample_regression": "(spark=None, n_rows=10000, seed=42)",
    "smallaxe.get_seed": "()",
    "smallaxe.get_verbosity": "()",
    "smallaxe.load": "(path)",
    "smallaxe.load_model": "(path)",
    "smallaxe.load_pipeline": "(path)",
    "smallaxe.metrics.classification.accuracy": "(dataframe, label_col='label', prediction_col='predict_label')",
    "smallaxe.metrics.classification.auc_pr": "(dataframe, label_col='label', probability_col='probability')",
    "smallaxe.metrics.classification.auc_roc": "(dataframe, label_col='label', probability_col='probability')",
    "smallaxe.metrics.classification.f1_score": "(dataframe, label_col='label', prediction_col='predict_label', *, average='binary')",
    "smallaxe.metrics.classification.log_loss": "(dataframe, label_col='label', probability_col='probability', *, eps=1e-15)",
    "smallaxe.metrics.classification.multiclass_scores": "(dataframe, label_col='label', prediction_col='predict_label')",
    "smallaxe.metrics.classification.precision": "(dataframe, label_col='label', prediction_col='predict_label', *, average='binary')",
    "smallaxe.metrics.classification.recall": "(dataframe, label_col='label', prediction_col='predict_label', *, average='binary')",
    "smallaxe.metrics.regression.mae": "(dataframe, label_col='label', prediction_col='predict_label')",
    "smallaxe.metrics.regression.mape": "(dataframe, label_col='label', prediction_col='predict_label')",
    "smallaxe.metrics.regression.mse": "(dataframe, label_col='label', prediction_col='predict_label')",
    "smallaxe.metrics.regression.r2": "(dataframe, label_col='label', prediction_col='predict_label')",
    "smallaxe.metrics.regression.rmse": "(dataframe, label_col='label', prediction_col='predict_label')",
    "smallaxe.search.optimize.run": "(model, dataframe, label_col, param_space, metric=None, max_evals=20, validation='train_test', n_folds=5, test_size=0.2, feature_cols=None, exclude_cols=None, fold_key=None, seed=None, refit=True, verbose=None, early_stopping=False, early_stopping_rounds=10)",
    "smallaxe.set_seed": "(seed)",
    "smallaxe.set_verbosity": "(level)",
    "smallaxe.training.algorithm.get": "(name)",
    "smallaxe.training.algorithm.names": "()",
    "smallaxe.training.classifiers.algorithms": "()",
    "smallaxe.training.classifiers.available_models": "()",
    "smallaxe.training.classifiers.catboost": "(task='binary', **params)",
    "smallaxe.training.classifiers.lightgbm": "(task='binary', **params)",
    "smallaxe.training.classifiers.load": "(path)",
    "smallaxe.training.classifiers.random_forest": "(task='binary', **params)",
    "smallaxe.training.classifiers.xgboost": "(task='binary', **params)",
    "smallaxe.training.regressors.algorithms": "()",
    "smallaxe.training.regressors.available_models": "()",
    "smallaxe.training.regressors.catboost": "(**params)",
    "smallaxe.training.regressors.lightgbm": "(**params)",
    "smallaxe.training.regressors.load": "(path)",
    "smallaxe.training.regressors.random_forest": "(**params)",
    "smallaxe.training.regressors.xgboost": "(**params)",
    "smallaxe.verbosity": "(level)",
}

EXPECTED_CLASSES: Dict[str, Dict[str, str]] = {
    "smallaxe.auto.AutomatedTraining": {
        "__init__": "(self, task, algorithms=None, metric=None, params=None, param_spaces=None, max_evals=20, seed=None)",
        "algorithms": "property",
        "best_algorithm": "property",
        "best_model": "property",
        "best_score": "property",
        "comparison": "property",
        "evaluate": "(self, dataframe, label_col=None)",
        "fit": "(self, dataframe, label_col, feature_cols=None, exclude_cols=None, *, validation='train_test', test_size=0.2, n_folds=5, fold_key=None, cache_strategy=None)",
        "metric": "property",
        "models": "property",
        "predict": "(self, dataframe, output_col='predict_label')",
        "predict_proba": "(self, dataframe, output_col='probability')",
        "task": "property",
    },
    "smallaxe.exceptions.ColumnNotFoundError": {
        "__init__": "(self, message='Required column not found in DataFrame.', column=None, available_columns=None)",
    },
    "smallaxe.exceptions.ConfigurationError": {
        "__init__": "(self, message='Invalid configuration settings.', setting=None, value=<no value>, allowed_values=None)",
    },
    "smallaxe.exceptions.DependencyError": {
        "__init__": "(self, message='Missing optional dependency.', package=None, install_command=None)",
    },
    "smallaxe.exceptions.ModelNotFittedError": {
        "__init__": "(self, message='Model has not been fitted. Call fit() before predict().')",
    },
    "smallaxe.exceptions.PreprocessingError": {
        "__init__": "(self, message='Missing required preprocessing steps.', algorithm=None, missing_step=None)",
    },
    "smallaxe.exceptions.SmallaxeError": {
        "__init__": "(self, message='An error occurred in smallaxe.')",
    },
    "smallaxe.exceptions.ValidationError": {
        "__init__": "(self, message='Invalid input parameters or data.')",
    },
    "smallaxe.pipeline.pipeline.Pipeline": {
        "__init__": "(self, steps)",
        "evaluate": "(self, dataframe, label_col=None)",
        "feature_cols": "property",
        "fit": "(self, dataframe, label_col=None, numerical_cols=None, categorical_cols=None, *, validation='none', test_size=0.2, n_folds=5, fold_key=None, refit=True, cache_strategy=None)",
        "fit_transform": "(self, dataframe, label_col=None, numerical_cols=None, categorical_cols=None, *, validation='none', test_size=0.2, n_folds=5, fold_key=None, refit=True, cache_strategy=None)",
        "load": "classmethod (cls, path)",
        "named_steps": "property",
        "predict": "(self, dataframe, output_col='predict_label')",
        "predict_proba": "(self, dataframe, output_col='probability')",
        "save": "(self, path)",
        "skipped_steps": "property",
        "steps": "property",
        "transform": "(self, dataframe)",
        "validation_scores": "property",
    },
    "smallaxe.preprocessing.encoder.Encoder": {
        "__init__": "(self, method='onehot', max_categories=100, handle_rare='other')",
        "categorical_cols": "property",
        "category_mappings": "property",
        "fit": "(self, dataframe, categorical_cols)",
        "fit_transform": "(self, dataframe, categorical_cols)",
        "handle_rare": "property",
        "load": "classmethod (cls, path)",
        "max_categories": "property",
        "method": "property",
        "output_cols": "property",
        "save": "(self, path)",
        "transform": "(self, dataframe)",
    },
    "smallaxe.preprocessing.imputer.Imputer": {
        "__init__": "(self, numerical_strategy=None, categorical_strategy=None)",
        "categorical_fill_values": "property",
        "fit": "(self, dataframe, numerical_cols=None, categorical_cols=None)",
        "fit_transform": "(self, dataframe, numerical_cols=None, categorical_cols=None)",
        "load": "classmethod (cls, path)",
        "numerical_fill_values": "property",
        "save": "(self, path)",
        "transform": "(self, dataframe)",
    },
    "smallaxe.preprocessing.scaler.Scaler": {
        "__init__": "(self, method='standard')",
        "fit": "(self, dataframe, numerical_cols)",
        "fit_transform": "(self, dataframe, numerical_cols)",
        "load": "classmethod (cls, path)",
        "method": "property",
        "numerical_cols": "property",
        "save": "(self, path)",
        "transform": "(self, dataframe)",
    },
    "smallaxe.search.optimize.SearchResult": {
        "__init__": "(self, best_params, best_score, best_model, trials_history, metric, validation, max_evals)",
        "n_successful_trials": "property",
        "n_trials": "property",
    },
    "smallaxe.training.algorithm.Algorithm": {
        "__init__": "(self, name, params, estimators, models, cols=None, task_params=<factory>, importances=None, dependency=None, accepts_raw_categoricals=False, accepts_missing=False, prepare=None, fit_context=None)",
        "accepts_missing": "attribute",
        "accepts_raw_categoricals": "attribute",
        "available": "(self)",
        "cols": "attribute",
        "dependency": "attribute",
        "estimator_class": "(self, task)",
        "fit_context": "attribute",
        "importances": "attribute",
        "model_class": "(self, task)",
        "module": "property",
        "prepare": "attribute",
        "require": "(self)",
        "tasks": "property",
        "visible_params": "(self, task)",
    },
    "smallaxe.training.algorithm.Dependency": {
        "__init__": "(self, package, install_hint)",
    },
    "smallaxe.training.algorithm.Param": {
        "__init__": "(self, name, description, default, spark=None, setter=None, tasks=frozenset({'binary', 'multiclass', 'simple_regression'}))",
        "setter": "attribute",
        "spark": "attribute",
        "tasks": "attribute",
    },
    "smallaxe.training.base.BaseClassifier": {
        "FEATURES_COL": "attribute",
        "PREDICTION_COL": "attribute",
        "PROBABILITY_COL": "attribute",
        "RAW_PREDICTION_COL": "attribute",
        "__init__": "(self, task='binary')",
        "default_params": "property",
        "evaluate": "(self, dataframe, label_col=None)",
        "feature_importances": "property",
        "fit": "(self, dataframe, label_col, feature_cols=None, exclude_cols=None, validation='none', test_size=0.2, n_folds=5, cache_strategy=None, fold_key=None, refit=True)",
        "get_param": "(self, name)",
        "get_params": "(self)",
        "metadata": "property",
        "params": "property",
        "predict": "(self, dataframe, output_col='predict_label')",
        "predict_proba": "(self, dataframe, output_col='probability')",
        "save": "(self, path)",
        "set_params": "(self, **params)",
        "task": "property",
        "task_type": "property",
        "validation_scores": "property",
    },
    "smallaxe.training.base.BaseModel": {
        "FEATURES_COL": "attribute",
        "PREDICTION_COL": "attribute",
        "PROBABILITY_COL": "attribute",
        "RAW_PREDICTION_COL": "attribute",
        "__init__": "(self, task)",
        "default_params": "property",
        "evaluate": "(self, dataframe, label_col=None)",
        "feature_importances": "property",
        "fit": "(self, dataframe, label_col, feature_cols=None, exclude_cols=None, validation='none', test_size=0.2, n_folds=5, cache_strategy=None, fold_key=None, refit=True)",
        "get_param": "(self, name)",
        "get_params": "(self)",
        "metadata": "property",
        "params": "property",
        "predict": "(self, dataframe, output_col='predict_label')",
        "save": "(self, path)",
        "set_params": "(self, **params)",
        "task": "property",
        "task_type": "property",
        "validation_scores": "property",
    },
    "smallaxe.training.base.BaseRegressor": {
        "FEATURES_COL": "attribute",
        "PREDICTION_COL": "attribute",
        "PROBABILITY_COL": "attribute",
        "RAW_PREDICTION_COL": "attribute",
        "__init__": "(self, task='simple_regression')",
        "default_params": "property",
        "evaluate": "(self, dataframe, label_col=None)",
        "feature_importances": "property",
        "fit": "(self, dataframe, label_col, feature_cols=None, exclude_cols=None, validation='none', test_size=0.2, n_folds=5, cache_strategy=None, fold_key=None, refit=True)",
        "get_param": "(self, name)",
        "get_params": "(self)",
        "metadata": "property",
        "params": "property",
        "predict": "(self, dataframe, output_col='predict_label')",
        "save": "(self, path)",
        "set_params": "(self, **params)",
        "task": "property",
        "task_type": "property",
        "validation_scores": "property",
    },
    "smallaxe.training.base.Classifier": {
        "FEATURES_COL": "attribute",
        "PREDICTION_COL": "attribute",
        "PROBABILITY_COL": "attribute",
        "RAW_PREDICTION_COL": "attribute",
        "__init__": "(self, algorithm, task='binary')",
        "algorithm": "property",
        "clone": "(self)",
        "default_params": "property",
        "evaluate": "(self, dataframe, label_col=None)",
        "feature_importances": "property",
        "fit": "(self, dataframe, label_col, feature_cols=None, exclude_cols=None, validation='none', test_size=0.2, n_folds=5, cache_strategy=None, fold_key=None, refit=True)",
        "get_param": "(self, name)",
        "get_params": "(self)",
        "load": "classmethod (cls, path)",
        "metadata": "property",
        "params": "property",
        "predict": "(self, dataframe, output_col='predict_label')",
        "predict_proba": "(self, dataframe, output_col='probability')",
        "save": "(self, path)",
        "set_params": "(self, **params)",
        "task": "property",
        "task_type": "property",
        "validation_scores": "property",
    },
    "smallaxe.training.base.Model": {
        "FEATURES_COL": "attribute",
        "PREDICTION_COL": "attribute",
        "PROBABILITY_COL": "attribute",
        "RAW_PREDICTION_COL": "attribute",
        "__init__": "(self, algorithm, task)",
        "algorithm": "property",
        "clone": "(self)",
        "default_params": "property",
        "evaluate": "(self, dataframe, label_col=None)",
        "feature_importances": "property",
        "fit": "(self, dataframe, label_col, feature_cols=None, exclude_cols=None, validation='none', test_size=0.2, n_folds=5, cache_strategy=None, fold_key=None, refit=True)",
        "get_param": "(self, name)",
        "get_params": "(self)",
        "load": "classmethod (cls, path)",
        "metadata": "property",
        "params": "property",
        "predict": "(self, dataframe, output_col='predict_label')",
        "save": "(self, path)",
        "set_params": "(self, **params)",
        "task": "property",
        "task_type": "property",
        "validation_scores": "property",
    },
    "smallaxe.training.base.Regressor": {
        "FEATURES_COL": "attribute",
        "PREDICTION_COL": "attribute",
        "PROBABILITY_COL": "attribute",
        "RAW_PREDICTION_COL": "attribute",
        "__init__": "(self, algorithm, task='simple_regression')",
        "algorithm": "property",
        "clone": "(self)",
        "default_params": "property",
        "evaluate": "(self, dataframe, label_col=None)",
        "feature_importances": "property",
        "fit": "(self, dataframe, label_col, feature_cols=None, exclude_cols=None, validation='none', test_size=0.2, n_folds=5, cache_strategy=None, fold_key=None, refit=True)",
        "get_param": "(self, name)",
        "get_params": "(self)",
        "load": "classmethod (cls, path)",
        "metadata": "property",
        "params": "property",
        "predict": "(self, dataframe, output_col='predict_label')",
        "save": "(self, path)",
        "set_params": "(self, **params)",
        "task": "property",
        "task_type": "property",
        "validation_scores": "property",
    },
    "smallaxe.training.mixins.metadata_mixin.MetadataMixin": {
        "metadata": "property",
    },
    "smallaxe.training.mixins.param_mixin.ParamMixin": {
        "default_params": "property",
        "get_param": "(self, name)",
        "get_params": "(self)",
        "params": "property",
        "set_params": "(self, **params)",
    },
    "smallaxe.training.mixins.persistence_mixin.PersistenceMixin": {
        "save": "(self, path)",
    },
    "smallaxe.training.mixins.spark_model_mixin.SparkModelMixin": {
        "FEATURES_COL": "attribute",
        "PREDICTION_COL": "attribute",
        "PROBABILITY_COL": "attribute",
        "RAW_PREDICTION_COL": "attribute",
        "feature_importances": "property",
    },
    "smallaxe.training.mixins.validation_mixin.ValidationMixin": {},
}
