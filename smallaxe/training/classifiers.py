"""Classification models: one constructor function per algorithm, plus load.

Example:
    >>> from smallaxe.training import classifiers
    >>>
    >>> model = classifiers.random_forest(n_estimators=100, max_depth=10)
    >>> model.fit(dataframe, label_col='label', feature_cols=['f1', 'f2'])
    >>> model.save('/path/to/model')
    >>> loaded_model = classifiers.load('/path/to/model')

Each function returns a :class:`~smallaxe.training.base.Classifier` bound to one
algorithm, so no algorithm-specific class needs to be imported. The ``Classifiers``
class is the deprecated 0.x spelling of these functions and is removed in smallaxe 1.0.
"""

from typing import Any, Dict, List

from smallaxe import _deprecation
from smallaxe.exceptions import ValidationError
from smallaxe.training import algorithm
from smallaxe.training.base import Classifier, Model

__all__ = [
    "random_forest",
    "xgboost",
    "lightgbm",
    "catboost",
    "load",
    "algorithms",
    "available_models",
]

_TASK_TYPE = "classification"


def _create(name: str, task: str, params: Dict[str, Any]) -> Classifier:
    model = Classifier(name, task=task)
    if params:
        model.set_params(**params)
    return model


def random_forest(task: str = "binary", **params: Any) -> Classifier:
    """Create a Random Forest classifier.

    Args:
        task: The classification task: 'binary' (default) or 'multiclass'.
        **params: Parameters to pass to the model. Common parameters include:
            - n_estimators: Number of trees in the forest (default: 20)
            - max_depth: Maximum depth of each tree (default: 5)
            - max_bins: Maximum number of bins for discretizing features (default: 32)
            - min_instances_per_node: Minimum instances per node (default: 1)
            - min_info_gain: Minimum information gain for a split (default: 0.0)
            - subsampling_rate: Fraction of data for training each tree (default: 1.0)
            - feature_subset_strategy: Strategy for selecting features (default: 'auto')
            - seed: Random seed for reproducibility (default: None)

    Returns:
        A configured Random Forest classifier.

    Raises:
        ValidationError: If the task is invalid, or a parameter is unknown or has the
            wrong type.

    Example:
        >>> model = classifiers.random_forest(n_estimators=100, max_depth=10)
        >>> model.fit(dataframe, label_col='label', feature_cols=['f1', 'f2'])
    """
    return _create("random_forest", task, params)


def xgboost(task: str = "binary", **params: Any) -> Classifier:
    """Create an XGBoost classifier.

    Note:
        This requires the xgboost package to be installed.
        Install with: pip install smallaxe[xgboost]

    Args:
        task: The classification task: 'binary' (default) or 'multiclass'.
        **params: Parameters to pass to the model. Common parameters include:
            - n_estimators: Number of boosting rounds (default: 100)
            - max_depth: Maximum depth of each tree (default: 6)
            - learning_rate: Step size shrinkage (default: 0.3)
            - subsample: Fraction of samples for training each tree (default: 1.0)
            - colsample_bytree: Fraction of features for training each tree (default: 1.0)
            - min_child_weight: Minimum sum of instance weight in a child (default: 1)
            - reg_alpha: L1 regularization term (default: 0.0)
            - reg_lambda: L2 regularization term (default: 1.0)
            - gamma: Minimum loss reduction for a split (default: 0.0)
            - seed: Random seed for reproducibility (default: None)

    Returns:
        A configured XGBoost classifier.

    Raises:
        DependencyError: If xgboost is not installed.
        ValidationError: If the task is invalid, or a parameter is unknown or has the
            wrong type.

    Example:
        >>> model = classifiers.xgboost(n_estimators=100, max_depth=6)
        >>> model.fit(dataframe, label_col='label', feature_cols=['f1', 'f2'])
    """
    return _create("xgboost", task, params)


def lightgbm(task: str = "binary", **params: Any) -> Classifier:
    """Create a LightGBM classifier.

    Note:
        This requires SynapseML LightGBM support to be installed and
        configured for the active Spark session.
        Install with: pip install smallaxe[lightgbm]

    Args:
        task: The classification task: 'binary' (default) or 'multiclass'.
        **params: Parameters to pass to the model. Common parameters include:
            - n_estimators: Number of boosting iterations (default: 100)
            - max_depth: Maximum depth of each tree (default: -1)
            - learning_rate: Boosting learning rate (default: 0.1)
            - num_leaves: Maximum number of leaves in one tree (default: 31)
            - seed: Random seed for reproducibility (default: None)

    Returns:
        A configured LightGBM classifier.

    Raises:
        DependencyError: If SynapseML LightGBM support is not installed.
        ValidationError: If the task is invalid, or a parameter is unknown or has the
            wrong type.
    """
    return _create("lightgbm", task, params)


def catboost(task: str = "binary", **params: Any) -> Classifier:
    """Create a CatBoost classifier.

    Note:
        This requires CatBoost Spark support to be installed and configured
        for the active Spark session.
        Install with: pip install smallaxe[catboost]

    Args:
        task: The classification task: 'binary' (default) or 'multiclass'.
        **params: Parameters to pass to the model. Common parameters include:
            - n_estimators: Number of boosting iterations (default: 100)
            - max_depth: Maximum tree depth (default: 6)
            - learning_rate: Boosting learning rate (default: 0.03)
            - seed: Random seed for reproducibility (default: None)

    Returns:
        A configured CatBoost classifier.

    Raises:
        DependencyError: If CatBoost Spark support is not installed.
        ValidationError: If the task is invalid, or a parameter is unknown or has the
            wrong type.
    """
    return _create("catboost", task, params)


def load(path: str) -> Classifier:
    """Load a classifier from disk.

    The algorithm is read from the saved metadata, so any classifier saved by
    smallaxe (including 0.8.x artifacts) loads through this one function.

    Args:
        path: Directory path where the model was saved.

    Returns:
        The loaded classifier.

    Raises:
        ValidationError: If the path or its ``metadata.json`` does not exist, the
            artifact is invalid (see :meth:`~smallaxe.training.base.Model.load`), or the
            saved model is not a classifier.
        DependencyError: If the saved model's algorithm is not installed.

    Example:
        >>> model = classifiers.random_forest(n_estimators=100)
        >>> model.fit(dataframe, label_col='label', feature_cols=['f1', 'f2'])
        >>> model.save('/path/to/model')
        >>>
        >>> loaded_model = classifiers.load('/path/to/model')
        >>> predictions = loaded_model.predict(dataframe)
    """
    model = Model.load(path)
    if not isinstance(model, Classifier):
        saved = algorithm._legacy_class_name(model.algorithm.name, model.task_type)
        raise ValidationError(
            f"Model type '{saved}' is not a supported classifier: {path!r} holds a "
            "regressor. Load it with regressors.load() or smallaxe.load_model()."
        )
    return model


def algorithms() -> List[str]:
    """The algorithm names, one per constructor function, whether installed or not.

    Use :func:`available_models` to see which are installed.

    Returns:
        ``["random_forest", "xgboost", "lightgbm", "catboost"]``.
    """
    return algorithm.names()


def available_models() -> Dict[str, Dict[str, Any]]:
    """Report installed and unavailable classifiers with install hints.

    Returns:
        Dictionary keyed by algorithm (constructor function) name. Each entry holds
        ``available``, the optional ``dependency`` package, and ``install_hint`` when
        the algorithm is not installed.

    Example:
        >>> classifiers.available_models()["random_forest"]
        {'available': True, 'dependency': None, 'install_hint': None}
    """
    report = {}
    for name in algorithm.names():
        record = algorithm.get(name)
        available = record.available()
        dependency = record.dependency
        report[name] = {
            "available": available,
            "dependency": dependency.package if dependency else None,
            "install_hint": None if available or dependency is None else dependency.install_hint,
        }
    return report


def _warn_static(method: str, replacement: str) -> None:
    _deprecation.warn(
        f"Classifiers.{method}() is deprecated and will be removed in smallaxe 1.0. Use "
        f"{replacement} (from smallaxe.training import classifiers) instead."
    )


class Classifiers:
    """Deprecated: use the functions of :mod:`smallaxe.training.classifiers`. Removed in 1.0.

    Every static method warns once and calls the module function that replaces it:
    ``Classifiers.xgboost(...)`` is ``classifiers.xgboost(...)``.
    """

    @staticmethod
    def random_forest(task: str = "binary", **kwargs: Any) -> Classifier:
        """Deprecated: use :func:`smallaxe.training.classifiers.random_forest`."""
        _warn_static("random_forest", "classifiers.random_forest()")
        return random_forest(task=task, **kwargs)

    @staticmethod
    def xgboost(task: str = "binary", **kwargs: Any) -> Classifier:
        """Deprecated: use :func:`smallaxe.training.classifiers.xgboost`."""
        _warn_static("xgboost", "classifiers.xgboost()")
        return xgboost(task=task, **kwargs)

    @staticmethod
    def lightgbm(task: str = "binary", **kwargs: Any) -> Classifier:
        """Deprecated: use :func:`smallaxe.training.classifiers.lightgbm`."""
        _warn_static("lightgbm", "classifiers.lightgbm()")
        return lightgbm(task=task, **kwargs)

    @staticmethod
    def catboost(task: str = "binary", **kwargs: Any) -> Classifier:
        """Deprecated: use :func:`smallaxe.training.classifiers.catboost`."""
        _warn_static("catboost", "classifiers.catboost()")
        return catboost(task=task, **kwargs)

    @staticmethod
    def load(path: str) -> Classifier:
        """Deprecated: use :func:`smallaxe.training.classifiers.load`."""
        _warn_static("load", "classifiers.load()")
        return load(path)

    @staticmethod
    def list_models() -> List[str]:
        """Deprecated: use :func:`~smallaxe.training.classifiers.algorithms` and
        :func:`~smallaxe.training.classifiers.available_models`.

        Returns:
            The 0.8.x class names of the classifiers whose dependencies are installed.
        """
        _warn_static(
            "list_models",
            "classifiers.algorithms() (and classifiers.available_models() for what is "
            "installed)",
        )
        return [
            algorithm._legacy_class_name(name, _TASK_TYPE)
            for name in algorithm.names()
            if algorithm.get(name).available()
        ]

    @staticmethod
    def available_models() -> Dict[str, Dict[str, Any]]:
        """Deprecated: use :func:`smallaxe.training.classifiers.available_models`.

        Returns:
            The module function's report, each entry led by ``class_name``, the 0.8.x
            class name.
        """
        _warn_static("available_models", "classifiers.available_models()")
        return {
            name: {"class_name": algorithm._legacy_class_name(name, _TASK_TYPE), **entry}
            for name, entry in available_models().items()
        }
