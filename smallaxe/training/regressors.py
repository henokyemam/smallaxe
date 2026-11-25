"""Regression models: one constructor function per algorithm, plus load.

Example:
    >>> from smallaxe.training import regressors
    >>>
    >>> model = regressors.random_forest(n_estimators=100, max_depth=10)
    >>> model.fit(dataframe, label_col='target', feature_cols=['f1', 'f2'])
    >>> model.save('/path/to/model')
    >>> loaded_model = regressors.load('/path/to/model')

Each function returns a :class:`~smallaxe.training.base.Regressor` bound to one
algorithm, so no algorithm-specific class needs to be imported. The ``Regressors``
class is the deprecated 0.x spelling of these functions and is removed in smallaxe 1.0.
"""

from typing import Any, Dict, List

from smallaxe import _deprecation
from smallaxe.exceptions import ValidationError
from smallaxe.training import algorithm
from smallaxe.training.base import Model, Regressor

__all__ = [
    "random_forest",
    "xgboost",
    "lightgbm",
    "catboost",
    "load",
    "algorithms",
    "available_models",
]

_TASK_TYPE = "regression"


def _create(name: str, params: Dict[str, Any]) -> Regressor:
    model = Regressor(name)
    if params:
        model.set_params(**params)
    return model


def random_forest(**params: Any) -> Regressor:
    """Create a Random Forest regressor.

    Args:
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
        A configured Random Forest regressor.

    Raises:
        ValidationError: If a parameter is unknown or has the wrong type.

    Example:
        >>> model = regressors.random_forest(n_estimators=100, max_depth=10)
        >>> model.fit(dataframe, label_col='target', feature_cols=['f1', 'f2'])
    """
    return _create("random_forest", params)


def xgboost(**params: Any) -> Regressor:
    """Create an XGBoost regressor.

    Note:
        This requires the xgboost package to be installed.
        Install with: pip install smallaxe[xgboost]

    Args:
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
        A configured XGBoost regressor.

    Raises:
        DependencyError: If xgboost is not installed.
        ValidationError: If a parameter is unknown or has the wrong type.

    Example:
        >>> model = regressors.xgboost(n_estimators=100, max_depth=6)
        >>> model.fit(dataframe, label_col='target', feature_cols=['f1', 'f2'])
    """
    return _create("xgboost", params)


def lightgbm(**params: Any) -> Regressor:
    """Create a LightGBM regressor.

    Note:
        This requires SynapseML LightGBM support to be installed and
        configured for the active Spark session.
        Install with: pip install smallaxe[lightgbm]

    Args:
        **params: Parameters to pass to the model. Common parameters include:
            - n_estimators: Number of boosting iterations (default: 100)
            - max_depth: Maximum depth of each tree (default: -1)
            - learning_rate: Boosting learning rate (default: 0.1)
            - num_leaves: Maximum number of leaves in one tree (default: 31)
            - seed: Random seed for reproducibility (default: None)

    Returns:
        A configured LightGBM regressor.

    Raises:
        DependencyError: If SynapseML LightGBM support is not installed.
        ValidationError: If a parameter is unknown or has the wrong type.
    """
    return _create("lightgbm", params)


def catboost(**params: Any) -> Regressor:
    """Create a CatBoost regressor.

    Note:
        This requires CatBoost Spark support to be installed and configured
        for the active Spark session.
        Install with: pip install smallaxe[catboost]

    Args:
        **params: Parameters to pass to the model. Common parameters include:
            - n_estimators: Number of boosting iterations (default: 100)
            - max_depth: Maximum tree depth (default: 6)
            - learning_rate: Boosting learning rate (default: 0.03)
            - seed: Random seed for reproducibility (default: None)

    Returns:
        A configured CatBoost regressor.

    Raises:
        DependencyError: If CatBoost Spark support is not installed.
        ValidationError: If a parameter is unknown or has the wrong type.
    """
    return _create("catboost", params)


def load(path: str) -> Regressor:
    """Load a regressor from disk.

    The algorithm is read from the saved metadata, so any regressor saved by
    smallaxe (including 0.8.x artifacts) loads through this one function.

    Args:
        path: Directory path where the model was saved.

    Returns:
        The loaded regressor.

    Raises:
        ValidationError: If the path or its ``metadata.json`` does not exist, the
            artifact is invalid (see :meth:`~smallaxe.training.base.Model.load`), or the
            saved model is not a regressor.
        DependencyError: If the saved model's algorithm is not installed.

    Example:
        >>> model = regressors.random_forest(n_estimators=100)
        >>> model.fit(dataframe, label_col='target', feature_cols=['f1', 'f2'])
        >>> model.save('/path/to/model')
        >>>
        >>> loaded_model = regressors.load('/path/to/model')
        >>> predictions = loaded_model.predict(dataframe)
    """
    model = Model.load(path)
    if not isinstance(model, Regressor):
        saved = algorithm._legacy_class_name(model.algorithm.name, model.task_type)
        raise ValidationError(
            f"Model type '{saved}' is not a supported regressor: {path!r} holds a "
            f"{model.task} classifier. Load it with classifiers.load() or "
            "smallaxe.load_model()."
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
    """Report installed and unavailable regressors with install hints.

    Returns:
        Dictionary keyed by algorithm (constructor function) name. Each entry holds
        ``available``, the optional ``dependency`` package, and ``install_hint`` when
        the algorithm is not installed.

    Example:
        >>> regressors.available_models()["random_forest"]
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
        f"Regressors.{method}() is deprecated and will be removed in smallaxe 1.0. Use "
        f"{replacement} (from smallaxe.training import regressors) instead."
    )


class Regressors:
    """Deprecated: use the functions of :mod:`smallaxe.training.regressors`. Removed in 1.0.

    Every static method warns once and calls the module function that replaces it:
    ``Regressors.xgboost(...)`` is ``regressors.xgboost(...)``.
    """

    @staticmethod
    def random_forest(**kwargs: Any) -> Regressor:
        """Deprecated: use :func:`smallaxe.training.regressors.random_forest`."""
        _warn_static("random_forest", "regressors.random_forest()")
        return random_forest(**kwargs)

    @staticmethod
    def xgboost(**kwargs: Any) -> Regressor:
        """Deprecated: use :func:`smallaxe.training.regressors.xgboost`."""
        _warn_static("xgboost", "regressors.xgboost()")
        return xgboost(**kwargs)

    @staticmethod
    def lightgbm(**kwargs: Any) -> Regressor:
        """Deprecated: use :func:`smallaxe.training.regressors.lightgbm`."""
        _warn_static("lightgbm", "regressors.lightgbm()")
        return lightgbm(**kwargs)

    @staticmethod
    def catboost(**kwargs: Any) -> Regressor:
        """Deprecated: use :func:`smallaxe.training.regressors.catboost`."""
        _warn_static("catboost", "regressors.catboost()")
        return catboost(**kwargs)

    @staticmethod
    def load(path: str) -> Regressor:
        """Deprecated: use :func:`smallaxe.training.regressors.load`."""
        _warn_static("load", "regressors.load()")
        return load(path)

    @staticmethod
    def list_models() -> List[str]:
        """Deprecated: use :func:`~smallaxe.training.regressors.algorithms` and
        :func:`~smallaxe.training.regressors.available_models`.

        Returns:
            The 0.8.x class names of the regressors whose dependencies are installed.
        """
        _warn_static(
            "list_models",
            "regressors.algorithms() (and regressors.available_models() for what is " "installed)",
        )
        return [
            algorithm._legacy_class_name(name, _TASK_TYPE)
            for name in algorithm.names()
            if algorithm.get(name).available()
        ]

    @staticmethod
    def available_models() -> Dict[str, Dict[str, Any]]:
        """Deprecated: use :func:`smallaxe.training.regressors.available_models`.

        Returns:
            The module function's report, each entry led by ``class_name``, the 0.8.x
            class name.
        """
        _warn_static("available_models", "regressors.available_models()")
        return {
            name: {"class_name": algorithm._legacy_class_name(name, _TASK_TYPE), **entry}
            for name, entry in available_models().items()
        }
