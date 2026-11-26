"""Regression models: one constructor function per algorithm, plus load.

Each function returns a :class:`~smallaxe.training.Regressor` bound to one algorithm,
so no algorithm-specific class needs to be imported. These functions are the only
factories.

Example:
    >>> from smallaxe.training import regressors
    >>>
    >>> model = regressors.random_forest(n_estimators=100, max_depth=10)
    >>> model.fit(dataframe, label_col='target', feature_cols=['f1', 'f2'])
    >>> model.save('/path/to/model')
    >>> loaded_model = regressors.load('/path/to/model')
"""

from typing import Any, Dict, List

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


def _create(name: str, params: Dict[str, Any]) -> Regressor:
    """A regressor for algorithm ``name`` with ``params`` set."""
    return Regressor(name).set_params(**params)


def random_forest(**params: Any) -> Regressor:
    """Create an unfitted Random Forest regressor.

    Random Forest is PySpark MLlib's own estimator and needs no optional package.

    Args:
        **params: Parameter values by name; every parameter left out keeps its default.
            ``model.params`` describes them and ``model.default_params`` lists the
            defaults. The parameters are:

            - n_estimators: Number of trees in the forest (default: 20)
            - max_depth: Maximum depth of each tree (0 = a single leaf node; at most 30)
              (default: 5)
            - max_bins: Maximum number of bins for discretizing continuous features (default: 32)
            - min_instances_per_node: Minimum number of instances per node (default: 1)
            - min_info_gain: Minimum information gain for a split (default: 0.0)
            - subsampling_rate: Fraction of data used for training each tree (default: 1.0)
            - feature_subset_strategy: Strategy for selecting features: 'auto', 'all', 'sqrt',
              'log2', 'onethird' (default: 'auto')
            - seed: Random seed for reproducibility (default: None)

    Returns:
        A :class:`~smallaxe.training.Regressor` bound to the ``random_forest`` algorithm.

    Raises:
        ValidationError: If a parameter is unknown or its value has the wrong type.

    Example:
        >>> model = regressors.random_forest(n_estimators=100, max_depth=10)
        >>> model.fit(dataframe, label_col='target')
    """
    return _create("random_forest", params)


def xgboost(**params: Any) -> Regressor:
    """Create an unfitted XGBoost regressor.

    Needs the xgboost package: ``pip install smallaxe[xgboost]``.

    Args:
        **params: Parameter values by name; every parameter left out keeps its default.
            ``model.params`` describes them and ``model.default_params`` lists the
            defaults. The parameters are:

            - n_estimators: Number of boosting rounds (default: 100)
            - max_depth: Maximum depth of each tree (default: 6)
            - learning_rate: Step size shrinkage used in update to prevent overfitting
              (default: 0.3)
            - subsample: Fraction of samples used for training each tree (default: 1.0)
            - colsample_bytree: Fraction of features used for training each tree (default: 1.0)
            - min_child_weight: Minimum sum of instance weight needed in a child (default: 1)
            - reg_alpha: L1 regularization term on weights (default: 0.0)
            - reg_lambda: L2 regularization term on weights (default: 1.0)
            - gamma: Minimum loss reduction required to make a further partition (default: 0.0)
            - seed: Random seed for reproducibility (default: None)

    Returns:
        A :class:`~smallaxe.training.Regressor` bound to the ``xgboost`` algorithm.

    Raises:
        DependencyError: If xgboost is not installed or cannot be imported.
        ValidationError: If a parameter is unknown or its value has the wrong type.

    Example:
        >>> model = regressors.xgboost(n_estimators=200, learning_rate=0.1)
        >>> model.fit(dataframe, label_col='target')
    """
    return _create("xgboost", params)


def lightgbm(**params: Any) -> Regressor:
    """Create an unfitted LightGBM regressor.

    Needs SynapseML's LightGBM: ``pip install smallaxe[lightgbm]`` and the SynapseML Spark
    package on the session's classpath.

    Args:
        **params: Parameter values by name; every parameter left out keeps its default.
            ``model.params`` describes them and ``model.default_params`` lists the
            defaults. The parameters are:

            - n_estimators: Number of boosting iterations (default: 100)
            - max_depth: Maximum depth of each tree (-1 for no limit) (default: -1)
            - learning_rate: Boosting learning rate (default: 0.1)
            - num_leaves: Maximum number of leaves in one tree (default: 31)
            - min_data_in_leaf: Minimum number of data points in a leaf (default: 20)
            - feature_fraction: Fraction of features used for training each tree (default: 1.0)
            - bagging_fraction: Fraction of data used for training each tree (default: 1.0)
            - bagging_freq: Frequency for bagging (0 means disable bagging) (default: 0)
            - lambda_l1: L1 regularization term on weights (default: 0.0)
            - lambda_l2: L2 regularization term on weights (default: 0.0)
            - seed: Random seed for reproducibility (default: None)

    Returns:
        A :class:`~smallaxe.training.Regressor` bound to the ``lightgbm`` algorithm.

    Raises:
        DependencyError: If synapseml is not installed or cannot be imported.
        ValidationError: If a parameter is unknown or its value has the wrong type.

    Example:
        >>> model = regressors.lightgbm(n_estimators=200, num_leaves=63)
        >>> model.fit(dataframe, label_col='target')
    """
    return _create("lightgbm", params)


def catboost(**params: Any) -> Regressor:
    """Create an unfitted CatBoost regressor.

    Needs catboost-spark: ``pip install smallaxe[catboost]`` and the catboost-spark Spark
    package on the session's classpath.

    Args:
        **params: Parameter values by name; every parameter left out keeps its default.
            ``model.params`` describes them and ``model.default_params`` lists the
            defaults. The parameters are:

            - n_estimators: Number of boosting iterations (default: 100)
            - max_depth: Maximum tree depth (default: 6)
            - learning_rate: Boosting learning rate (default: 0.03)
            - subsample: Sample rate for bagging (default: None)
            - l2_leaf_reg: L2 regularization coefficient (default: 3.0)
            - random_strength: Amount of randomness used when scoring splits (default: 1.0)
            - one_hot_max_size: Maximum categorical cardinality for one-hot encoding (default: None)
            - allow_writing_files: Whether CatBoost may write training artifacts (default: False)
            - train_dir: Directory for CatBoost training artifacts (default: None)
            - seed: Random seed for reproducibility (default: None)

    Returns:
        A :class:`~smallaxe.training.Regressor` bound to the ``catboost`` algorithm.

    Raises:
        DependencyError: If catboost_spark is not installed or cannot be imported.
        ValidationError: If a parameter is unknown or its value has the wrong type.

    Example:
        >>> model = regressors.catboost(n_estimators=200, max_depth=8)
        >>> model.fit(dataframe, label_col='target')
    """
    return _create("catboost", params)


def load(path: str) -> Regressor:
    """Load a saved regressor.

    The algorithm is read from the saved metadata, so every regressor saved by smallaxe
    (0.8.1 onward) loads through this one function.

    Args:
        path: The artifact directory (local, ``dbfs:/``, ``s3://``, ``hdfs://``, ...).

    Returns:
        The fitted regressor.

    Raises:
        ValidationError: If the path or its ``metadata.json`` does not exist, the
            artifact is invalid (see :meth:`~smallaxe.training.Model.load`), or it holds
            a classifier.
        DependencyError: If the saved model's algorithm is not installed.

    Example:
        >>> model = regressors.load('/path/to/model')
        >>> predictions = model.predict(dataframe)
    """
    model = Model.load(path)
    if not isinstance(model, Regressor):
        raise ValidationError(
            f"Cannot load a regressor: {path!r} holds a {model.algorithm.name!r} classifier "
            f"(task {model.task!r}). Load it with classifiers.load() or smallaxe.load_model()."
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
    """Report which algorithms are installed, with install hints for the others.

    Returns:
        A dict keyed by algorithm (constructor function) name. Each entry holds
        ``available`` (bool), ``dependency`` (the optional package, or None) and
        ``install_hint`` (how to install it when it is missing, else None).

    Example:
        >>> regressors.available_models()["random_forest"]
        {'available': True, 'dependency': None, 'install_hint': None}
    """
    return algorithm._availability()
