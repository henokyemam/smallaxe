"""XGBoost via ``xgboost.spark`` (``pip install smallaxe[xgboost]``)."""

from typing import Any, List, Optional

from smallaxe import _deprecation
from smallaxe.training.algorithm import Algorithm, Dependency, Param
from smallaxe.training.base import Classifier, Regressor, _warn_deprecated_alias


def _importances(model: Any, n_features: int) -> Optional[List[float]]:
    """Total gain per feature; features the booster never split on score 0."""
    scores = model.get_booster().get_score(importance_type="gain")
    return [float(scores.get(f"f{i}", 0.0)) for i in range(n_features)]


ALGORITHM = Algorithm(
    name="xgboost",
    params=(
        # xgboost.spark reads the round count from ``n_estimators``. smallaxe <= 0.10 sent
        # ``num_round``, which xgboost.spark files under ``arbitrary_params_dict`` and ignores.
        Param("n_estimators", "Number of boosting rounds", 100),
        Param("max_depth", "Maximum depth of each tree", 6),
        Param("learning_rate", "Step size shrinkage used in update to prevent overfitting", 0.3),
        Param("subsample", "Fraction of samples used for training each tree", 1.0),
        Param("colsample_bytree", "Fraction of features used for training each tree", 1.0),
        Param("min_child_weight", "Minimum sum of instance weight needed in a child", 1),
        Param("reg_alpha", "L1 regularization term on weights", 0.0),
        Param("reg_lambda", "L2 regularization term on weights", 1.0),
        Param("gamma", "Minimum loss reduction required to make a further partition", 0.0),
        # Not a declared xgboost.spark param: it reaches the booster through
        # ``arbitrary_params_dict`` as XGBoost's native ``seed``, which takes effect.
        Param("seed", "Random seed for reproducibility", None),
    ),
    estimators={
        "simple_regression": "xgboost.spark.SparkXGBRegressor",
        "binary": "xgboost.spark.SparkXGBClassifier",
        "multiclass": "xgboost.spark.SparkXGBClassifier",
    },
    models={
        "simple_regression": "xgboost.spark.SparkXGBRegressorModel",
        "binary": "xgboost.spark.SparkXGBClassifierModel",
        "multiclass": "xgboost.spark.SparkXGBClassifierModel",
    },
    cols={
        "features": "features_col",
        "label": "label_col",
        "prediction": "prediction_col",
        "probability": "probability_col",
        "raw_prediction": "raw_prediction_col",
    },
    importances=_importances,
    dependency=Dependency("xgboost", "pip install smallaxe[xgboost]"),
    # xgboost.spark trains with ``missing=nan`` and scores every row.
    accepts_missing=True,
)


# Deprecated public names, removed in smallaxe 1.0. ``XGBOOST_AVAILABLE`` is probed on every
# read, so importing this module imports no optional package.
_MOVED = {
    "XGBOOST_AVAILABLE": (
        "Use smallaxe.training.algorithm.get('xgboost').available() or "
        "regressors.available_models() instead.",
        lambda: ALGORITHM.available(),
    ),
}
__getattr__ = _deprecation.module_getattr(__name__, _MOVED)
__dir__ = _deprecation.module_dir(globals(), _MOVED)


class XGBoostRegressor(Regressor):
    """Deprecated alias; use ``regressors.xgboost()``."""

    _alias_algorithm = "xgboost"

    def __init__(self, task: str = "simple_regression") -> None:
        _warn_deprecated_alias("XGBoostRegressor", "regressors.xgboost()")
        super().__init__(ALGORITHM, task)


class XGBoostClassifier(Classifier):
    """Deprecated alias; use ``classifiers.xgboost()``."""

    _alias_algorithm = "xgboost"

    def __init__(self, task: str = "binary") -> None:
        _warn_deprecated_alias("XGBoostClassifier", "classifiers.xgboost()")
        super().__init__(ALGORITHM, task)
