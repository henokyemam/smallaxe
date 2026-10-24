"""CatBoost via ``catboost_spark`` (``pip install smallaxe[catboost]`` plus the Spark package).

``catboost_spark`` only becomes importable once the ``ai.catboost:catboost-spark``
JVM package is on the Spark classpath, so availability is re-probed on every call.
Training fails if executors join or leave mid-fit, so use a fixed-size cluster.
"""

import collections
import datetime
import enum
import shutil
import tempfile
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from smallaxe.training.algorithm import CLASSIFICATION_TASKS, Algorithm, Dependency, Param
from smallaxe.training.base import Classifier, Regressor, _warn_deprecated_alias

_INSTALL_HINT = (
    "pip install smallaxe[catboost] and configure Spark with "
    "ai.catboost:catboost-spark_3.5_2.12:1.2.10"
)


def catboost_install_hint() -> str:
    """Return the install and Spark package hint for CatBoost support."""
    return _INSTALL_HINT


def is_catboost_available() -> bool:
    """Return whether CatBoost Spark support is currently importable."""
    return ALGORITHM.available()


# Param value types that catboost_spark converts itself, through the ``_py2java`` it installs
# on ``pyspark.ml.wrapper`` (e.g. its ``timedelta`` timeout defaults -> ``java.time.Duration``).
_CATBOOST_CONVERTED_TYPES = (datetime.timedelta, enum.Enum, collections.OrderedDict)


def _patched_by_synapseml(fn: Any) -> bool:
    return getattr(fn, "__module__", "").startswith("synapse.")


def _catboost_from_java() -> Optional[Callable[[Any], Any]]:
    """Return catboost_spark's own ``JavaParams._from_java`` replacement, if available."""
    try:
        from catboost_spark import core
    except ImportError:
        return None
    fn = getattr(core, "_from_java_patched_for_catboost", None)
    return getattr(fn, "__func__", fn)


def _catboost_converters() -> Optional[Tuple[Callable[..., Any], Callable[..., Any]]]:
    """Return catboost_spark's ``(_py2java, _java2py)``, which know its enums and types."""
    try:
        from catboost_spark import core
    except ImportError:
        return None
    return core._py2java, core._java2py


def _is_catboost_java_object(java_obj: Any) -> bool:
    try:
        return bool(java_obj.getClass().getName().startswith("ai.catboost.spark."))
    except Exception:  # noqa: BLE001 - not a Java object, or the gateway is gone
        return False


def _ensure_synapseml_compat() -> None:
    """Keep CatBoost working when SynapseML (LightGBM) is imported in the same session.

    Both libraries monkeypatch PySpark's ``JavaParams`` / ``JavaWrapper`` process-wide, and
    SynapseML's patches win whenever it is imported after catboost_spark (as smallaxe does):

    - ``_make_java_param_pair`` pickles values it does not know, so CatBoost's ``timedelta``
      defaults reach the JVM as pickled objects and ``fit`` fails with a ``ClassCastException``.
    - ``_from_java`` only resolves ``pyspark.``/``synapse.ml.`` classes, so loading a saved
      CatBoost model fails.
    - ``_call_java`` pickles CatBoost's enum arguments (``EFstrType``), so model methods such as
      ``getFeatureImportance`` fail with a ``PickleException``.

    Route only CatBoost's values, stages, and model calls back to catboost_spark's converters;
    everything else still goes through SynapseML. Safe to call repeatedly; a no-op without
    SynapseML.
    """
    from pyspark import SparkContext
    from pyspark.ml import wrapper
    from pyspark.ml.wrapper import JavaParams, JavaWrapper

    make_pair = JavaParams._make_java_param_pair
    if _patched_by_synapseml(make_pair):

        def _make_java_param_pair(self: Any, param: Any, value: Any) -> Any:
            if isinstance(value, _CATBOOST_CONVERTED_TYPES):
                java_param = self._java_obj.getParam(self._resolveParam(param).name)
                return java_param.w(wrapper._py2java(SparkContext._active_spark_context, value))
            return make_pair(self, param, value)

        JavaParams._make_java_param_pair = _make_java_param_pair

    from_java = JavaParams._from_java
    catboost_from_java = _catboost_from_java()
    if _patched_by_synapseml(from_java) and catboost_from_java is not None:

        def _from_java(java_stage: Any) -> Any:
            if _is_catboost_java_object(java_stage):
                return catboost_from_java(java_stage)
            return from_java(java_stage)

        JavaParams._from_java = staticmethod(_from_java)

    call_java = JavaWrapper._call_java
    converters = _catboost_converters()
    if _patched_by_synapseml(call_java) and converters is not None:
        py2java, java2py = converters

        def _call_java(self: Any, name: str, *args: Any) -> Any:
            if not _is_catboost_java_object(self._java_obj):
                return call_java(self, name, *args)
            sc = SparkContext._active_spark_context
            java_args = [py2java(sc, arg) for arg in args]
            return java2py(sc, getattr(self._java_obj, name)(*java_args))

        JavaWrapper._call_java = _call_java


@contextmanager
def _training_dir(values: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """Give CatBoost a temporary ``trainDir`` for the duration of fit unless the user set one."""
    if values.get("train_dir") is not None:
        yield {}
        return
    path = tempfile.mkdtemp(prefix="smallaxe_catboost_")
    try:
        yield {"trainDir": path}
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _importances(model: Any, n_features: int) -> Optional[List[float]]:
    """CatBoost's PredictionValuesChange importances, which sum to 100."""
    return [float(v) for v in model.getFeatureImportance()]


ALGORITHM = Algorithm(
    name="catboost",
    params=(
        Param("n_estimators", "Number of boosting iterations", 100, spark="iterations"),
        Param("max_depth", "Maximum tree depth", 6, spark="depth"),
        Param("learning_rate", "Boosting learning rate", 0.03, spark="learningRate"),
        Param("subsample", "Sample rate for bagging", None),
        Param("l2_leaf_reg", "L2 regularization coefficient", 3.0, spark="l2LeafReg"),
        Param(
            "random_strength",
            "Amount of randomness used when scoring splits",
            1.0,
            spark="randomStrength",
        ),
        Param(
            "one_hot_max_size",
            "Maximum categorical cardinality for one-hot encoding",
            None,
            spark="oneHotMaxSize",
        ),
        Param(
            "scale_pos_weight",
            "Class 1 weight multiplier for binary classification",
            None,
            spark="scalePosWeight",
            tasks=CLASSIFICATION_TASKS,
        ),
        Param(
            "allow_writing_files",
            "Whether CatBoost may write training artifacts",
            False,
            spark="allowWritingFiles",
        ),
        Param("train_dir", "Directory for CatBoost training artifacts", None, spark="trainDir"),
        Param("seed", "Random seed for reproducibility", None, spark="randomSeed"),
    ),
    estimators={
        "simple_regression": "catboost_spark.CatBoostRegressor",
        "binary": "catboost_spark.CatBoostClassifier",
        "multiclass": "catboost_spark.CatBoostClassifier",
    },
    models={
        "simple_regression": "catboost_spark.CatBoostRegressionModel",
        "binary": "catboost_spark.CatBoostClassificationModel",
        "multiclass": "catboost_spark.CatBoostClassificationModel",
    },
    cols={
        "features": "featuresCol",
        "label": "labelCol",
        "prediction": "predictionCol",
        "probability": "probabilityCol",
        "raw_prediction": "rawPredictionCol",
    },
    task_params={
        "simple_regression": {"lossFunction": "RMSE"},
        "binary": {"lossFunction": "Logloss"},
        "multiclass": {"lossFunction": "MultiClass"},
    },
    importances=_importances,
    dependency=Dependency("catboost_spark", _INSTALL_HINT),
    accepts_raw_categoricals=True,
    prepare=_ensure_synapseml_compat,
    fit_context=_training_dir,
)

CATBOOST_AVAILABLE = ALGORITHM.available()


class CatBoostRegressor(Regressor):
    """Deprecated alias; use ``Regressors.catboost()``."""

    _alias_algorithm = "catboost"

    def __init__(self, task: str = "simple_regression") -> None:
        _warn_deprecated_alias("CatBoostRegressor", "Regressors.catboost()")
        super().__init__(ALGORITHM, task)


class CatBoostClassifier(Classifier):
    """Deprecated alias; use ``Classifiers.catboost()``."""

    _alias_algorithm = "catboost"

    def __init__(self, task: str = "binary") -> None:
        _warn_deprecated_alias("CatBoostClassifier", "Classifiers.catboost()")
        super().__init__(ALGORITHM, task)
