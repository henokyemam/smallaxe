"""CatBoost models for regression and classification."""

import shutil
import tempfile
from typing import Any, Dict, List, Optional

from pyspark.sql import DataFrame

from smallaxe.exceptions import DependencyError
from smallaxe.training.base import BaseClassifier, BaseRegressor

CATBOOST_AVAILABLE = False
SparkCatBoostRegressor = None
SparkCatBoostRegressionModel = None
SparkCatBoostClassifier = None
SparkCatBoostClassificationModel = None


def _load_catboost_spark() -> bool:
    """Load CatBoost Spark classes if Spark has made them importable."""
    global CATBOOST_AVAILABLE
    global SparkCatBoostRegressor
    global SparkCatBoostRegressionModel
    global SparkCatBoostClassifier
    global SparkCatBoostClassificationModel

    if CATBOOST_AVAILABLE:
        return True

    try:
        from catboost_spark import (
            CatBoostClassificationModel,
            CatBoostClassifier,
            CatBoostRegressionModel,
            CatBoostRegressor,
        )
    except ImportError:
        return False

    SparkCatBoostRegressor = CatBoostRegressor
    SparkCatBoostRegressionModel = CatBoostRegressionModel
    SparkCatBoostClassifier = CatBoostClassifier
    SparkCatBoostClassificationModel = CatBoostClassificationModel
    CATBOOST_AVAILABLE = True
    return True


_load_catboost_spark()


def _check_catboost_available() -> None:
    """Check if CatBoost Spark support is available."""
    if not _load_catboost_spark():
        raise DependencyError(
            package="catboost_spark",
            install_command=(
                "pip install smallaxe[catboost] and configure Spark with "
                "ai.catboost:catboost-spark_3.5_2.12:1.2.10"
            ),
        )


def is_catboost_available() -> bool:
    """Return whether CatBoost Spark support is currently importable."""
    return _load_catboost_spark()


def catboost_install_hint() -> str:
    """Return the install and Spark package hint for CatBoost support."""
    return (
        "pip install smallaxe[catboost] and configure Spark with "
        "ai.catboost:catboost-spark_3.5_2.12:1.2.10"
    )


class CatBoostRegressor(BaseRegressor):
    """CatBoost regressor for regression tasks.

    This class wraps CatBoost for Spark's CatBoostRegressor to provide the
    same smallaxe fit/predict/save/load interface as the other Spark-backed
    regressors.
    """

    def __init__(self, task: str = "simple_regression") -> None:
        """Initialize the CatBoost regressor."""
        _check_catboost_available()
        super().__init__(task)

    @property
    def params(self) -> Dict[str, str]:
        """Get parameter descriptions."""
        return {
            "n_estimators": "Number of boosting iterations",
            "max_depth": "Maximum tree depth",
            "learning_rate": "Boosting learning rate",
            "subsample": "Sample rate for bagging",
            "l2_leaf_reg": "L2 regularization coefficient",
            "random_strength": "Amount of randomness used when scoring splits",
            "one_hot_max_size": "Maximum categorical cardinality for one-hot encoding",
            "allow_writing_files": "Whether CatBoost may write training artifacts",
            "train_dir": "Directory for CatBoost training artifacts",
            "seed": "Random seed for reproducibility",
        }

    @property
    def default_params(self) -> Dict[str, Any]:
        """Get default parameter values."""
        return {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.03,
            "subsample": None,
            "l2_leaf_reg": 3.0,
            "random_strength": 1.0,
            "one_hot_max_size": None,
            "allow_writing_files": False,
            "train_dir": None,
            "seed": None,
        }

    def _catboost_params(
        self,
        label_col: Optional[str] = None,
        train_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Translate smallaxe parameter names to CatBoost Spark parameter names."""
        params = {
            "iterations": self.get_param("n_estimators"),
            "depth": self.get_param("max_depth"),
            "learningRate": self.get_param("learning_rate"),
            "l2LeafReg": self.get_param("l2_leaf_reg"),
            "randomStrength": self.get_param("random_strength"),
            "lossFunction": "RMSE",
            "allowWritingFiles": self.get_param("allow_writing_files"),
            "featuresCol": self.FEATURES_COL,
            "predictionCol": self.PREDICTION_COL,
        }
        if label_col is not None:
            params["labelCol"] = label_col

        configured_train_dir = train_dir or self.get_param("train_dir")
        if configured_train_dir is not None:
            params["trainDir"] = configured_train_dir

        optional_params = {
            "subsample": self.get_param("subsample"),
            "oneHotMaxSize": self.get_param("one_hot_max_size"),
            "randomSeed": self.get_param("seed"),
        }
        params.update({name: value for name, value in optional_params.items() if value is not None})
        return params

    def _uses_constructor_col_params(self) -> bool:
        return True

    def _create_spark_estimator(
        self,
        features_col: Optional[str] = None,
        label_col: Optional[str] = None,
        prediction_col: Optional[str] = None,
    ) -> Any:
        """Create the underlying CatBoost Spark regressor."""
        return SparkCatBoostRegressor(**self._catboost_params(label_col=label_col))

    def _fit_spark_model(
        self,
        df: DataFrame,
        label_col: str,
        feature_cols: List[str],
    ) -> Any:
        """Fit the CatBoost Spark regressor."""
        df_with_features = self._assemble_features(df, feature_cols)
        temp_train_dir = None
        if self.get_param("train_dir") is None:
            temp_train_dir = tempfile.mkdtemp(prefix="smallaxe_catboost_")
        estimator = SparkCatBoostRegressor(
            **self._catboost_params(label_col, train_dir=temp_train_dir)
        )

        self._feature_cols = feature_cols
        self._label_col = label_col
        try:
            self._spark_model = estimator.fit(df_with_features)
        finally:
            if temp_train_dir is not None:
                shutil.rmtree(temp_train_dir, ignore_errors=True)

        return self._spark_model

    def _load_artifacts(self, path: str) -> None:
        """Load the CatBoost Spark model from disk."""
        self._load_spark_model(path, SparkCatBoostRegressionModel)


class CatBoostClassifier(BaseClassifier):
    """CatBoost classifier for binary and multiclass classification tasks."""

    def __init__(self, task: str = "binary") -> None:
        """Initialize the CatBoost classifier."""
        _check_catboost_available()
        super().__init__(task)

    @property
    def params(self) -> Dict[str, str]:
        """Get parameter descriptions."""
        return {
            "n_estimators": "Number of boosting iterations",
            "max_depth": "Maximum tree depth",
            "learning_rate": "Boosting learning rate",
            "subsample": "Sample rate for bagging",
            "l2_leaf_reg": "L2 regularization coefficient",
            "random_strength": "Amount of randomness used when scoring splits",
            "one_hot_max_size": "Maximum categorical cardinality for one-hot encoding",
            "scale_pos_weight": "Class 1 weight multiplier for binary classification",
            "allow_writing_files": "Whether CatBoost may write training artifacts",
            "train_dir": "Directory for CatBoost training artifacts",
            "seed": "Random seed for reproducibility",
        }

    @property
    def default_params(self) -> Dict[str, Any]:
        """Get default parameter values."""
        return {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.03,
            "subsample": None,
            "l2_leaf_reg": 3.0,
            "random_strength": 1.0,
            "one_hot_max_size": None,
            "scale_pos_weight": None,
            "allow_writing_files": False,
            "train_dir": None,
            "seed": None,
        }

    def _catboost_params(
        self,
        label_col: Optional[str] = None,
        train_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Translate smallaxe parameter names to CatBoost Spark parameter names."""
        loss_function = "Logloss" if self.task == "binary" else "MultiClass"
        params = {
            "iterations": self.get_param("n_estimators"),
            "depth": self.get_param("max_depth"),
            "learningRate": self.get_param("learning_rate"),
            "l2LeafReg": self.get_param("l2_leaf_reg"),
            "randomStrength": self.get_param("random_strength"),
            "lossFunction": loss_function,
            "allowWritingFiles": self.get_param("allow_writing_files"),
            "featuresCol": self.FEATURES_COL,
            "predictionCol": self.PREDICTION_COL,
            "probabilityCol": self.PROBABILITY_COL,
            "rawPredictionCol": self.RAW_PREDICTION_COL,
        }
        if label_col is not None:
            params["labelCol"] = label_col

        configured_train_dir = train_dir or self.get_param("train_dir")
        if configured_train_dir is not None:
            params["trainDir"] = configured_train_dir

        optional_params = {
            "subsample": self.get_param("subsample"),
            "oneHotMaxSize": self.get_param("one_hot_max_size"),
            "scalePosWeight": self.get_param("scale_pos_weight"),
            "randomSeed": self.get_param("seed"),
        }
        params.update({name: value for name, value in optional_params.items() if value is not None})
        return params

    def _uses_constructor_col_params(self) -> bool:
        return True

    def _create_spark_estimator(
        self,
        features_col: Optional[str] = None,
        label_col: Optional[str] = None,
        prediction_col: Optional[str] = None,
    ) -> Any:
        """Create the underlying CatBoost Spark classifier."""
        return SparkCatBoostClassifier(**self._catboost_params(label_col=label_col))

    def _fit_spark_model(
        self,
        df: DataFrame,
        label_col: str,
        feature_cols: List[str],
    ) -> Any:
        """Fit the CatBoost Spark classifier."""
        df_with_features = self._assemble_features(df, feature_cols)
        temp_train_dir = None
        if self.get_param("train_dir") is None:
            temp_train_dir = tempfile.mkdtemp(prefix="smallaxe_catboost_")
        estimator = SparkCatBoostClassifier(
            **self._catboost_params(label_col, train_dir=temp_train_dir)
        )

        self._feature_cols = feature_cols
        self._label_col = label_col
        try:
            self._spark_model = estimator.fit(df_with_features)
        finally:
            if temp_train_dir is not None:
                shutil.rmtree(temp_train_dir, ignore_errors=True)

        return self._spark_model

    def _load_artifacts(self, path: str) -> None:
        """Load the CatBoost Spark model from disk."""
        self._load_spark_model(path, SparkCatBoostClassificationModel)
