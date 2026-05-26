"""XGBoost models for regression and classification."""

from typing import Any, Dict, Optional

from smallaxe.exceptions import DependencyError
from smallaxe.training.base import BaseClassifier, BaseRegressor

# Check for xgboost availability
try:
    from xgboost.spark import (
        SparkXGBClassifier,
        SparkXGBClassifierModel,
        SparkXGBRegressor,
        SparkXGBRegressorModel,
    )

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    SparkXGBRegressor = None
    SparkXGBRegressorModel = None
    SparkXGBClassifier = None
    SparkXGBClassifierModel = None


def _check_xgboost_available() -> None:
    """Check if xgboost is available and raise DependencyError if not."""
    if not XGBOOST_AVAILABLE:
        raise DependencyError(
            package="xgboost",
            install_command="pip install xgboost",
        )


class XGBoostRegressor(BaseRegressor):
    """XGBoost Regressor for regression tasks.

    This class wraps XGBoost's SparkXGBRegressor to provide
    a scikit-learn-like interface with support for train/test and k-fold
    cross-validation.

    Note:
        This requires the xgboost package to be installed.
        Install with: pip install xgboost

    Args:
        task: The regression task type. Default is 'simple_regression'.

    Example:
        >>> from smallaxe.training import XGBoostRegressor
        >>> model = XGBoostRegressor()
        >>> model.set_param({"n_estimators": 100, "max_depth": 6})
        >>> model.fit(df, label_col='target', feature_cols=['f1', 'f2'])
        >>> predictions = model.predict(df)

    Raises:
        DependencyError: If xgboost is not installed.
    """

    def __init__(self, task: str = "simple_regression") -> None:
        """Initialize the XGBoost regressor.

        Args:
            task: The regression task type.

        Raises:
            DependencyError: If xgboost is not installed.
            ValidationError: If task is not a valid regression task.
        """
        _check_xgboost_available()
        super().__init__(task)

    @property
    def params(self) -> Dict[str, str]:
        """Get parameter descriptions.

        Returns:
            Dictionary mapping parameter names to their descriptions.
        """
        return {
            "n_estimators": "Number of boosting rounds",
            "max_depth": "Maximum depth of each tree",
            "learning_rate": "Step size shrinkage used in update to prevent overfitting",
            "subsample": "Fraction of samples used for training each tree",
            "colsample_bytree": "Fraction of features used for training each tree",
            "min_child_weight": "Minimum sum of instance weight needed in a child",
            "reg_alpha": "L1 regularization term on weights",
            "reg_lambda": "L2 regularization term on weights",
            "gamma": "Minimum loss reduction required to make a further partition",
            "seed": "Random seed for reproducibility",
        }

    @property
    def default_params(self) -> Dict[str, Any]:
        """Get default parameter values.

        Returns:
            Dictionary mapping parameter names to their default values.
        """
        return {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.3,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "min_child_weight": 1,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "gamma": 0.0,
            "seed": None,
        }

    def _uses_constructor_col_params(self) -> bool:
        return True

    def _create_spark_estimator(
        self,
        features_col: Optional[str] = None,
        label_col: Optional[str] = None,
        prediction_col: Optional[str] = None,
    ) -> Any:
        """Create the underlying SparkXGBRegressor.

        Returns:
            Configured SparkXGBRegressor instance.
        """
        seed = self.get_param("seed")

        estimator_params = {
            "num_round": self.get_param("n_estimators"),
            "max_depth": self.get_param("max_depth"),
            "learning_rate": self.get_param("learning_rate"),
            "subsample": self.get_param("subsample"),
            "colsample_bytree": self.get_param("colsample_bytree"),
            "min_child_weight": self.get_param("min_child_weight"),
            "reg_alpha": self.get_param("reg_alpha"),
            "reg_lambda": self.get_param("reg_lambda"),
            "gamma": self.get_param("gamma"),
        }

        if seed is not None:
            estimator_params["seed"] = seed
        if features_col is not None:
            estimator_params["features_col"] = features_col
        if label_col is not None:
            estimator_params["label_col"] = label_col
        if prediction_col is not None:
            estimator_params["prediction_col"] = prediction_col

        return SparkXGBRegressor(**estimator_params)

    def _load_artifacts(self, path: str) -> None:
        """Load the Spark model from disk.

        Args:
            path: Directory path where the model is saved.
        """
        self._load_spark_model(path, SparkXGBRegressorModel)


class XGBoostClassifier(BaseClassifier):
    """XGBoost Classifier for classification tasks.

    This class wraps XGBoost's SparkXGBClassifier to provide
    a scikit-learn-like interface with support for train/test and k-fold
    cross-validation, including stratified sampling for classification.

    Note:
        This requires the xgboost package to be installed.
        Install with: pip install xgboost

    Args:
        task: The classification task type. Options are 'binary' or 'multiclass'.
            Default is 'binary'.

    Example:
        >>> from smallaxe.training import XGBoostClassifier
        >>> model = XGBoostClassifier(task='binary')
        >>> model.set_param({"n_estimators": 100, "max_depth": 6})
        >>> model.fit(df, label_col='label', feature_cols=['f1', 'f2'])
        >>> predictions = model.predict(df)
        >>> probabilities = model.predict_proba(df)

    Raises:
        DependencyError: If xgboost is not installed.
    """

    def __init__(self, task: str = "binary") -> None:
        """Initialize the XGBoost classifier.

        Args:
            task: The classification task type.

        Raises:
            DependencyError: If xgboost is not installed.
            ValidationError: If task is not a valid classification task.
        """
        _check_xgboost_available()
        super().__init__(task)

    @property
    def params(self) -> Dict[str, str]:
        """Get parameter descriptions.

        Returns:
            Dictionary mapping parameter names to their descriptions.
        """
        return {
            "n_estimators": "Number of boosting rounds",
            "max_depth": "Maximum depth of each tree",
            "learning_rate": "Step size shrinkage used in update to prevent overfitting",
            "subsample": "Fraction of samples used for training each tree",
            "colsample_bytree": "Fraction of features used for training each tree",
            "min_child_weight": "Minimum sum of instance weight needed in a child",
            "reg_alpha": "L1 regularization term on weights",
            "reg_lambda": "L2 regularization term on weights",
            "gamma": "Minimum loss reduction required to make a further partition",
            "seed": "Random seed for reproducibility",
        }

    @property
    def default_params(self) -> Dict[str, Any]:
        """Get default parameter values.

        Returns:
            Dictionary mapping parameter names to their default values.
        """
        return {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.3,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "min_child_weight": 1,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "gamma": 0.0,
            "seed": None,
        }

    def _uses_constructor_col_params(self) -> bool:
        return True

    def _create_spark_estimator(
        self,
        features_col: Optional[str] = None,
        label_col: Optional[str] = None,
        prediction_col: Optional[str] = None,
    ) -> Any:
        """Create the underlying SparkXGBClassifier.

        Returns:
            Configured SparkXGBClassifier instance.
        """
        seed = self.get_param("seed")

        estimator_params = {
            "num_round": self.get_param("n_estimators"),
            "max_depth": self.get_param("max_depth"),
            "learning_rate": self.get_param("learning_rate"),
            "subsample": self.get_param("subsample"),
            "colsample_bytree": self.get_param("colsample_bytree"),
            "min_child_weight": self.get_param("min_child_weight"),
            "reg_alpha": self.get_param("reg_alpha"),
            "reg_lambda": self.get_param("reg_lambda"),
            "gamma": self.get_param("gamma"),
            "probability_col": self.PROBABILITY_COL,
            "raw_prediction_col": self.RAW_PREDICTION_COL,
        }

        if seed is not None:
            estimator_params["seed"] = seed
        if features_col is not None:
            estimator_params["features_col"] = features_col
        if label_col is not None:
            estimator_params["label_col"] = label_col
        if prediction_col is not None:
            estimator_params["prediction_col"] = prediction_col

        return SparkXGBClassifier(**estimator_params)

    def _load_artifacts(self, path: str) -> None:
        """Load the Spark model from disk.

        Args:
            path: Directory path where the model is saved.
        """
        self._load_spark_model(path, SparkXGBClassifierModel)
