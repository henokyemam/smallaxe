"""Tests for training base classes."""

from typing import Any, Dict

import pytest
from pyspark.ml.classification import RandomForestClassifier as SparkRFClassifier
from pyspark.ml.regression import RandomForestRegressor as SparkRFRegressor
from pyspark.ml.regression import RandomForestRegressionModel
from pyspark.ml.classification import RandomForestClassificationModel

from smallaxe.exceptions import ModelNotFittedError, ValidationError
from smallaxe.training.base import BaseClassifier, BaseModel, BaseRegressor


# --- Concrete implementations for testing ---


class ConcreteRegressor(BaseRegressor):
    """Concrete implementation of BaseRegressor for testing."""

    @property
    def params(self) -> Dict[str, str]:
        return {
            "n_estimators": "Number of trees",
            "max_depth": "Maximum tree depth",
        }

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "n_estimators": 10,
            "max_depth": 5,
        }

    def _create_spark_estimator(self) -> Any:
        n_estimators = self.get_param("n_estimators")
        max_depth = self.get_param("max_depth")
        return SparkRFRegressor(numTrees=n_estimators, maxDepth=max_depth)

    def _load_artifacts(self, path: str) -> None:
        """Load the Spark model from disk."""
        self._load_spark_model(path, RandomForestRegressionModel)


class ConcreteClassifier(BaseClassifier):
    """Concrete implementation of BaseClassifier for testing."""

    @property
    def params(self) -> Dict[str, str]:
        return {
            "n_estimators": "Number of trees",
            "max_depth": "Maximum tree depth",
        }

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "n_estimators": 10,
            "max_depth": 5,
        }

    def _create_spark_estimator(self) -> Any:
        n_estimators = self.get_param("n_estimators")
        max_depth = self.get_param("max_depth")
        return SparkRFClassifier(numTrees=n_estimators, maxDepth=max_depth)

    def _load_artifacts(self, path: str) -> None:
        """Load the Spark model from disk."""
        self._load_spark_model(path, RandomForestClassificationModel)


# --- Test fixtures ---


@pytest.fixture
def regression_df(spark_session):
    """Create a sample DataFrame for regression testing."""
    data = [(i, 20.0 + (i % 30), 40000.0 + (i * 1000), 50.0 + (i * 5)) for i in range(1, 51)]
    columns = ["id", "age", "income", "target"]
    return spark_session.createDataFrame(data, columns)


@pytest.fixture
def classification_df(spark_session):
    """Create a sample DataFrame for classification testing."""
    data = [
        (1, 25.0, 50000.0, 0),
        (2, 30.0, 60000.0, 1),
        (3, 35.0, 70000.0, 0),
        (4, 40.0, 80000.0, 1),
        (5, 45.0, 90000.0, 1),
        (6, 28.0, 55000.0, 0),
        (7, 33.0, 65000.0, 1),
        (8, 38.0, 75000.0, 0),
        (9, 42.0, 85000.0, 1),
        (10, 48.0, 95000.0, 1),
        (11, 27.0, 52000.0, 0),
        (12, 32.0, 62000.0, 1),
        (13, 36.0, 72000.0, 0),
        (14, 41.0, 82000.0, 1),
        (15, 46.0, 92000.0, 1),
        (16, 29.0, 57000.0, 0),
        (17, 34.0, 67000.0, 1),
        (18, 39.0, 77000.0, 0),
        (19, 43.0, 87000.0, 1),
        (20, 49.0, 97000.0, 0),
    ]
    columns = ["id", "age", "income", "label"]
    return spark_session.createDataFrame(data, columns)


# --- BaseModel Tests ---


class TestBaseModelTaskType:
    """Tests for task_type property."""

    def test_task_type_regression(self):
        """Test that regression tasks return 'regression' task_type."""
        model = ConcreteRegressor(task="simple_regression")
        assert model.task == "simple_regression"
        assert model.task_type == "regression"

    def test_task_type_binary_classification(self):
        """Test that binary classification returns 'classification' task_type."""
        model = ConcreteClassifier(task="binary")
        assert model.task == "binary"
        assert model.task_type == "classification"

    def test_task_type_multiclass_classification(self):
        """Test that multiclass classification returns 'classification' task_type."""
        model = ConcreteClassifier(task="multiclass")
        assert model.task == "multiclass"
        assert model.task_type == "classification"

    def test_invalid_task_raises_error(self):
        """Test that invalid task raises ValidationError."""
        with pytest.raises(ValidationError, match="Invalid regression task"):
            ConcreteRegressor(task="invalid_task")

    def test_regressor_with_classification_task_raises_error(self):
        """Test that BaseRegressor with classification task raises ValidationError."""
        with pytest.raises(ValidationError, match="Invalid regression task"):
            ConcreteRegressor(task="binary")

    def test_classifier_with_regression_task_raises_error(self):
        """Test that BaseClassifier with regression task raises ValidationError."""
        with pytest.raises(ValidationError, match="Invalid classification task"):
            ConcreteClassifier(task="simple_regression")


class TestBaseModelExcludeCols:
    """Tests for exclude_cols handling in fit."""

    def test_exclude_cols_excludes_columns(self, regression_df):
        """Test that exclude_cols properly excludes columns from features."""
        model = ConcreteRegressor()
        model.fit(
            regression_df,
            label_col="target",
            exclude_cols=["id"],
        )

        # id should not be in feature_cols
        assert "id" not in model._feature_cols
        assert "age" in model._feature_cols
        assert "income" in model._feature_cols

    def test_exclude_cols_stored_in_metadata(self, regression_df):
        """Test that exclude_cols is stored in metadata."""
        model = ConcreteRegressor()
        model.fit(
            regression_df,
            label_col="target",
            exclude_cols=["id"],
        )

        metadata = model.metadata
        assert "exclude_cols" in metadata
        assert metadata["exclude_cols"] == ["id"]

    def test_exclude_cols_with_explicit_feature_cols(self, regression_df):
        """Test that exclude_cols is still stored when feature_cols is explicit."""
        model = ConcreteRegressor()
        model.fit(
            regression_df,
            label_col="target",
            feature_cols=["age", "income"],
            exclude_cols=["id"],
        )

        # Feature cols should be as specified
        assert model._feature_cols == ["age", "income"]
        # Exclude cols should still be stored
        assert model._exclude_cols == ["id"]

    def test_no_exclude_cols(self, regression_df):
        """Test fitting without exclude_cols."""
        model = ConcreteRegressor()
        model.fit(
            regression_df,
            label_col="target",
            feature_cols=["age", "income"],
        )

        assert model._exclude_cols == []


class TestBaseModelPredictBeforeFit:
    """Tests for predict before fit raises ModelNotFittedError."""

    def test_predict_before_fit_raises_error(self, regression_df):
        """Test that predict before fit raises ModelNotFittedError."""
        model = ConcreteRegressor()
        with pytest.raises(ModelNotFittedError, match="Model has not been fitted"):
            model.predict(regression_df)

    def test_predict_proba_before_fit_raises_error(self, classification_df):
        """Test that predict_proba before fit raises ModelNotFittedError."""
        model = ConcreteClassifier()
        with pytest.raises(ModelNotFittedError, match="Model has not been fitted"):
            model.predict_proba(classification_df)

    def test_validation_scores_before_fit_raises_error(self):
        """Test that accessing validation_scores before fit raises ModelNotFittedError."""
        model = ConcreteRegressor()
        with pytest.raises(ModelNotFittedError, match="Model has not been fitted"):
            _ = model.validation_scores


class TestBaseModelMetadata:
    """Tests for metadata populated after fit."""

    def test_metadata_populated_after_fit(self, regression_df):
        """Test that metadata is populated after fit."""
        model = ConcreteRegressor()
        model.fit(
            regression_df,
            label_col="target",
            feature_cols=["age", "income"],
        )

        metadata = model.metadata
        assert "training_timestamp" in metadata
        assert "spark_version" in metadata
        assert "smallaxe_version" in metadata
        assert metadata["label_col"] == "target"
        assert metadata["feature_cols"] == ["age", "income"]
        assert metadata["n_samples"] == 50
        assert metadata["n_features"] == 2
        assert metadata["task"] == "simple_regression"
        assert metadata["task_type"] == "regression"

    def test_metadata_contains_label_stats_regression(self, regression_df):
        """Test that regression metadata contains label statistics."""
        model = ConcreteRegressor()
        model.fit(
            regression_df,
            label_col="target",
            feature_cols=["age", "income"],
        )

        metadata = model.metadata
        assert "label_min" in metadata
        assert "label_max" in metadata
        assert "label_mean" in metadata

    def test_metadata_contains_label_stats_classification(self, classification_df):
        """Test that classification metadata contains class counts."""
        model = ConcreteClassifier()
        model.fit(
            classification_df,
            label_col="label",
            feature_cols=["age", "income"],
        )

        metadata = model.metadata
        assert "class_counts" in metadata
        assert "n_classes" in metadata
        assert metadata["n_classes"] == 2


class TestBaseModelFit:
    """Tests for fit functionality."""

    def test_fit_returns_self(self, regression_df):
        """Test that fit returns self for method chaining."""
        model = ConcreteRegressor()
        result = model.fit(
            regression_df,
            label_col="target",
            feature_cols=["age", "income"],
        )
        assert result is model

    def test_fit_marks_model_as_fitted(self, regression_df):
        """Test that fit marks the model as fitted."""
        model = ConcreteRegressor()
        assert not model._is_fitted
        model.fit(
            regression_df,
            label_col="target",
            feature_cols=["age", "income"],
        )
        assert model._is_fitted

    def test_fit_infers_feature_cols(self, regression_df):
        """Test that fit infers feature columns when not provided."""
        model = ConcreteRegressor()
        model.fit(
            regression_df,
            label_col="target",
            exclude_cols=["id"],
        )

        # Should infer age and income as features (numeric columns excluding id and target)
        assert set(model._feature_cols) == {"age", "income"}

    def test_fit_invalid_validation_raises_error(self, regression_df):
        """Test that invalid validation strategy raises ValidationError."""
        model = ConcreteRegressor()
        with pytest.raises(ValidationError, match="Invalid validation strategy"):
            model.fit(
                regression_df,
                label_col="target",
                feature_cols=["age", "income"],
                validation="invalid",
            )

    def test_fit_invalid_cache_strategy_raises_error(self, regression_df):
        """Test that invalid cache_strategy raises ValidationError."""
        model = ConcreteRegressor()
        with pytest.raises(ValidationError, match="Invalid cache_strategy"):
            model.fit(
                regression_df,
                label_col="target",
                feature_cols=["age", "income"],
                cache_strategy="invalid",
            )


class TestBaseModelPredict:
    """Tests for predict functionality."""

    def test_predict_returns_dataframe(self, regression_df):
        """Test that predict returns a DataFrame."""
        model = ConcreteRegressor()
        model.fit(
            regression_df,
            label_col="target",
            feature_cols=["age", "income"],
        )

        result = model.predict(regression_df)
        assert result is not None
        assert result.count() == regression_df.count()

    def test_predict_adds_prediction_column(self, regression_df):
        """Test that predict adds the prediction column."""
        model = ConcreteRegressor()
        model.fit(
            regression_df,
            label_col="target",
            feature_cols=["age", "income"],
        )

        result = model.predict(regression_df)
        assert "predict_label" in result.columns

    def test_predict_custom_output_col(self, regression_df):
        """Test that predict uses custom output column name."""
        model = ConcreteRegressor()
        model.fit(
            regression_df,
            label_col="target",
            feature_cols=["age", "income"],
        )

        result = model.predict(regression_df, output_col="my_prediction")
        assert "my_prediction" in result.columns


class TestBaseClassifier:
    """Tests for BaseClassifier specific functionality."""

    def test_predict_proba_returns_dataframe(self, classification_df):
        """Test that predict_proba returns a DataFrame."""
        model = ConcreteClassifier()
        model.fit(
            classification_df,
            label_col="label",
            feature_cols=["age", "income"],
        )

        result = model.predict_proba(classification_df)
        assert result is not None
        assert result.count() == classification_df.count()

    def test_predict_proba_adds_probability_column(self, classification_df):
        """Test that predict_proba adds the probability column."""
        model = ConcreteClassifier()
        model.fit(
            classification_df,
            label_col="label",
            feature_cols=["age", "income"],
        )

        result = model.predict_proba(classification_df)
        assert "probability" in result.columns

    def test_predict_proba_custom_output_col(self, classification_df):
        """Test that predict_proba uses custom output column name."""
        model = ConcreteClassifier()
        model.fit(
            classification_df,
            label_col="label",
            feature_cols=["age", "income"],
        )

        result = model.predict_proba(classification_df, output_col="my_proba")
        assert "my_proba" in result.columns


class TestBaseModelValidation:
    """Tests for validation functionality."""

    def test_validation_none(self, regression_df):
        """Test that validation='none' results in no validation scores."""
        model = ConcreteRegressor()
        model.fit(
            regression_df,
            label_col="target",
            feature_cols=["age", "income"],
            validation="none",
        )

        assert model.validation_scores is None

    def test_validation_train_test(self, regression_df):
        """Test that validation='train_test' produces validation scores."""
        model = ConcreteRegressor()
        model.fit(
            regression_df,
            label_col="target",
            feature_cols=["age", "income"],
            validation="train_test",
            test_size=0.3,
        )

        scores = model.validation_scores
        assert scores is not None
        assert scores["validation_type"] == "train_test"
        assert scores["test_size"] == 0.3
        assert "rmse" in scores
        assert "mae" in scores
        assert "r2" in scores

    def test_validation_kfold(self, regression_df):
        """Test that validation='kfold' produces validation scores."""
        model = ConcreteRegressor()
        model.fit(
            regression_df,
            label_col="target",
            feature_cols=["age", "income"],
            validation="kfold",
            n_folds=3,
        )

        scores = model.validation_scores
        assert scores is not None
        assert scores["validation_type"] == "kfold"
        assert scores["n_folds"] == 3
        assert "mean_rmse" in scores
        assert "mean_mae" in scores
        assert "mean_r2" in scores
        assert "fold_scores" in scores
        assert len(scores["fold_scores"]) == 3

    def test_validation_classification_metrics(self, classification_df):
        """Test that classification validation produces classification metrics."""
        model = ConcreteClassifier()
        model.fit(
            classification_df,
            label_col="label",
            feature_cols=["age", "income"],
            validation="train_test",
        )

        scores = model.validation_scores
        assert scores is not None
        assert "accuracy" in scores
        assert "precision" in scores
        assert "recall" in scores
        assert "f1_score" in scores


class TestBaseModelCacheStrategy:
    """Tests for cache_strategy functionality."""

    def test_cache_strategy_none(self, regression_df):
        """Test that cache_strategy='none' works."""
        model = ConcreteRegressor()
        model.fit(
            regression_df,
            label_col="target",
            feature_cols=["age", "income"],
            cache_strategy="none",
        )
        assert model._is_fitted

    def test_cache_strategy_memory(self, regression_df):
        """Test that cache_strategy='memory' works."""
        model = ConcreteRegressor()
        model.fit(
            regression_df,
            label_col="target",
            feature_cols=["age", "income"],
            cache_strategy="memory",
        )
        assert model._is_fitted

    def test_cache_strategy_disk(self, regression_df):
        """Test that cache_strategy='disk' works."""
        model = ConcreteRegressor()
        model.fit(
            regression_df,
            label_col="target",
            feature_cols=["age", "income"],
            cache_strategy="disk",
        )
        assert model._is_fitted
