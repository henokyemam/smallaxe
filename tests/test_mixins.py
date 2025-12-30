"""Tests for training mixins."""

import os
import tempfile
from typing import Any, Dict, List

import pytest
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.classification import RandomForestClassifier as SparkRFClassifier
from pyspark.ml.regression import RandomForestRegressionModel
from pyspark.ml.regression import RandomForestRegressor as SparkRFRegressor
from pyspark.sql import functions as F

import smallaxe
from smallaxe.exceptions import ModelNotFittedError, ValidationError
from smallaxe.training.mixins import (
    MetadataMixin,
    ParamMixin,
    PersistenceMixin,
    SparkModelMixin,
    ValidationMixin,
)


# --- Concrete implementations for testing ---


class ConcreteParamMixin(ParamMixin):
    """Concrete implementation of ParamMixin for testing."""

    @property
    def params(self) -> Dict[str, str]:
        return {
            "n_estimators": "Number of trees",
            "max_depth": "Maximum tree depth",
            "learning_rate": "Learning rate",
        }

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.1,
        }


class ConcretePersistenceMixin(PersistenceMixin):
    """Concrete implementation of PersistenceMixin for testing."""

    def __init__(self):
        self._value = None
        self._is_fitted = False

    def _get_persistence_state(self) -> Dict[str, Any]:
        return {
            "value": self._value,
            "is_fitted": self._is_fitted,
        }

    def _set_persistence_state(self, state: Dict[str, Any]) -> None:
        self._value = state.get("value")
        self._is_fitted = state.get("is_fitted", False)

    def fit(self, value: int) -> "ConcretePersistenceMixin":
        self._value = value
        self._is_fitted = True
        return self


class ConcreteValidationMixin(ValidationMixin):
    """Concrete implementation of ValidationMixin for testing."""

    pass


class ConcreteMetadataMixin(MetadataMixin):
    """Concrete implementation of MetadataMixin for testing."""

    pass


class ConcreteSparkModelMixin(SparkModelMixin):
    """Concrete implementation of SparkModelMixin for regression testing."""

    def __init__(self):
        self._spark_model = None
        self._feature_cols = None
        self._label_col = None

    def _create_spark_estimator(self) -> Any:
        return SparkRFRegressor(numTrees=5, maxDepth=3)


class ConcreteSparkClassifierMixin(SparkModelMixin):
    """Concrete implementation of SparkModelMixin for classification testing."""

    def __init__(self):
        self._spark_model = None
        self._feature_cols = None
        self._label_col = None

    def _create_spark_estimator(self) -> Any:
        return SparkRFClassifier(numTrees=5, maxDepth=3)


# --- Test fixtures ---


@pytest.fixture
def regression_df(spark_session):
    """Create a sample DataFrame for regression testing."""
    data = [
        (1, 25.0, 50000.0, 100.0),
        (2, 30.0, 60000.0, 150.0),
        (3, 35.0, 70000.0, 200.0),
        (4, 40.0, 80000.0, 250.0),
        (5, 45.0, 90000.0, 300.0),
        (6, 28.0, 55000.0, 125.0),
        (7, 33.0, 65000.0, 175.0),
        (8, 38.0, 75000.0, 225.0),
        (9, 42.0, 85000.0, 275.0),
        (10, 48.0, 95000.0, 325.0),
    ]
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


# --- ParamMixin Tests ---


class TestParamMixin:
    """Tests for ParamMixin."""

    def test_params_property(self):
        """Test that params property returns parameter descriptions."""
        mixin = ConcreteParamMixin()
        params = mixin.params
        assert "n_estimators" in params
        assert params["n_estimators"] == "Number of trees"

    def test_default_params_property(self):
        """Test that default_params property returns default values."""
        mixin = ConcreteParamMixin()
        defaults = mixin.default_params
        assert defaults["n_estimators"] == 100
        assert defaults["max_depth"] == 5
        assert defaults["learning_rate"] == 0.1

    def test_set_param_single(self):
        """Test setting a single parameter."""
        mixin = ConcreteParamMixin()
        mixin.set_param({"n_estimators": 200})
        assert mixin.get_param("n_estimators") == 200

    def test_set_param_multiple(self):
        """Test setting multiple parameters."""
        mixin = ConcreteParamMixin()
        mixin.set_param({"n_estimators": 200, "max_depth": 10})
        assert mixin.get_param("n_estimators") == 200
        assert mixin.get_param("max_depth") == 10

    def test_set_param_returns_self(self):
        """Test that set_param returns self for chaining."""
        mixin = ConcreteParamMixin()
        result = mixin.set_param({"n_estimators": 200})
        assert result is mixin

    def test_set_param_invalid_name_raises_error(self):
        """Test that invalid parameter name raises ValidationError."""
        mixin = ConcreteParamMixin()
        with pytest.raises(ValidationError, match="Invalid parameter"):
            mixin.set_param({"invalid_param": 100})

    def test_get_param_invalid_name_raises_error(self):
        """Test that getting invalid parameter raises ValidationError."""
        mixin = ConcreteParamMixin()
        with pytest.raises(ValidationError, match="Invalid parameter"):
            mixin.get_param("invalid_param")

    def test_get_params_returns_all(self):
        """Test that get_params returns all current values."""
        mixin = ConcreteParamMixin()
        mixin.set_param({"n_estimators": 200})
        params = mixin.get_params()
        assert params["n_estimators"] == 200
        assert params["max_depth"] == 5  # default
        assert params["learning_rate"] == 0.1  # default

    def test_set_param_type_validation(self):
        """Test that type validation works."""
        mixin = ConcreteParamMixin()
        with pytest.raises(ValidationError, match="must be of type"):
            mixin.set_param({"n_estimators": "not an int"})

    def test_set_param_int_for_float_allowed(self):
        """Test that int is allowed for float parameters."""
        mixin = ConcreteParamMixin()
        mixin.set_param({"learning_rate": 1})  # int instead of float
        assert mixin.get_param("learning_rate") == 1


# --- PersistenceMixin Tests ---


class TestPersistenceMixin:
    """Tests for PersistenceMixin."""

    def test_save_before_fit_raises_error(self):
        """Test that save before fit raises ModelNotFittedError."""
        mixin = ConcretePersistenceMixin()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model")
            with pytest.raises(ModelNotFittedError, match="not been fitted"):
                mixin.save(path)

    def test_save_and_load_roundtrip(self):
        """Test save and load roundtrip."""
        mixin = ConcretePersistenceMixin()
        mixin.fit(42)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model")
            mixin.save(path)

            loaded = ConcretePersistenceMixin.load(path)
            assert loaded._value == 42
            assert loaded._is_fitted is True

    def test_save_creates_directory(self):
        """Test that save creates the directory if it doesn't exist."""
        mixin = ConcretePersistenceMixin()
        mixin.fit(42)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "nested", "model")
            mixin.save(path)
            assert os.path.exists(path)
            assert os.path.exists(os.path.join(path, "metadata.json"))

    def test_load_invalid_path_raises_error(self):
        """Test that load with invalid path raises ValidationError."""
        with pytest.raises(ValidationError, match="does not exist"):
            ConcretePersistenceMixin.load("/nonexistent/path")

    def test_load_missing_metadata_raises_error(self):
        """Test that load without metadata.json raises ValidationError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValidationError, match="metadata.json not found"):
                ConcretePersistenceMixin.load(tmpdir)

    def test_save_empty_path_raises_error(self):
        """Test that save with empty path raises ValidationError."""
        mixin = ConcretePersistenceMixin()
        mixin.fit(42)
        with pytest.raises(ValidationError, match="cannot be empty"):
            mixin.save("")


# --- ValidationMixin Tests ---


class TestValidationMixin:
    """Tests for ValidationMixin."""

    def test_train_test_split_default(self, regression_df):
        """Test train_test_split with default parameters."""
        mixin = ConcreteValidationMixin()
        train_df, test_df = mixin._train_test_split(regression_df, test_size=0.2)

        # Check splits are non-empty
        assert train_df.count() > 0
        assert test_df.count() > 0

        # Check total count is preserved
        assert train_df.count() + test_df.count() == regression_df.count()

    def test_train_test_split_with_seed(self, spark_session, regression_df):
        """Test train_test_split with seed produces reproducible results."""
        smallaxe.set_seed(42)
        mixin = ConcreteValidationMixin()

        train1, test1 = mixin._train_test_split(regression_df, test_size=0.3)
        train2, test2 = mixin._train_test_split(regression_df, test_size=0.3)

        # With same seed, should get same splits
        assert train1.count() == train2.count()
        assert test1.count() == test2.count()

        # Reset seed
        smallaxe.set_seed(None)

    def test_train_test_split_invalid_test_size(self, regression_df):
        """Test that invalid test_size raises ValidationError."""
        mixin = ConcreteValidationMixin()
        with pytest.raises(ValidationError, match="test_size must be between"):
            mixin._train_test_split(regression_df, test_size=0)
        with pytest.raises(ValidationError, match="test_size must be between"):
            mixin._train_test_split(regression_df, test_size=1)
        with pytest.raises(ValidationError, match="test_size must be between"):
            mixin._train_test_split(regression_df, test_size=1.5)

    def test_train_test_split_stratified_without_label_raises_error(self, classification_df):
        """Test that stratified split without label_col raises ValidationError."""
        mixin = ConcreteValidationMixin()
        with pytest.raises(ValidationError, match="label_col must be specified"):
            mixin._train_test_split(classification_df, stratified=True)

    def test_train_test_split_stratified(self, classification_df):
        """Test stratified train_test_split preserves class distribution."""
        mixin = ConcreteValidationMixin()
        train_df, test_df = mixin._train_test_split(
            classification_df,
            test_size=0.3,
            stratified=True,
            label_col="label",
        )

        # Both splits should have both classes
        train_classes = [r["label"] for r in train_df.select("label").distinct().collect()]
        test_classes = [r["label"] for r in test_df.select("label").distinct().collect()]

        assert len(train_classes) == 2
        assert len(test_classes) == 2

    def test_kfold_split_default(self, regression_df):
        """Test kfold_split with default parameters."""
        mixin = ConcreteValidationMixin()
        folds = list(mixin._kfold_split(regression_df, n_folds=5))

        assert len(folds) == 5

        # Each fold should have train and val sets
        for train_df, val_df in folds:
            assert train_df.count() > 0
            assert val_df.count() > 0

    def test_kfold_split_invalid_n_folds(self, regression_df):
        """Test that n_folds < 2 raises ValidationError."""
        mixin = ConcreteValidationMixin()
        with pytest.raises(ValidationError, match="n_folds must be at least 2"):
            list(mixin._kfold_split(regression_df, n_folds=1))

    def test_kfold_split_stratified_without_label_raises_error(self, classification_df):
        """Test that stratified kfold without label_col raises ValidationError."""
        mixin = ConcreteValidationMixin()
        with pytest.raises(ValidationError, match="label_col must be specified"):
            list(mixin._kfold_split(classification_df, stratified=True))

    def test_kfold_split_stratified(self, classification_df):
        """Test stratified kfold_split preserves class distribution in each fold."""
        mixin = ConcreteValidationMixin()
        folds = list(mixin._kfold_split(
            classification_df,
            n_folds=3,
            stratified=True,
            label_col="label",
        ))

        assert len(folds) == 3

        # Each fold's train set should have both classes
        for train_df, val_df in folds:
            train_classes = [r["label"] for r in train_df.select("label").distinct().collect()]
            assert len(train_classes) == 2


# --- MetadataMixin Tests ---


class TestMetadataMixin:
    """Tests for MetadataMixin."""

    def test_metadata_before_fit_raises_error(self):
        """Test that accessing metadata before fit raises ModelNotFittedError."""
        mixin = ConcreteMetadataMixin()
        with pytest.raises(ModelNotFittedError, match="not been fitted"):
            _ = mixin.metadata

    def test_capture_metadata(self, regression_df):
        """Test _capture_metadata stores correct information."""
        mixin = ConcreteMetadataMixin()
        mixin._capture_metadata(
            regression_df,
            label_col="target",
            feature_cols=["age", "income"],
        )

        metadata = mixin.metadata
        assert metadata["label_col"] == "target"
        assert metadata["feature_cols"] == ["age", "income"]
        assert metadata["n_samples"] == 10
        assert metadata["n_features"] == 2
        assert "training_timestamp" in metadata
        assert "smallaxe_version" in metadata

    def test_capture_metadata_with_extra(self, regression_df):
        """Test _capture_metadata with extra metadata."""
        mixin = ConcreteMetadataMixin()
        mixin._capture_metadata(
            regression_df,
            label_col="target",
            feature_cols=["age", "income"],
            extra_metadata={"custom_key": "custom_value"},
        )

        metadata = mixin.metadata
        assert metadata["custom_key"] == "custom_value"

    def test_capture_label_stats_regression(self, regression_df):
        """Test _capture_label_stats for regression."""
        mixin = ConcreteMetadataMixin()
        stats = mixin._capture_label_stats(regression_df, "target", "regression")

        assert "label_min" in stats
        assert "label_max" in stats
        assert "label_mean" in stats
        assert stats["label_min"] == 100.0
        assert stats["label_max"] == 325.0

    def test_capture_label_stats_classification(self, classification_df):
        """Test _capture_label_stats for classification."""
        mixin = ConcreteMetadataMixin()
        stats = mixin._capture_label_stats(classification_df, "label", "binary")

        assert "class_counts" in stats
        assert "n_classes" in stats
        assert stats["n_classes"] == 2

    def test_update_metadata(self, regression_df):
        """Test _update_metadata."""
        mixin = ConcreteMetadataMixin()
        mixin._capture_metadata(regression_df, "target", ["age"])
        mixin._update_metadata("new_key", "new_value")

        assert mixin.metadata["new_key"] == "new_value"

    def test_metadata_is_copy(self, regression_df):
        """Test that metadata property returns a copy."""
        mixin = ConcreteMetadataMixin()
        mixin._capture_metadata(regression_df, "target", ["age"])

        metadata = mixin.metadata
        metadata["modified"] = True

        # Original should not be modified
        assert "modified" not in mixin.metadata


# --- SparkModelMixin Tests ---


class TestSparkModelMixin:
    """Tests for SparkModelMixin."""

    def test_assemble_features(self, regression_df):
        """Test _assemble_features creates features column."""
        mixin = ConcreteSparkModelMixin()
        result = mixin._assemble_features(
            regression_df,
            feature_cols=["age", "income"],
        )

        assert "features" in result.columns

    def test_assemble_features_custom_output_col(self, regression_df):
        """Test _assemble_features with custom output column."""
        mixin = ConcreteSparkModelMixin()
        result = mixin._assemble_features(
            regression_df,
            feature_cols=["age", "income"],
            output_col="my_features",
        )

        assert "my_features" in result.columns

    def test_assemble_features_idempotent(self, regression_df):
        """Test _assemble_features doesn't duplicate features column."""
        mixin = ConcreteSparkModelMixin()
        result1 = mixin._assemble_features(regression_df, ["age", "income"])
        result2 = mixin._assemble_features(result1, ["age", "income"])

        # Should only have one features column
        assert result2.columns.count("features") == 1

    def test_fit_spark_model(self, regression_df):
        """Test _fit_spark_model trains a model."""
        mixin = ConcreteSparkModelMixin()
        model = mixin._fit_spark_model(
            regression_df,
            label_col="target",
            feature_cols=["age", "income"],
        )

        assert model is not None
        assert mixin._spark_model is not None

    def test_predict_spark_model(self, regression_df):
        """Test _predict_spark_model generates predictions."""
        mixin = ConcreteSparkModelMixin()
        mixin._fit_spark_model(
            regression_df,
            label_col="target",
            feature_cols=["age", "income"],
        )

        result = mixin._predict_spark_model(regression_df)

        assert "predict_label" in result.columns
        assert "features" not in result.columns  # Should be dropped
        assert result.count() == regression_df.count()

    def test_predict_spark_model_custom_output_col(self, regression_df):
        """Test _predict_spark_model with custom output column."""
        mixin = ConcreteSparkModelMixin()
        mixin._fit_spark_model(
            regression_df,
            label_col="target",
            feature_cols=["age", "income"],
        )

        result = mixin._predict_spark_model(regression_df, output_col="my_prediction")

        assert "my_prediction" in result.columns

    def test_predict_spark_model_before_fit_raises_error(self, regression_df):
        """Test that predict before fit raises ModelNotFittedError."""
        mixin = ConcreteSparkModelMixin()
        with pytest.raises(ModelNotFittedError, match="not been fitted"):
            mixin._predict_spark_model(regression_df)

    def test_predict_proba_classifier(self, classification_df):
        """Test _predict_proba_spark_model for classification."""
        mixin = ConcreteSparkClassifierMixin()
        mixin._fit_spark_model(
            classification_df,
            label_col="label",
            feature_cols=["age", "income"],
        )

        result = mixin._predict_proba_spark_model(classification_df)

        assert "probability" in result.columns
        assert "features" not in result.columns
        assert "prediction" not in result.columns

    def test_feature_importances(self, regression_df):
        """Test feature_importances property."""
        mixin = ConcreteSparkModelMixin()
        mixin._fit_spark_model(
            regression_df,
            label_col="target",
            feature_cols=["age", "income"],
        )

        importances = mixin.feature_importances

        assert importances is not None
        assert "age" in importances
        assert "income" in importances
        assert len(importances) == 2

    def test_feature_importances_before_fit_raises_error(self):
        """Test that feature_importances before fit raises ModelNotFittedError."""
        mixin = ConcreteSparkModelMixin()
        with pytest.raises(ModelNotFittedError, match="not been fitted"):
            _ = mixin.feature_importances

    def test_save_and_load_spark_model(self, regression_df):
        """Test _save_spark_model and _load_spark_model."""
        mixin = ConcreteSparkModelMixin()
        mixin._fit_spark_model(
            regression_df,
            label_col="target",
            feature_cols=["age", "income"],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            mixin._save_spark_model(tmpdir)
            assert os.path.exists(os.path.join(tmpdir, "spark_model"))

            # Load into new mixin
            mixin2 = ConcreteSparkModelMixin()
            mixin2._feature_cols = ["age", "income"]
            mixin2._load_spark_model(tmpdir, RandomForestRegressionModel)

            assert mixin2._spark_model is not None
