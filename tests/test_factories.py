"""Integration tests for Regressors and Classifiers factory classes."""

import os
import tempfile

import pytest

from smallaxe.exceptions import ValidationError
from smallaxe.training import (
    Classifiers,
    RandomForestClassifier,
    RandomForestRegressor,
    Regressors,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def regression_df(spark_session):
    """Create a sample DataFrame for regression testing."""
    data = [(i, 20.0 + (i % 30), 40000.0 + (i * 1000), 50.0 + (i * 5)) for i in range(1, 101)]
    columns = ["id", "age", "income", "target"]
    return spark_session.createDataFrame(data, columns)


@pytest.fixture
def classification_df(spark_session):
    """Create a sample DataFrame for classification testing."""
    data = [(i, 20.0 + (i % 30), 40000.0 + (i * 1000), i % 2) for i in range(1, 101)]
    columns = ["id", "age", "income", "label"]
    return spark_session.createDataFrame(data, columns)


# =============================================================================
# Regressors Factory Tests
# =============================================================================


class TestRegressorsFactory:
    """Tests for Regressors factory class."""

    def test_random_forest_creates_regressor(self):
        """Test that random_forest creates a RandomForestRegressor."""
        model = Regressors.random_forest()
        assert isinstance(model, RandomForestRegressor)
        assert model.task == "simple_regression"
        assert model.task_type == "regression"

    def test_random_forest_with_params(self):
        """Test that random_forest sets parameters correctly."""
        model = Regressors.random_forest(n_estimators=100, max_depth=10)
        assert model.get_param("n_estimators") == 100
        assert model.get_param("max_depth") == 10

    def test_random_forest_default_params(self):
        """Test that random_forest without params uses defaults."""
        model = Regressors.random_forest()
        assert model.get_param("n_estimators") == 20
        assert model.get_param("max_depth") == 5

    def test_random_forest_fit_predict(self, regression_df):
        """Test full workflow with factory-created regressor."""
        model = Regressors.random_forest(n_estimators=10, seed=42)
        model.fit(regression_df, label_col="target", feature_cols=["age", "income"])

        predictions = model.predict(regression_df)
        assert "predict_label" in predictions.columns
        assert predictions.count() == regression_df.count()

    def test_list_models(self):
        """Test list_models returns available regressors."""
        models = Regressors.list_models()
        assert isinstance(models, list)
        assert "RandomForestRegressor" in models

    def test_load_regressor(self, regression_df):
        """Test loading a regressor using factory load method."""
        model = Regressors.random_forest(n_estimators=30, max_depth=8)
        model.fit(regression_df, label_col="target", feature_cols=["age", "income"])

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model")
            model.save(save_path)

            loaded_model = Regressors.load(save_path)

            assert isinstance(loaded_model, RandomForestRegressor)
            assert loaded_model._is_fitted
            assert loaded_model.get_param("n_estimators") == 30
            assert loaded_model.get_param("max_depth") == 8

    def test_load_regressor_can_predict(self, regression_df):
        """Test that loaded regressor can make predictions."""
        model = Regressors.random_forest(seed=42)
        model.fit(regression_df, label_col="target", feature_cols=["age", "income"])

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model")
            model.save(save_path)

            loaded_model = Regressors.load(save_path)
            predictions = loaded_model.predict(regression_df)

            assert "predict_label" in predictions.columns
            assert predictions.count() == regression_df.count()

    def test_load_nonexistent_path_raises_error(self):
        """Test that loading from nonexistent path raises error."""
        with pytest.raises(FileNotFoundError):
            Regressors.load("/nonexistent/path")

    def test_load_classifier_as_regressor_raises_error(self, classification_df):
        """Test that loading a classifier using Regressors.load raises error."""
        model = Classifiers.random_forest()
        model.fit(classification_df, label_col="label", feature_cols=["age", "income"])

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model")
            model.save(save_path)

            with pytest.raises(ValidationError, match="not a supported regressor"):
                Regressors.load(save_path)


# =============================================================================
# Classifiers Factory Tests
# =============================================================================


class TestClassifiersFactory:
    """Tests for Classifiers factory class."""

    def test_random_forest_creates_classifier(self):
        """Test that random_forest creates a RandomForestClassifier."""
        model = Classifiers.random_forest()
        assert isinstance(model, RandomForestClassifier)
        assert model.task == "binary"
        assert model.task_type == "classification"

    def test_random_forest_with_binary_task(self):
        """Test that random_forest creates binary classifier."""
        model = Classifiers.random_forest(task="binary")
        assert model.task == "binary"

    def test_random_forest_with_multiclass_task(self):
        """Test that random_forest creates multiclass classifier."""
        model = Classifiers.random_forest(task="multiclass")
        assert model.task == "multiclass"

    def test_random_forest_with_params(self):
        """Test that random_forest sets parameters correctly."""
        model = Classifiers.random_forest(n_estimators=100, max_depth=10)
        assert model.get_param("n_estimators") == 100
        assert model.get_param("max_depth") == 10

    def test_random_forest_with_task_and_params(self):
        """Test that random_forest sets both task and parameters."""
        model = Classifiers.random_forest(task="multiclass", n_estimators=50)
        assert model.task == "multiclass"
        assert model.get_param("n_estimators") == 50

    def test_random_forest_default_params(self):
        """Test that random_forest without params uses defaults."""
        model = Classifiers.random_forest()
        assert model.get_param("n_estimators") == 20
        assert model.get_param("max_depth") == 5

    def test_random_forest_fit_predict(self, classification_df):
        """Test full workflow with factory-created classifier."""
        model = Classifiers.random_forest(n_estimators=10, seed=42)
        model.fit(classification_df, label_col="label", feature_cols=["age", "income"])

        predictions = model.predict(classification_df)
        assert "predict_label" in predictions.columns
        assert predictions.count() == classification_df.count()

    def test_random_forest_fit_predict_proba(self, classification_df):
        """Test predict_proba with factory-created classifier."""
        model = Classifiers.random_forest(n_estimators=10, seed=42)
        model.fit(classification_df, label_col="label", feature_cols=["age", "income"])

        proba = model.predict_proba(classification_df)
        assert "probability" in proba.columns
        assert proba.count() == classification_df.count()

    def test_list_models(self):
        """Test list_models returns available classifiers."""
        models = Classifiers.list_models()
        assert isinstance(models, list)
        assert "RandomForestClassifier" in models

    def test_load_classifier(self, classification_df):
        """Test loading a classifier using factory load method."""
        model = Classifiers.random_forest(n_estimators=30, max_depth=8)
        model.fit(classification_df, label_col="label", feature_cols=["age", "income"])

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model")
            model.save(save_path)

            loaded_model = Classifiers.load(save_path)

            assert isinstance(loaded_model, RandomForestClassifier)
            assert loaded_model._is_fitted
            assert loaded_model.get_param("n_estimators") == 30
            assert loaded_model.get_param("max_depth") == 8

    def test_load_classifier_can_predict(self, classification_df):
        """Test that loaded classifier can make predictions."""
        model = Classifiers.random_forest(seed=42)
        model.fit(classification_df, label_col="label", feature_cols=["age", "income"])

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model")
            model.save(save_path)

            loaded_model = Classifiers.load(save_path)
            predictions = loaded_model.predict(classification_df)

            assert "predict_label" in predictions.columns
            assert predictions.count() == classification_df.count()

    def test_load_classifier_can_predict_proba(self, classification_df):
        """Test that loaded classifier can make probability predictions."""
        model = Classifiers.random_forest(seed=42)
        model.fit(classification_df, label_col="label", feature_cols=["age", "income"])

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model")
            model.save(save_path)

            loaded_model = Classifiers.load(save_path)
            proba = loaded_model.predict_proba(classification_df)

            assert "probability" in proba.columns
            assert proba.count() == classification_df.count()

    def test_load_nonexistent_path_raises_error(self):
        """Test that loading from nonexistent path raises error."""
        with pytest.raises(FileNotFoundError):
            Classifiers.load("/nonexistent/path")

    def test_load_regressor_as_classifier_raises_error(self, regression_df):
        """Test that loading a regressor using Classifiers.load raises error."""
        model = Regressors.random_forest()
        model.fit(regression_df, label_col="target", feature_cols=["age", "income"])

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model")
            model.save(save_path)

            with pytest.raises(ValidationError, match="not a supported classifier"):
                Classifiers.load(save_path)


# =============================================================================
# Integration Tests
# =============================================================================


class TestFactoryIntegration:
    """Integration tests for factory classes."""

    def test_regressor_full_workflow(self, regression_df):
        """Test complete regressor workflow: create, fit, predict, save, load, predict."""
        # Create model via factory
        model = Regressors.random_forest(n_estimators=15, max_depth=6, seed=42)

        # Fit with validation
        model.fit(
            regression_df,
            label_col="target",
            feature_cols=["age", "income"],
            validation="train_test",
            test_size=0.2,
        )

        # Verify model is fitted and has validation scores
        assert model._is_fitted
        assert model.validation_scores is not None
        assert "rmse" in model.validation_scores

        # Get predictions
        predictions = model.predict(regression_df)
        original_count = predictions.count()

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model")
            model.save(save_path)

            loaded_model = Regressors.load(save_path)

            # Verify loaded model works
            loaded_predictions = loaded_model.predict(regression_df)
            assert loaded_predictions.count() == original_count

            # Verify params preserved
            assert loaded_model.get_param("n_estimators") == 15
            assert loaded_model.get_param("max_depth") == 6

    def test_classifier_full_workflow(self, classification_df):
        """Test complete classifier workflow: create, fit, predict, save, load, predict."""
        # Create model via factory
        model = Classifiers.random_forest(task="binary", n_estimators=15, max_depth=6, seed=42)

        # Fit with validation
        model.fit(
            classification_df,
            label_col="label",
            feature_cols=["age", "income"],
            validation="train_test",
            test_size=0.2,
        )

        # Verify model is fitted and has validation scores
        assert model._is_fitted
        assert model.validation_scores is not None
        assert "accuracy" in model.validation_scores

        # Get predictions
        predictions = model.predict(classification_df)
        _proba = model.predict_proba(classification_df)
        original_count = predictions.count()

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model")
            model.save(save_path)

            loaded_model = Classifiers.load(save_path)

            # Verify loaded model works
            loaded_predictions = loaded_model.predict(classification_df)
            loaded_proba = loaded_model.predict_proba(classification_df)
            assert loaded_predictions.count() == original_count
            assert loaded_proba.count() == original_count

            # Verify params preserved
            assert loaded_model.get_param("n_estimators") == 15
            assert loaded_model.get_param("max_depth") == 6
            assert loaded_model.task == "binary"

    def test_model_interoperability(self, regression_df, classification_df):
        """Test that regressor and classifier save/load are independent."""
        reg_model = Regressors.random_forest(seed=42)
        clf_model = Classifiers.random_forest(seed=42)

        reg_model.fit(regression_df, label_col="target", feature_cols=["age", "income"])
        clf_model.fit(classification_df, label_col="label", feature_cols=["age", "income"])

        with tempfile.TemporaryDirectory() as tmpdir:
            reg_path = os.path.join(tmpdir, "reg_model")
            clf_path = os.path.join(tmpdir, "clf_model")

            reg_model.save(reg_path)
            clf_model.save(clf_path)

            # Load correctly
            loaded_reg = Regressors.load(reg_path)
            loaded_clf = Classifiers.load(clf_path)

            assert isinstance(loaded_reg, RandomForestRegressor)
            assert isinstance(loaded_clf, RandomForestClassifier)

            # Cross-loading should fail
            with pytest.raises(ValidationError):
                Regressors.load(clf_path)

            with pytest.raises(ValidationError):
                Classifiers.load(reg_path)
