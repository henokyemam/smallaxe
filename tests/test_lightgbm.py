"""Tests for LightGBMRegressor and LightGBMClassifier."""

import os
import tempfile

import pytest

from smallaxe.exceptions import DependencyError, ModelNotFittedError, ValidationError

# Check if LightGBM (via SynapseML) is available
try:
    from synapse.ml.lightgbm import LightGBMClassifier  # noqa: F401

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# Skip all tests if LightGBM is not installed
pytestmark = pytest.mark.skipif(
    not LIGHTGBM_AVAILABLE,
    reason="synapseml not installed",
)


@pytest.fixture
def regression_df(spark_session):
    """Create a sample DataFrame for regression testing."""
    data = [(i, 20.0 + (i % 30), 40000.0 + (i * 1000), 50.0 + (i * 5)) for i in range(1, 101)]
    columns = ["id", "age", "income", "target"]
    return spark_session.createDataFrame(data, columns)


@pytest.fixture
def classification_df(spark_session):
    """Create a sample DataFrame for binary classification testing."""
    data = [(i, 20.0 + (i % 30), 40000.0 + (i * 1000), i % 2) for i in range(1, 101)]
    columns = ["id", "age", "income", "label"]
    return spark_session.createDataFrame(data, columns)


@pytest.fixture
def multiclass_df(spark_session):
    """Create a sample DataFrame for multiclass classification testing."""
    data = [(i, 20.0 + (i % 30), 40000.0 + (i * 1000), i % 3) for i in range(1, 101)]
    columns = ["id", "age", "income", "label"]
    return spark_session.createDataFrame(data, columns)


# =============================================================================
# LightGBMRegressor Tests
# =============================================================================


class TestLightGBMRegressorInit:
    """Tests for LightGBMRegressor initialization."""

    def test_default_task(self):
        """Test that default task is 'simple_regression'."""
        from smallaxe.training.lightgbm import LightGBMRegressor

        model = LightGBMRegressor()
        assert model.task == "simple_regression"
        assert model.task_type == "regression"

    def test_explicit_task(self):
        """Test that explicit task is set correctly."""
        from smallaxe.training.lightgbm import LightGBMRegressor

        model = LightGBMRegressor(task="simple_regression")
        assert model.task == "simple_regression"

    def test_invalid_task_raises_error(self):
        """Test that invalid task raises ValidationError."""
        from smallaxe.training.lightgbm import LightGBMRegressor

        with pytest.raises(ValidationError, match="Invalid regression task"):
            LightGBMRegressor(task="binary")


class TestLightGBMRegressorParams:
    """Tests for params and default_params."""

    def test_params_dict(self):
        """Test that params returns parameter descriptions."""
        from smallaxe.training.lightgbm import LightGBMRegressor

        model = LightGBMRegressor()
        params = model.params

        assert "n_estimators" in params
        assert "max_depth" in params
        assert "learning_rate" in params
        assert "num_leaves" in params
        assert "min_data_in_leaf" in params
        assert "feature_fraction" in params
        assert "bagging_fraction" in params
        assert "bagging_freq" in params
        assert "lambda_l1" in params
        assert "lambda_l2" in params
        assert "seed" in params

        # Check descriptions are strings
        for _key, value in params.items():
            assert isinstance(value, str)
            assert len(value) > 0

    def test_default_params_dict(self):
        """Test that default_params returns default values."""
        from smallaxe.training.lightgbm import LightGBMRegressor

        model = LightGBMRegressor()
        defaults = model.default_params

        assert defaults["n_estimators"] == 100
        assert defaults["max_depth"] == -1
        assert defaults["learning_rate"] == 0.1
        assert defaults["num_leaves"] == 31
        assert defaults["min_data_in_leaf"] == 20
        assert defaults["feature_fraction"] == 1.0
        assert defaults["bagging_fraction"] == 1.0
        assert defaults["bagging_freq"] == 0
        assert defaults["lambda_l1"] == 0.0
        assert defaults["lambda_l2"] == 0.0
        assert defaults["seed"] is None

    def test_get_param_returns_default(self):
        """Test that get_param returns default value when not set."""
        from smallaxe.training.lightgbm import LightGBMRegressor

        model = LightGBMRegressor()
        assert model.get_param("n_estimators") == 100
        assert model.get_param("max_depth") == -1


class TestLightGBMRegressorSetParam:
    """Tests for set_param method."""

    def test_set_param_single(self):
        """Test setting a single parameter."""
        from smallaxe.training.lightgbm import LightGBMRegressor

        model = LightGBMRegressor()
        model.set_param({"n_estimators": 200})
        assert model.get_param("n_estimators") == 200

    def test_set_param_multiple(self):
        """Test setting multiple parameters."""
        from smallaxe.training.lightgbm import LightGBMRegressor

        model = LightGBMRegressor()
        model.set_param(
            {
                "n_estimators": 50,
                "max_depth": 10,
                "learning_rate": 0.05,
            }
        )
        assert model.get_param("n_estimators") == 50
        assert model.get_param("max_depth") == 10
        assert model.get_param("learning_rate") == 0.05

    def test_set_param_invalid_key(self):
        """Test that invalid parameter key raises ValidationError."""
        from smallaxe.training.lightgbm import LightGBMRegressor

        model = LightGBMRegressor()
        with pytest.raises(ValidationError, match="Invalid parameter"):
            model.set_param({"invalid_param": 10})

    def test_set_param_returns_self(self):
        """Test that set_param returns self for method chaining."""
        from smallaxe.training.lightgbm import LightGBMRegressor

        model = LightGBMRegressor()
        result = model.set_param({"n_estimators": 100})
        assert result is model


class TestLightGBMRegressorFit:
    """Tests for fit method."""

    def test_fit_returns_self(self, regression_df):
        """Test that fit returns self for method chaining."""
        from smallaxe.training.lightgbm import LightGBMRegressor

        model = LightGBMRegressor()
        result = model.fit(
            regression_df,
            label_col="target",
            feature_cols=["age", "income"],
        )
        assert result is model

    def test_fit_marks_model_as_fitted(self, regression_df):
        """Test that fit marks the model as fitted."""
        from smallaxe.training.lightgbm import LightGBMRegressor

        model = LightGBMRegressor()
        assert not model._is_fitted
        model.fit(
            regression_df,
            label_col="target",
            feature_cols=["age", "income"],
        )
        assert model._is_fitted

    def test_fit_infers_feature_cols(self, regression_df):
        """Test that fit infers feature columns when not provided."""
        from smallaxe.training.lightgbm import LightGBMRegressor

        model = LightGBMRegressor()
        model.fit(
            regression_df,
            label_col="target",
            exclude_cols=["id"],
        )
        assert set(model._feature_cols) == {"age", "income"}

    def test_fit_with_custom_params(self, regression_df):
        """Test that fit uses custom parameters."""
        from smallaxe.training.lightgbm import LightGBMRegressor

        model = LightGBMRegressor()
        model.set_param({"n_estimators": 50, "max_depth": 4})
        model.fit(
            regression_df,
            label_col="target",
            feature_cols=["age", "income"],
        )
        assert model._is_fitted


class TestLightGBMRegressorPredict:
    """Tests for predict method."""

    def test_predict_before_fit_raises_error(self, regression_df):
        """Test that predict before fit raises ModelNotFittedError."""
        from smallaxe.training.lightgbm import LightGBMRegressor

        model = LightGBMRegressor()
        with pytest.raises(ModelNotFittedError, match="Model has not been fitted"):
            model.predict(regression_df)

    def test_predict_returns_dataframe(self, regression_df):
        """Test that predict returns a DataFrame."""
        from smallaxe.training.lightgbm import LightGBMRegressor

        model = LightGBMRegressor()
        model.fit(
            regression_df,
            label_col="target",
            feature_cols=["age", "income"],
        )
        result = model.predict(regression_df)
        assert result is not None
        assert result.count() == regression_df.count()

    def test_predict_adds_default_column(self, regression_df):
        """Test that predict adds default prediction column."""
        from smallaxe.training.lightgbm import LightGBMRegressor

        model = LightGBMRegressor()
        model.fit(
            regression_df,
            label_col="target",
            feature_cols=["age", "income"],
        )
        result = model.predict(regression_df)
        assert "predict_label" in result.columns

    def test_predict_custom_output_col(self, regression_df):
        """Test that predict uses custom output column name."""
        from smallaxe.training.lightgbm import LightGBMRegressor

        model = LightGBMRegressor()
        model.fit(
            regression_df,
            label_col="target",
            feature_cols=["age", "income"],
        )
        result = model.predict(regression_df, output_col="my_prediction")
        assert "my_prediction" in result.columns
        assert "predict_label" not in result.columns

    def test_predict_preserves_original_columns(self, regression_df):
        """Test that predict preserves original DataFrame columns."""
        from smallaxe.training.lightgbm import LightGBMRegressor

        model = LightGBMRegressor()
        model.fit(
            regression_df,
            label_col="target",
            feature_cols=["age", "income"],
        )
        result = model.predict(regression_df)
        for col in ["id", "age", "income", "target"]:
            assert col in result.columns


class TestLightGBMRegressorValidation:
    """Tests for validation strategies."""

    def test_validation_none(self, regression_df):
        """Test validation='none' results in no validation scores."""
        from smallaxe.training.lightgbm import LightGBMRegressor

        model = LightGBMRegressor()
        model.fit(
            regression_df,
            label_col="target",
            feature_cols=["age", "income"],
            validation="none",
        )
        assert model.validation_scores is None

    def test_validation_train_test(self, regression_df):
        """Test validation='train_test' produces validation scores."""
        from smallaxe.training.lightgbm import LightGBMRegressor

        model = LightGBMRegressor()
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
        assert "mse" in scores
        assert "rmse" in scores
        assert "mae" in scores
        assert "r2" in scores

    def test_validation_kfold(self, regression_df):
        """Test validation='kfold' produces validation scores."""
        from smallaxe.training.lightgbm import LightGBMRegressor

        model = LightGBMRegressor()
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
        assert "fold_scores" in scores
        assert len(scores["fold_scores"]) == 3


class TestLightGBMRegressorSaveLoad:
    """Tests for save/load roundtrip."""

    def test_save_load_roundtrip(self, regression_df):
        """Test saving and loading a model."""
        from smallaxe.training.lightgbm import LightGBMRegressor

        model = LightGBMRegressor()
        model.set_param({"n_estimators": 30, "max_depth": 4})
        model.fit(
            regression_df,
            label_col="target",
            feature_cols=["age", "income"],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model")
            model.save(save_path)

            loaded_model = LightGBMRegressor.load(save_path)

            assert loaded_model._is_fitted
            assert loaded_model.get_param("n_estimators") == 30
            assert loaded_model.get_param("max_depth") == 4
            assert loaded_model._feature_cols == ["age", "income"]
            assert loaded_model._label_col == "target"

    def test_loaded_model_can_predict(self, regression_df):
        """Test that loaded model can make predictions."""
        from smallaxe.training.lightgbm import LightGBMRegressor

        model = LightGBMRegressor()
        model.fit(
            regression_df,
            label_col="target",
            feature_cols=["age", "income"],
        )

        original_predictions = model.predict(regression_df).collect()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model")
            model.save(save_path)

            loaded_model = LightGBMRegressor.load(save_path)
            loaded_predictions = loaded_model.predict(regression_df).collect()

        assert len(loaded_predictions) == len(original_predictions)

    def test_save_before_fit_raises_error(self, regression_df):
        """Test that saving before fit raises error."""
        from smallaxe.training.lightgbm import LightGBMRegressor

        model = LightGBMRegressor()
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model")
            with pytest.raises(ModelNotFittedError):
                model.save(save_path)

    def test_loaded_model_preserves_validation_scores(self, regression_df):
        """Test that loaded model preserves validation scores."""
        from smallaxe.training.lightgbm import LightGBMRegressor

        model = LightGBMRegressor()
        model.fit(
            regression_df,
            label_col="target",
            feature_cols=["age", "income"],
            validation="train_test",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model")
            model.save(save_path)

            loaded_model = LightGBMRegressor.load(save_path)

        assert loaded_model.validation_scores is not None
        assert loaded_model.validation_scores["validation_type"] == "train_test"


# =============================================================================
# LightGBMClassifier Tests
# =============================================================================


class TestLightGBMClassifierInit:
    """Tests for LightGBMClassifier initialization."""

    def test_default_task(self):
        """Test that default task is 'binary'."""
        from smallaxe.training.lightgbm import LightGBMClassifier

        model = LightGBMClassifier()
        assert model.task == "binary"
        assert model.task_type == "classification"

    def test_explicit_binary_task(self):
        """Test that explicit binary task is set correctly."""
        from smallaxe.training.lightgbm import LightGBMClassifier

        model = LightGBMClassifier(task="binary")
        assert model.task == "binary"

    def test_explicit_multiclass_task(self):
        """Test that explicit multiclass task is set correctly."""
        from smallaxe.training.lightgbm import LightGBMClassifier

        model = LightGBMClassifier(task="multiclass")
        assert model.task == "multiclass"

    def test_invalid_task_raises_error(self):
        """Test that invalid task raises ValidationError."""
        from smallaxe.training.lightgbm import LightGBMClassifier

        with pytest.raises(ValidationError, match="Invalid classification task"):
            LightGBMClassifier(task="simple_regression")


class TestLightGBMClassifierParams:
    """Tests for params and default_params."""

    def test_params_dict(self):
        """Test that params returns parameter descriptions."""
        from smallaxe.training.lightgbm import LightGBMClassifier

        model = LightGBMClassifier()
        params = model.params

        assert "n_estimators" in params
        assert "max_depth" in params
        assert "learning_rate" in params
        assert "num_leaves" in params
        assert "min_data_in_leaf" in params
        assert "feature_fraction" in params
        assert "bagging_fraction" in params
        assert "bagging_freq" in params
        assert "lambda_l1" in params
        assert "lambda_l2" in params
        assert "seed" in params

        # Check descriptions are strings
        for _key, value in params.items():
            assert isinstance(value, str)
            assert len(value) > 0

    def test_default_params_dict(self):
        """Test that default_params returns default values."""
        from smallaxe.training.lightgbm import LightGBMClassifier

        model = LightGBMClassifier()
        defaults = model.default_params

        assert defaults["n_estimators"] == 100
        assert defaults["max_depth"] == -1
        assert defaults["learning_rate"] == 0.1
        assert defaults["num_leaves"] == 31
        assert defaults["min_data_in_leaf"] == 20
        assert defaults["feature_fraction"] == 1.0
        assert defaults["bagging_fraction"] == 1.0
        assert defaults["bagging_freq"] == 0
        assert defaults["lambda_l1"] == 0.0
        assert defaults["lambda_l2"] == 0.0
        assert defaults["seed"] is None


class TestLightGBMClassifierFit:
    """Tests for fit method."""

    def test_fit_returns_self(self, classification_df):
        """Test that fit returns self for method chaining."""
        from smallaxe.training.lightgbm import LightGBMClassifier

        model = LightGBMClassifier()
        result = model.fit(
            classification_df,
            label_col="label",
            feature_cols=["age", "income"],
        )
        assert result is model

    def test_fit_marks_model_as_fitted(self, classification_df):
        """Test that fit marks the model as fitted."""
        from smallaxe.training.lightgbm import LightGBMClassifier

        model = LightGBMClassifier()
        assert not model._is_fitted
        model.fit(
            classification_df,
            label_col="label",
            feature_cols=["age", "income"],
        )
        assert model._is_fitted

    def test_fit_multiclass(self, multiclass_df):
        """Test fitting on multiclass data."""
        from smallaxe.training.lightgbm import LightGBMClassifier

        model = LightGBMClassifier(task="multiclass")
        model.fit(
            multiclass_df,
            label_col="label",
            feature_cols=["age", "income"],
        )
        assert model._is_fitted


class TestLightGBMClassifierPredict:
    """Tests for predict method."""

    def test_predict_before_fit_raises_error(self, classification_df):
        """Test that predict before fit raises ModelNotFittedError."""
        from smallaxe.training.lightgbm import LightGBMClassifier

        model = LightGBMClassifier()
        with pytest.raises(ModelNotFittedError, match="Model has not been fitted"):
            model.predict(classification_df)

    def test_predict_returns_dataframe(self, classification_df):
        """Test that predict returns a DataFrame."""
        from smallaxe.training.lightgbm import LightGBMClassifier

        model = LightGBMClassifier()
        model.fit(
            classification_df,
            label_col="label",
            feature_cols=["age", "income"],
        )
        result = model.predict(classification_df)
        assert result is not None
        assert result.count() == classification_df.count()

    def test_predict_adds_default_column(self, classification_df):
        """Test that predict adds default prediction column."""
        from smallaxe.training.lightgbm import LightGBMClassifier

        model = LightGBMClassifier()
        model.fit(
            classification_df,
            label_col="label",
            feature_cols=["age", "income"],
        )
        result = model.predict(classification_df)
        assert "predict_label" in result.columns

    def test_predict_values_are_class_labels(self, classification_df):
        """Test that predictions are valid class labels."""
        from smallaxe.training.lightgbm import LightGBMClassifier

        model = LightGBMClassifier()
        model.fit(
            classification_df,
            label_col="label",
            feature_cols=["age", "income"],
        )
        result = model.predict(classification_df)
        predictions = [row.predict_label for row in result.collect()]
        # All predictions should be 0 or 1 for binary classification
        assert all(p in [0.0, 1.0] for p in predictions)


class TestLightGBMClassifierPredictProba:
    """Tests for predict_proba method."""

    def test_predict_proba_before_fit_raises_error(self, classification_df):
        """Test that predict_proba before fit raises ModelNotFittedError."""
        from smallaxe.training.lightgbm import LightGBMClassifier

        model = LightGBMClassifier()
        with pytest.raises(ModelNotFittedError, match="Model has not been fitted"):
            model.predict_proba(classification_df)

    def test_predict_proba_returns_dataframe(self, classification_df):
        """Test that predict_proba returns a DataFrame."""
        from smallaxe.training.lightgbm import LightGBMClassifier

        model = LightGBMClassifier()
        model.fit(
            classification_df,
            label_col="label",
            feature_cols=["age", "income"],
        )
        result = model.predict_proba(classification_df)
        assert result is not None
        assert result.count() == classification_df.count()

    def test_predict_proba_adds_probability_column(self, classification_df):
        """Test that predict_proba adds probability column."""
        from smallaxe.training.lightgbm import LightGBMClassifier

        model = LightGBMClassifier()
        model.fit(
            classification_df,
            label_col="label",
            feature_cols=["age", "income"],
        )
        result = model.predict_proba(classification_df)
        assert "probability" in result.columns

    def test_predict_proba_custom_output_col(self, classification_df):
        """Test that predict_proba uses custom output column name."""
        from smallaxe.training.lightgbm import LightGBMClassifier

        model = LightGBMClassifier()
        model.fit(
            classification_df,
            label_col="label",
            feature_cols=["age", "income"],
        )
        result = model.predict_proba(classification_df, output_col="my_proba")
        assert "my_proba" in result.columns
        assert "probability" not in result.columns


class TestLightGBMClassifierValidation:
    """Tests for validation strategies."""

    def test_validation_train_test(self, classification_df):
        """Test validation='train_test' produces validation scores."""
        from smallaxe.training.lightgbm import LightGBMClassifier

        model = LightGBMClassifier()
        model.fit(
            classification_df,
            label_col="label",
            feature_cols=["age", "income"],
            validation="train_test",
            test_size=0.3,
        )

        scores = model.validation_scores
        assert scores is not None
        assert scores["validation_type"] == "train_test"
        assert "accuracy" in scores
        assert "precision" in scores
        assert "recall" in scores
        assert "f1_score" in scores

    def test_validation_kfold(self, classification_df):
        """Test validation='kfold' produces validation scores."""
        from smallaxe.training.lightgbm import LightGBMClassifier

        model = LightGBMClassifier()
        model.fit(
            classification_df,
            label_col="label",
            feature_cols=["age", "income"],
            validation="kfold",
            n_folds=3,
        )

        scores = model.validation_scores
        assert scores is not None
        assert scores["validation_type"] == "kfold"
        assert scores["n_folds"] == 3
        assert "mean_accuracy" in scores
        assert "fold_scores" in scores
        assert len(scores["fold_scores"]) == 3


class TestLightGBMClassifierSaveLoad:
    """Tests for save/load roundtrip."""

    def test_save_load_roundtrip(self, classification_df):
        """Test saving and loading a classifier."""
        from smallaxe.training.lightgbm import LightGBMClassifier

        model = LightGBMClassifier()
        model.set_param({"n_estimators": 30, "max_depth": 4})
        model.fit(
            classification_df,
            label_col="label",
            feature_cols=["age", "income"],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model")
            model.save(save_path)

            loaded_model = LightGBMClassifier.load(save_path)

            assert loaded_model._is_fitted
            assert loaded_model.get_param("n_estimators") == 30
            assert loaded_model.get_param("max_depth") == 4
            assert loaded_model._feature_cols == ["age", "income"]
            assert loaded_model._label_col == "label"

    def test_loaded_model_can_predict(self, classification_df):
        """Test that loaded model can make predictions."""
        from smallaxe.training.lightgbm import LightGBMClassifier

        model = LightGBMClassifier()
        model.fit(
            classification_df,
            label_col="label",
            feature_cols=["age", "income"],
        )

        original_predictions = model.predict(classification_df).collect()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model")
            model.save(save_path)

            loaded_model = LightGBMClassifier.load(save_path)
            loaded_predictions = loaded_model.predict(classification_df).collect()

        assert len(loaded_predictions) == len(original_predictions)

    def test_loaded_model_can_predict_proba(self, classification_df):
        """Test that loaded model can make probability predictions."""
        from smallaxe.training.lightgbm import LightGBMClassifier

        model = LightGBMClassifier()
        model.fit(
            classification_df,
            label_col="label",
            feature_cols=["age", "income"],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model")
            model.save(save_path)

            loaded_model = LightGBMClassifier.load(save_path)
            proba_result = loaded_model.predict_proba(classification_df)

        assert "probability" in proba_result.columns
        assert proba_result.count() == classification_df.count()

    def test_loaded_model_preserves_validation_scores(self, classification_df):
        """Test that loaded model preserves validation scores."""
        from smallaxe.training.lightgbm import LightGBMClassifier

        model = LightGBMClassifier()
        model.fit(
            classification_df,
            label_col="label",
            feature_cols=["age", "income"],
            validation="train_test",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model")
            model.save(save_path)

            loaded_model = LightGBMClassifier.load(save_path)

        assert loaded_model.validation_scores is not None
        assert loaded_model.validation_scores["validation_type"] == "train_test"


# =============================================================================
# DependencyError Tests (run always, even without LightGBM)
# =============================================================================


@pytest.mark.skipif(
    LIGHTGBM_AVAILABLE,
    reason="Test only runs when synapseml is NOT installed",
)
class TestLightGBMDependencyError:
    """Tests for DependencyError when synapseml is not installed."""

    def test_regressor_raises_dependency_error(self):
        """Test that LightGBMRegressor raises DependencyError when synapseml not installed."""
        from smallaxe.training.lightgbm import LightGBMRegressor

        with pytest.raises(DependencyError, match="synapseml is not installed"):
            LightGBMRegressor()

    def test_classifier_raises_dependency_error(self):
        """Test that LightGBMClassifier raises DependencyError when synapseml not installed."""
        from smallaxe.training.lightgbm import LightGBMClassifier

        with pytest.raises(DependencyError, match="synapseml is not installed"):
            LightGBMClassifier()
