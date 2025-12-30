"""Tests for RandomForestRegressor and RandomForestClassifier."""

import os
import tempfile

import pytest

from smallaxe.exceptions import ModelNotFittedError, ValidationError
from smallaxe.training import RandomForestClassifier, RandomForestRegressor


@pytest.fixture
def regression_df(spark_session):
    """Create a sample DataFrame for regression testing."""
    data = [(i, 20.0 + (i % 30), 40000.0 + (i * 1000), 50.0 + (i * 5)) for i in range(1, 101)]
    columns = ["id", "age", "income", "target"]
    return spark_session.createDataFrame(data, columns)


class TestRandomForestRegressorInit:
    """Tests for RandomForestRegressor initialization."""

    def test_default_task(self):
        """Test that default task is 'simple_regression'."""
        model = RandomForestRegressor()
        assert model.task == "simple_regression"
        assert model.task_type == "regression"

    def test_explicit_task(self):
        """Test that explicit task is set correctly."""
        model = RandomForestRegressor(task="simple_regression")
        assert model.task == "simple_regression"

    def test_invalid_task_raises_error(self):
        """Test that invalid task raises ValidationError."""
        with pytest.raises(ValidationError, match="Invalid regression task"):
            RandomForestRegressor(task="binary")


class TestRandomForestRegressorParams:
    """Tests for params and default_params."""

    def test_params_dict(self):
        """Test that params returns parameter descriptions."""
        model = RandomForestRegressor()
        params = model.params

        assert "n_estimators" in params
        assert "max_depth" in params
        assert "max_bins" in params
        assert "min_instances_per_node" in params
        assert "min_info_gain" in params
        assert "subsampling_rate" in params
        assert "feature_subset_strategy" in params
        assert "seed" in params

        # Check descriptions are strings
        for _key, value in params.items():
            assert isinstance(value, str)
            assert len(value) > 0

    def test_default_params_dict(self):
        """Test that default_params returns default values."""
        model = RandomForestRegressor()
        defaults = model.default_params

        assert defaults["n_estimators"] == 20
        assert defaults["max_depth"] == 5
        assert defaults["max_bins"] == 32
        assert defaults["min_instances_per_node"] == 1
        assert defaults["min_info_gain"] == 0.0
        assert defaults["subsampling_rate"] == 1.0
        assert defaults["feature_subset_strategy"] == "auto"
        assert defaults["seed"] is None

    def test_get_param_returns_default(self):
        """Test that get_param returns default value when not set."""
        model = RandomForestRegressor()
        assert model.get_param("n_estimators") == 20
        assert model.get_param("max_depth") == 5


class TestRandomForestRegressorSetParam:
    """Tests for set_param method."""

    def test_set_param_single(self):
        """Test setting a single parameter."""
        model = RandomForestRegressor()
        model.set_param({"n_estimators": 100})
        assert model.get_param("n_estimators") == 100

    def test_set_param_multiple(self):
        """Test setting multiple parameters."""
        model = RandomForestRegressor()
        model.set_param(
            {
                "n_estimators": 50,
                "max_depth": 10,
                "subsampling_rate": 0.8,
            }
        )
        assert model.get_param("n_estimators") == 50
        assert model.get_param("max_depth") == 10
        assert model.get_param("subsampling_rate") == 0.8

    def test_set_param_invalid_key(self):
        """Test that invalid parameter key raises ValidationError."""
        model = RandomForestRegressor()
        with pytest.raises(ValidationError, match="Invalid parameter"):
            model.set_param({"invalid_param": 10})

    def test_set_param_returns_self(self):
        """Test that set_param returns self for method chaining."""
        model = RandomForestRegressor()
        result = model.set_param({"n_estimators": 100})
        assert result is model


class TestRandomForestRegressorFit:
    """Tests for fit method."""

    def test_fit_returns_self(self, regression_df):
        """Test that fit returns self for method chaining."""
        model = RandomForestRegressor()
        result = model.fit(
            regression_df,
            label_col="target",
            feature_cols=["age", "income"],
        )
        assert result is model

    def test_fit_marks_model_as_fitted(self, regression_df):
        """Test that fit marks the model as fitted."""
        model = RandomForestRegressor()
        assert not model._is_fitted
        model.fit(
            regression_df,
            label_col="target",
            feature_cols=["age", "income"],
        )
        assert model._is_fitted

    def test_fit_infers_feature_cols(self, regression_df):
        """Test that fit infers feature columns when not provided."""
        model = RandomForestRegressor()
        model.fit(
            regression_df,
            label_col="target",
            exclude_cols=["id"],
        )
        assert set(model._feature_cols) == {"age", "income"}

    def test_fit_with_custom_params(self, regression_df):
        """Test that fit uses custom parameters."""
        model = RandomForestRegressor()
        model.set_param({"n_estimators": 50, "max_depth": 10})
        model.fit(
            regression_df,
            label_col="target",
            feature_cols=["age", "income"],
        )
        assert model._is_fitted


class TestRandomForestRegressorPredict:
    """Tests for predict method."""

    def test_predict_before_fit_raises_error(self, regression_df):
        """Test that predict before fit raises ModelNotFittedError."""
        model = RandomForestRegressor()
        with pytest.raises(ModelNotFittedError, match="Model has not been fitted"):
            model.predict(regression_df)

    def test_predict_returns_dataframe(self, regression_df):
        """Test that predict returns a DataFrame."""
        model = RandomForestRegressor()
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
        model = RandomForestRegressor()
        model.fit(
            regression_df,
            label_col="target",
            feature_cols=["age", "income"],
        )
        result = model.predict(regression_df)
        assert "predict_label" in result.columns

    def test_predict_custom_output_col(self, regression_df):
        """Test that predict uses custom output column name."""
        model = RandomForestRegressor()
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
        model = RandomForestRegressor()
        model.fit(
            regression_df,
            label_col="target",
            feature_cols=["age", "income"],
        )
        result = model.predict(regression_df)
        for col in ["id", "age", "income", "target"]:
            assert col in result.columns


class TestRandomForestRegressorValidation:
    """Tests for validation strategies."""

    def test_validation_none(self, regression_df):
        """Test validation='none' results in no validation scores."""
        model = RandomForestRegressor()
        model.fit(
            regression_df,
            label_col="target",
            feature_cols=["age", "income"],
            validation="none",
        )
        assert model.validation_scores is None

    def test_validation_train_test(self, regression_df):
        """Test validation='train_test' produces validation scores."""
        model = RandomForestRegressor()
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
        assert "mape" in scores

    def test_validation_kfold(self, regression_df):
        """Test validation='kfold' produces validation scores."""
        model = RandomForestRegressor()
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

    def test_kfold_fold_scores_have_metrics(self, regression_df):
        """Test that k-fold individual fold scores have metrics."""
        model = RandomForestRegressor()
        model.fit(
            regression_df,
            label_col="target",
            feature_cols=["age", "income"],
            validation="kfold",
            n_folds=3,
        )

        fold_scores = model.validation_scores["fold_scores"]
        for fold in fold_scores:
            assert "fold" in fold
            assert "rmse" in fold
            assert "mae" in fold
            assert "r2" in fold


class TestRandomForestRegressorMetadata:
    """Tests for metadata after fit."""

    def test_metadata_populated_after_fit(self, regression_df):
        """Test that metadata is populated after fit."""
        model = RandomForestRegressor()
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
        assert metadata["n_samples"] == 100
        assert metadata["n_features"] == 2
        assert metadata["task"] == "simple_regression"
        assert metadata["task_type"] == "regression"

    def test_metadata_contains_label_stats(self, regression_df):
        """Test that metadata contains label statistics for regression."""
        model = RandomForestRegressor()
        model.fit(
            regression_df,
            label_col="target",
            feature_cols=["age", "income"],
        )

        metadata = model.metadata
        assert "label_min" in metadata
        assert "label_max" in metadata
        assert "label_mean" in metadata


class TestRandomForestRegressorFeatureImportance:
    """Tests for feature importance."""

    def test_feature_importances_after_fit(self, regression_df):
        """Test that feature_importances is available after fit."""
        model = RandomForestRegressor()
        model.fit(
            regression_df,
            label_col="target",
            feature_cols=["age", "income"],
        )

        importances = model.feature_importances
        assert importances is not None
        assert isinstance(importances, dict)
        assert "age" in importances
        assert "income" in importances

    def test_feature_importances_sum_to_one(self, regression_df):
        """Test that feature importances sum to approximately 1."""
        model = RandomForestRegressor()
        model.fit(
            regression_df,
            label_col="target",
            feature_cols=["age", "income"],
        )

        importances = model.feature_importances
        total = sum(importances.values())
        assert abs(total - 1.0) < 0.01

    def test_feature_importances_before_fit_raises_error(self, regression_df):
        """Test that feature_importances before fit raises error."""
        model = RandomForestRegressor()
        with pytest.raises(ModelNotFittedError):
            _ = model.feature_importances


class TestRandomForestRegressorSaveLoad:
    """Tests for save/load roundtrip."""

    def test_save_load_roundtrip(self, regression_df):
        """Test saving and loading a model."""
        model = RandomForestRegressor()
        model.set_param({"n_estimators": 30, "max_depth": 8})
        model.fit(
            regression_df,
            label_col="target",
            feature_cols=["age", "income"],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model")
            model.save(save_path)

            loaded_model = RandomForestRegressor.load(save_path)

            assert loaded_model._is_fitted
            assert loaded_model.get_param("n_estimators") == 30
            assert loaded_model.get_param("max_depth") == 8
            assert loaded_model._feature_cols == ["age", "income"]
            assert loaded_model._label_col == "target"

    def test_loaded_model_can_predict(self, regression_df):
        """Test that loaded model can make predictions."""
        model = RandomForestRegressor()
        model.fit(
            regression_df,
            label_col="target",
            feature_cols=["age", "income"],
        )

        original_predictions = model.predict(regression_df).collect()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model")
            model.save(save_path)

            loaded_model = RandomForestRegressor.load(save_path)
            loaded_predictions = loaded_model.predict(regression_df).collect()

        assert len(loaded_predictions) == len(original_predictions)

    def test_loaded_model_has_feature_importances(self, regression_df):
        """Test that loaded model has feature importances."""
        model = RandomForestRegressor()
        model.fit(
            regression_df,
            label_col="target",
            feature_cols=["age", "income"],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model")
            model.save(save_path)

            loaded_model = RandomForestRegressor.load(save_path)
            importances = loaded_model.feature_importances

        assert importances is not None
        assert "age" in importances
        assert "income" in importances

    def test_save_before_fit_raises_error(self, regression_df):
        """Test that saving before fit raises error."""
        model = RandomForestRegressor()
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model")
            with pytest.raises(ModelNotFittedError):
                model.save(save_path)

    def test_loaded_model_preserves_metadata(self, regression_df):
        """Test that loaded model preserves metadata."""
        model = RandomForestRegressor()
        model.fit(
            regression_df,
            label_col="target",
            feature_cols=["age", "income"],
            validation="train_test",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model")
            model.save(save_path)

            loaded_model = RandomForestRegressor.load(save_path)

        assert loaded_model.metadata is not None
        assert loaded_model.metadata["task"] == "simple_regression"

    def test_loaded_model_preserves_validation_scores(self, regression_df):
        """Test that loaded model preserves validation scores."""
        model = RandomForestRegressor()
        model.fit(
            regression_df,
            label_col="target",
            feature_cols=["age", "income"],
            validation="train_test",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model")
            model.save(save_path)

            loaded_model = RandomForestRegressor.load(save_path)

        assert loaded_model.validation_scores is not None
        assert loaded_model.validation_scores["validation_type"] == "train_test"


class TestRandomForestRegressorCacheStrategy:
    """Tests for cache strategy."""

    def test_cache_strategy_none(self, regression_df):
        """Test cache_strategy='none' works."""
        model = RandomForestRegressor()
        model.fit(
            regression_df,
            label_col="target",
            feature_cols=["age", "income"],
            cache_strategy="none",
        )
        assert model._is_fitted

    def test_cache_strategy_memory(self, regression_df):
        """Test cache_strategy='memory' works."""
        model = RandomForestRegressor()
        model.fit(
            regression_df,
            label_col="target",
            feature_cols=["age", "income"],
            cache_strategy="memory",
        )
        assert model._is_fitted

    def test_cache_strategy_disk(self, regression_df):
        """Test cache_strategy='disk' works."""
        model = RandomForestRegressor()
        model.fit(
            regression_df,
            label_col="target",
            feature_cols=["age", "income"],
            cache_strategy="disk",
        )
        assert model._is_fitted


# =============================================================================
# RandomForestClassifier Tests
# =============================================================================


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


class TestRandomForestClassifierInit:
    """Tests for RandomForestClassifier initialization."""

    def test_default_task(self):
        """Test that default task is 'binary'."""
        model = RandomForestClassifier()
        assert model.task == "binary"
        assert model.task_type == "classification"

    def test_explicit_binary_task(self):
        """Test that explicit binary task is set correctly."""
        model = RandomForestClassifier(task="binary")
        assert model.task == "binary"

    def test_explicit_multiclass_task(self):
        """Test that explicit multiclass task is set correctly."""
        model = RandomForestClassifier(task="multiclass")
        assert model.task == "multiclass"

    def test_invalid_task_raises_error(self):
        """Test that invalid task raises ValidationError."""
        with pytest.raises(ValidationError, match="Invalid classification task"):
            RandomForestClassifier(task="simple_regression")


class TestRandomForestClassifierParams:
    """Tests for params and default_params."""

    def test_params_dict(self):
        """Test that params returns parameter descriptions."""
        model = RandomForestClassifier()
        params = model.params

        assert "n_estimators" in params
        assert "max_depth" in params
        assert "max_bins" in params
        assert "min_instances_per_node" in params
        assert "min_info_gain" in params
        assert "subsampling_rate" in params
        assert "feature_subset_strategy" in params
        assert "seed" in params

        # Check descriptions are strings
        for _key, value in params.items():
            assert isinstance(value, str)
            assert len(value) > 0

    def test_default_params_dict(self):
        """Test that default_params returns default values."""
        model = RandomForestClassifier()
        defaults = model.default_params

        assert defaults["n_estimators"] == 20
        assert defaults["max_depth"] == 5
        assert defaults["max_bins"] == 32
        assert defaults["min_instances_per_node"] == 1
        assert defaults["min_info_gain"] == 0.0
        assert defaults["subsampling_rate"] == 1.0
        assert defaults["feature_subset_strategy"] == "auto"
        assert defaults["seed"] is None


class TestRandomForestClassifierFit:
    """Tests for fit method."""

    def test_fit_returns_self(self, classification_df):
        """Test that fit returns self for method chaining."""
        model = RandomForestClassifier()
        result = model.fit(
            classification_df,
            label_col="label",
            feature_cols=["age", "income"],
        )
        assert result is model

    def test_fit_marks_model_as_fitted(self, classification_df):
        """Test that fit marks the model as fitted."""
        model = RandomForestClassifier()
        assert not model._is_fitted
        model.fit(
            classification_df,
            label_col="label",
            feature_cols=["age", "income"],
        )
        assert model._is_fitted

    def test_fit_multiclass(self, multiclass_df):
        """Test fitting on multiclass data."""
        model = RandomForestClassifier(task="multiclass")
        model.fit(
            multiclass_df,
            label_col="label",
            feature_cols=["age", "income"],
        )
        assert model._is_fitted


class TestRandomForestClassifierPredict:
    """Tests for predict method."""

    def test_predict_before_fit_raises_error(self, classification_df):
        """Test that predict before fit raises ModelNotFittedError."""
        model = RandomForestClassifier()
        with pytest.raises(ModelNotFittedError, match="Model has not been fitted"):
            model.predict(classification_df)

    def test_predict_returns_dataframe(self, classification_df):
        """Test that predict returns a DataFrame."""
        model = RandomForestClassifier()
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
        model = RandomForestClassifier()
        model.fit(
            classification_df,
            label_col="label",
            feature_cols=["age", "income"],
        )
        result = model.predict(classification_df)
        assert "predict_label" in result.columns

    def test_predict_values_are_class_labels(self, classification_df):
        """Test that predictions are valid class labels."""
        model = RandomForestClassifier()
        model.fit(
            classification_df,
            label_col="label",
            feature_cols=["age", "income"],
        )
        result = model.predict(classification_df)
        predictions = [row.predict_label for row in result.collect()]
        # All predictions should be 0 or 1 for binary classification
        assert all(p in [0.0, 1.0] for p in predictions)


class TestRandomForestClassifierPredictProba:
    """Tests for predict_proba method."""

    def test_predict_proba_before_fit_raises_error(self, classification_df):
        """Test that predict_proba before fit raises ModelNotFittedError."""
        model = RandomForestClassifier()
        with pytest.raises(ModelNotFittedError, match="Model has not been fitted"):
            model.predict_proba(classification_df)

    def test_predict_proba_returns_dataframe(self, classification_df):
        """Test that predict_proba returns a DataFrame."""
        model = RandomForestClassifier()
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
        model = RandomForestClassifier()
        model.fit(
            classification_df,
            label_col="label",
            feature_cols=["age", "income"],
        )
        result = model.predict_proba(classification_df)
        assert "probability" in result.columns

    def test_predict_proba_custom_output_col(self, classification_df):
        """Test that predict_proba uses custom output column name."""
        model = RandomForestClassifier()
        model.fit(
            classification_df,
            label_col="label",
            feature_cols=["age", "income"],
        )
        result = model.predict_proba(classification_df, output_col="my_proba")
        assert "my_proba" in result.columns
        assert "probability" not in result.columns

    def test_predict_proba_is_vector(self, classification_df):
        """Test that probability output is a DenseVector."""
        from pyspark.ml.linalg import DenseVector

        model = RandomForestClassifier()
        model.fit(
            classification_df,
            label_col="label",
            feature_cols=["age", "income"],
        )
        result = model.predict_proba(classification_df)
        first_row = result.first()
        assert isinstance(first_row.probability, DenseVector)
        # Binary classification should have 2 probability values
        assert len(first_row.probability) == 2

    def test_predict_proba_sums_to_one(self, classification_df):
        """Test that probability values sum to approximately 1."""
        model = RandomForestClassifier()
        model.fit(
            classification_df,
            label_col="label",
            feature_cols=["age", "income"],
        )
        result = model.predict_proba(classification_df)
        rows = result.collect()
        for row in rows:
            prob_sum = sum(row.probability)
            assert abs(prob_sum - 1.0) < 0.01


class TestRandomForestClassifierStratifiedValidation:
    """Tests for stratified validation strategies."""

    def test_validation_train_test_stratified(self, classification_df):
        """Test validation='train_test' with stratified=True (default for classifiers)."""
        model = RandomForestClassifier()
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

    def test_validation_kfold_stratified(self, classification_df):
        """Test validation='kfold' with stratified splits."""
        model = RandomForestClassifier()
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
        assert "mean_precision" in scores
        assert "mean_recall" in scores
        assert "mean_f1_score" in scores
        assert "fold_scores" in scores
        assert len(scores["fold_scores"]) == 3

    def test_kfold_fold_scores_have_classification_metrics(self, classification_df):
        """Test that k-fold individual fold scores have classification metrics."""
        model = RandomForestClassifier()
        model.fit(
            classification_df,
            label_col="label",
            feature_cols=["age", "income"],
            validation="kfold",
            n_folds=3,
        )

        fold_scores = model.validation_scores["fold_scores"]
        for fold in fold_scores:
            assert "fold" in fold
            assert "accuracy" in fold
            assert "precision" in fold
            assert "recall" in fold
            assert "f1_score" in fold

    def test_binary_validation_includes_auc_metrics(self, classification_df):
        """Test that binary classification includes AUC metrics."""
        model = RandomForestClassifier(task="binary")
        model.fit(
            classification_df,
            label_col="label",
            feature_cols=["age", "income"],
            validation="train_test",
            test_size=0.3,
        )

        scores = model.validation_scores
        # AUC metrics should be present for binary classification
        assert "auc_roc" in scores
        assert "auc_pr" in scores


class TestRandomForestClassifierMetadata:
    """Tests for metadata after fit."""

    def test_metadata_populated_after_fit(self, classification_df):
        """Test that metadata is populated after fit."""
        model = RandomForestClassifier()
        model.fit(
            classification_df,
            label_col="label",
            feature_cols=["age", "income"],
        )

        metadata = model.metadata
        assert "training_timestamp" in metadata
        assert metadata["label_col"] == "label"
        assert metadata["feature_cols"] == ["age", "income"]
        assert metadata["n_samples"] == 100
        assert metadata["n_features"] == 2
        assert metadata["task"] == "binary"
        assert metadata["task_type"] == "classification"

    def test_metadata_contains_class_distribution(self, classification_df):
        """Test that metadata contains class distribution for classification."""
        model = RandomForestClassifier()
        model.fit(
            classification_df,
            label_col="label",
            feature_cols=["age", "income"],
        )

        metadata = model.metadata
        assert "class_counts" in metadata


class TestRandomForestClassifierFeatureImportance:
    """Tests for feature importance."""

    def test_feature_importances_after_fit(self, classification_df):
        """Test that feature_importances is available after fit."""
        model = RandomForestClassifier()
        model.fit(
            classification_df,
            label_col="label",
            feature_cols=["age", "income"],
        )

        importances = model.feature_importances
        assert importances is not None
        assert isinstance(importances, dict)
        assert "age" in importances
        assert "income" in importances

    def test_feature_importances_sum_to_one(self, classification_df):
        """Test that feature importances sum to approximately 1."""
        model = RandomForestClassifier()
        model.fit(
            classification_df,
            label_col="label",
            feature_cols=["age", "income"],
        )

        importances = model.feature_importances
        total = sum(importances.values())
        assert abs(total - 1.0) < 0.01


class TestRandomForestClassifierSaveLoad:
    """Tests for save/load roundtrip."""

    def test_save_load_roundtrip(self, classification_df):
        """Test saving and loading a classifier."""
        model = RandomForestClassifier()
        model.set_param({"n_estimators": 30, "max_depth": 8})
        model.fit(
            classification_df,
            label_col="label",
            feature_cols=["age", "income"],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model")
            model.save(save_path)

            loaded_model = RandomForestClassifier.load(save_path)

            assert loaded_model._is_fitted
            assert loaded_model.get_param("n_estimators") == 30
            assert loaded_model.get_param("max_depth") == 8
            assert loaded_model._feature_cols == ["age", "income"]
            assert loaded_model._label_col == "label"

    def test_loaded_model_can_predict(self, classification_df):
        """Test that loaded model can make predictions."""
        model = RandomForestClassifier()
        model.fit(
            classification_df,
            label_col="label",
            feature_cols=["age", "income"],
        )

        original_predictions = model.predict(classification_df).collect()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model")
            model.save(save_path)

            loaded_model = RandomForestClassifier.load(save_path)
            loaded_predictions = loaded_model.predict(classification_df).collect()

        assert len(loaded_predictions) == len(original_predictions)

    def test_loaded_model_can_predict_proba(self, classification_df):
        """Test that loaded model can make probability predictions."""
        model = RandomForestClassifier()
        model.fit(
            classification_df,
            label_col="label",
            feature_cols=["age", "income"],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model")
            model.save(save_path)

            loaded_model = RandomForestClassifier.load(save_path)
            proba_result = loaded_model.predict_proba(classification_df)

        assert "probability" in proba_result.columns
        assert proba_result.count() == classification_df.count()

    def test_loaded_model_has_feature_importances(self, classification_df):
        """Test that loaded model has feature importances."""
        model = RandomForestClassifier()
        model.fit(
            classification_df,
            label_col="label",
            feature_cols=["age", "income"],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model")
            model.save(save_path)

            loaded_model = RandomForestClassifier.load(save_path)
            importances = loaded_model.feature_importances

        assert importances is not None
        assert "age" in importances
        assert "income" in importances

    def test_loaded_model_preserves_validation_scores(self, classification_df):
        """Test that loaded model preserves validation scores."""
        model = RandomForestClassifier()
        model.fit(
            classification_df,
            label_col="label",
            feature_cols=["age", "income"],
            validation="train_test",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model")
            model.save(save_path)

            loaded_model = RandomForestClassifier.load(save_path)

        assert loaded_model.validation_scores is not None
        assert loaded_model.validation_scores["validation_type"] == "train_test"
