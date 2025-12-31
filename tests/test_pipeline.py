"""Tests for the Pipeline class."""

import os
import shutil
import tempfile

import pytest
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from smallaxe.exceptions import ModelNotFittedError, PreprocessingError, ValidationError
from smallaxe.pipeline import Pipeline
from smallaxe.preprocessing import Encoder, Imputer, Scaler


class MockModel:
    """A mock model for testing pipelines with models."""

    def __init__(self):
        self._is_fitted = False
        self._feature_cols = None
        self._label_col = None

    def fit(self, df: DataFrame, label_col: str, feature_cols: list) -> "MockModel":
        """Fit the mock model."""
        self._is_fitted = True
        self._feature_cols = feature_cols
        self._label_col = label_col
        return self

    def predict(self, df: DataFrame) -> DataFrame:
        """Make predictions (returns constant value)."""
        if not self._is_fitted:
            raise ModelNotFittedError(
                component="MockModel",
                message="MockModel has not been fitted.",
            )
        return df.withColumn("prediction", F.lit(1.0))


class RandomForestRegressor:
    """Mock RandomForestRegressor for testing preprocessing requirements."""

    def __init__(self):
        self._is_fitted = False

    def fit(self, df: DataFrame, label_col: str, feature_cols: list) -> "RandomForestRegressor":
        self._is_fitted = True
        return self

    def predict(self, df: DataFrame) -> DataFrame:
        return df.withColumn("prediction", F.lit(1.0))


class RandomForestClassifier:
    """Mock RandomForestClassifier for testing preprocessing requirements."""

    def __init__(self):
        self._is_fitted = False

    def fit(self, df: DataFrame, label_col: str, feature_cols: list) -> "RandomForestClassifier":
        self._is_fitted = True
        return self

    def predict(self, df: DataFrame) -> DataFrame:
        return df.withColumn("prediction", F.lit(1.0))


class XGBoostRegressor:
    """Mock XGBoostRegressor for testing preprocessing requirements."""

    def __init__(self):
        self._is_fitted = False

    def fit(self, df: DataFrame, label_col: str, feature_cols: list) -> "XGBoostRegressor":
        self._is_fitted = True
        return self

    def predict(self, df: DataFrame) -> DataFrame:
        return df.withColumn("prediction", F.lit(1.0))


class CatBoostRegressor:
    """Mock CatBoostRegressor for testing preprocessing requirements."""

    def __init__(self):
        self._is_fitted = False

    def fit(self, df: DataFrame, label_col: str, feature_cols: list) -> "CatBoostRegressor":
        self._is_fitted = True
        return self

    def predict(self, df: DataFrame) -> DataFrame:
        return df.withColumn("prediction", F.lit(1.0))


@pytest.fixture
def df_with_nulls(spark_session):
    """Create a DataFrame with null values for testing."""
    data = [
        (1, 25.0, 50000.0, "A", 100.0),
        (2, 30.0, None, "B", 150.0),
        (3, None, 70000.0, "A", 200.0),
        (4, 40.0, 80000.0, None, 250.0),
        (5, 45.0, 90000.0, "B", 300.0),
    ]
    columns = ["id", "age", "income", "category", "target"]
    return spark_session.createDataFrame(data, columns)


@pytest.fixture
def df_clean(spark_session):
    """Create a clean DataFrame without nulls."""
    data = [
        (1, 25.0, 50000.0, "A", 100.0),
        (2, 30.0, 60000.0, "B", 150.0),
        (3, 35.0, 70000.0, "A", 200.0),
        (4, 40.0, 80000.0, "C", 250.0),
        (5, 45.0, 90000.0, "B", 300.0),
    ]
    columns = ["id", "age", "income", "category", "target"]
    return spark_session.createDataFrame(data, columns)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for save/load tests."""
    path = tempfile.mkdtemp()
    yield path
    shutil.rmtree(path, ignore_errors=True)


class TestPipelineInit:
    """Tests for Pipeline initialization."""

    def test_create_pipeline_with_single_step(self):
        """Test creating a pipeline with a single step."""
        pipeline = Pipeline([("imputer", Imputer())])
        assert len(pipeline) == 1
        assert "imputer" in pipeline.named_steps

    def test_create_pipeline_with_multiple_steps(self):
        """Test creating a pipeline with multiple steps."""
        pipeline = Pipeline(
            [
                ("imputer", Imputer()),
                ("scaler", Scaler()),
                ("encoder", Encoder()),
            ]
        )
        assert len(pipeline) == 3

    def test_empty_steps_raises_error(self):
        """Test that empty steps list raises ValidationError."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            Pipeline([])

    def test_duplicate_names_raises_error(self):
        """Test that duplicate step names raise ValidationError."""
        with pytest.raises(ValidationError, match="must be unique"):
            Pipeline(
                [
                    ("step1", Imputer()),
                    ("step1", Scaler()),
                ]
            )

    def test_invalid_step_format_raises_error(self):
        """Test that invalid step format raises ValidationError."""
        with pytest.raises(ValidationError, match="must be a \\(name, step\\) tuple"):
            Pipeline([("imputer",)])  # Missing step object

    def test_empty_step_name_raises_error(self):
        """Test that empty step name raises ValidationError."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            Pipeline([("", Imputer())])

    def test_non_string_step_name_raises_error(self):
        """Test that non-string step name raises ValidationError."""
        with pytest.raises(ValidationError, match="must be a string"):
            Pipeline([(123, Imputer())])


class TestPipelineStepOrder:
    """Tests for pipeline step order validation."""

    def test_valid_preprocessing_order(self):
        """Test that valid preprocessing order is accepted."""
        pipeline = Pipeline(
            [
                ("imputer", Imputer()),
                ("scaler", Scaler()),
                ("encoder", Encoder()),
            ]
        )
        assert len(pipeline) == 3

    def test_valid_preprocessing_with_model(self):
        """Test that preprocessing before model is accepted."""
        pipeline = Pipeline(
            [
                ("imputer", Imputer()),
                ("scaler", Scaler()),
                ("model", MockModel()),
            ]
        )
        assert len(pipeline) == 3

    def test_invalid_order_preprocessing_after_model(self):
        """Test that preprocessing after model raises ValidationError."""
        with pytest.raises(ValidationError, match="Invalid step order"):
            Pipeline(
                [
                    ("model", MockModel()),
                    ("imputer", Imputer()),
                ]
            )

    def test_invalid_order_scaler_after_model(self):
        """Test that scaler after model raises ValidationError."""
        with pytest.raises(ValidationError, match="Invalid step order"):
            Pipeline(
                [
                    ("imputer", Imputer()),
                    ("model", MockModel()),
                    ("scaler", Scaler()),
                ]
            )


class TestPipelinePreprocessingOnly:
    """Tests for preprocessing-only pipelines."""

    def test_fit_single_imputer(self, df_with_nulls):
        """Test fitting a pipeline with just an imputer."""
        pipeline = Pipeline(
            [("imputer", Imputer(numerical_strategy="mean", categorical_strategy="most_frequent"))]
        )
        pipeline.fit(
            df_with_nulls,
            numerical_cols=["age", "income"],
            categorical_cols=["category"],
        )
        assert pipeline._is_fitted

    def test_transform_single_imputer(self, df_with_nulls):
        """Test transforming with a single imputer pipeline."""
        pipeline = Pipeline(
            [("imputer", Imputer(numerical_strategy="mean", categorical_strategy="most_frequent"))]
        )
        pipeline.fit(
            df_with_nulls,
            numerical_cols=["age", "income"],
            categorical_cols=["category"],
        )
        result = pipeline.transform(df_with_nulls)

        # Check no nulls remain in imputed columns
        null_count = result.filter(
            F.col("age").isNull() | F.col("income").isNull() | F.col("category").isNull()
        ).count()
        assert null_count == 0

    def test_fit_transform_pipeline(self, df_with_nulls):
        """Test fit_transform convenience method."""
        pipeline = Pipeline(
            [("imputer", Imputer(numerical_strategy="mean", categorical_strategy="most_frequent"))]
        )
        result = pipeline.fit_transform(
            df_with_nulls,
            numerical_cols=["age", "income"],
            categorical_cols=["category"],
        )

        # Check no nulls remain in imputed columns
        null_count = result.filter(
            F.col("age").isNull() | F.col("income").isNull() | F.col("category").isNull()
        ).count()
        assert null_count == 0

    def test_multi_step_preprocessing_pipeline(self, df_with_nulls):
        """Test pipeline with multiple preprocessing steps."""
        pipeline = Pipeline(
            [
                (
                    "imputer",
                    Imputer(numerical_strategy="mean", categorical_strategy="most_frequent"),
                ),
                ("scaler", Scaler()),
            ]
        )
        result = pipeline.fit_transform(
            df_with_nulls,
            numerical_cols=["age", "income"],
            categorical_cols=["category"],
        )

        # Check no nulls remain
        null_count = result.filter(F.col("age").isNull() | F.col("income").isNull()).count()
        assert null_count == 0

        # Check scaling was applied (values should be different from original)
        original_age = df_with_nulls.filter(F.col("id") == 1).first()["age"]
        scaled_age = result.filter(F.col("id") == 1).first()["age"]
        # After scaling, values are normalized
        assert scaled_age != original_age

    def test_imputer_scaler_encoder_pipeline(self, df_with_nulls):
        """Test pipeline with imputer, scaler, and encoder."""
        pipeline = Pipeline(
            [
                (
                    "imputer",
                    Imputer(numerical_strategy="mean", categorical_strategy="most_frequent"),
                ),
                ("scaler", Scaler()),
                ("encoder", Encoder(method="label")),
            ]
        )
        result = pipeline.fit_transform(
            df_with_nulls,
            numerical_cols=["age", "income"],
            categorical_cols=["category"],
        )

        # Check encoding was applied (category should be integers now)
        categories = [row["category"] for row in result.collect()]
        assert all(isinstance(c, (int, float)) for c in categories)


class TestPipelineWithModel:
    """Tests for pipelines with models."""

    def test_fit_pipeline_with_model(self, df_clean):
        """Test fitting a pipeline with a model."""
        pipeline = Pipeline(
            [
                (
                    "imputer",
                    Imputer(numerical_strategy="mean", categorical_strategy="most_frequent"),
                ),
                ("model", MockModel()),
            ]
        )
        pipeline.fit(
            df_clean,
            label_col="target",
            numerical_cols=["age", "income"],
            categorical_cols=["category"],
        )
        assert pipeline._is_fitted

    def test_predict_with_model(self, df_clean):
        """Test predict with a pipeline containing a model."""
        pipeline = Pipeline(
            [
                (
                    "imputer",
                    Imputer(numerical_strategy="mean", categorical_strategy="most_frequent"),
                ),
                ("model", MockModel()),
            ]
        )
        pipeline.fit(
            df_clean,
            label_col="target",
            numerical_cols=["age", "income"],
            categorical_cols=["category"],
        )
        result = pipeline.predict(df_clean)

        # Check prediction column was added
        assert "prediction" in result.columns

    def test_predict_without_model_raises_error(self, df_clean):
        """Test that predict on preprocessing-only pipeline raises error."""
        pipeline = Pipeline(
            [("imputer", Imputer(numerical_strategy="mean", categorical_strategy="most_frequent"))]
        )
        pipeline.fit(
            df_clean,
            numerical_cols=["age", "income"],
            categorical_cols=["category"],
        )

        with pytest.raises(ValidationError, match="does not contain a model"):
            pipeline.predict(df_clean)

    def test_fit_model_pipeline_without_label_col_raises_error(self, df_clean):
        """Test that fitting model pipeline without label_col raises error."""
        pipeline = Pipeline(
            [
                ("imputer", Imputer()),
                ("model", MockModel()),
            ]
        )

        with pytest.raises(ValidationError, match="label_col is required"):
            pipeline.fit(
                df_clean,
                numerical_cols=["age", "income"],
            )


class TestPipelineNotFitted:
    """Tests for error handling when pipeline is not fitted."""

    def test_transform_before_fit_raises_error(self, df_clean):
        """Test that transform before fit raises ModelNotFittedError."""
        pipeline = Pipeline([("imputer", Imputer())])

        with pytest.raises(ModelNotFittedError, match="has not been fitted"):
            pipeline.transform(df_clean)

    def test_predict_before_fit_raises_error(self, df_clean):
        """Test that predict before fit raises ModelNotFittedError."""
        pipeline = Pipeline(
            [
                ("imputer", Imputer()),
                ("model", MockModel()),
            ]
        )

        with pytest.raises(ModelNotFittedError, match="has not been fitted"):
            pipeline.predict(df_clean)


class TestPipelineSaveLoad:
    """Tests for pipeline save/load functionality."""

    def test_save_load_preprocessing_pipeline(self, df_clean, temp_dir):
        """Test save and load roundtrip for preprocessing pipeline."""
        pipeline = Pipeline(
            [("imputer", Imputer(numerical_strategy="mean", categorical_strategy="most_frequent"))]
        )
        pipeline.fit(
            df_clean,
            numerical_cols=["age", "income"],
            categorical_cols=["category"],
        )

        # Transform before save
        result_before = pipeline.transform(df_clean)

        # Save and load
        save_path = os.path.join(temp_dir, "pipeline")
        pipeline.save(save_path)
        loaded_pipeline = Pipeline.load(save_path)

        # Transform after load
        result_after = loaded_pipeline.transform(df_clean)

        # Results should be the same
        assert result_before.collect() == result_after.collect()

    def test_save_load_multi_step_pipeline(self, df_clean, temp_dir):
        """Test save and load for multi-step pipeline."""
        pipeline = Pipeline(
            [
                ("imputer", Imputer(numerical_strategy="mean")),
                ("scaler", Scaler()),
            ]
        )
        pipeline.fit(
            df_clean,
            numerical_cols=["age", "income"],
        )

        # Transform before save
        result_before = pipeline.transform(df_clean)

        # Save and load
        save_path = os.path.join(temp_dir, "pipeline")
        pipeline.save(save_path)
        loaded_pipeline = Pipeline.load(save_path)

        # Transform after load
        result_after = loaded_pipeline.transform(df_clean)

        # Results should be the same
        before_data = sorted([tuple(row) for row in result_before.collect()])
        after_data = sorted([tuple(row) for row in result_after.collect()])
        assert before_data == after_data

    def test_save_before_fit_raises_error(self, temp_dir):
        """Test that save before fit raises ModelNotFittedError."""
        pipeline = Pipeline([("imputer", Imputer())])

        with pytest.raises(ModelNotFittedError, match="has not been fitted"):
            pipeline.save(os.path.join(temp_dir, "pipeline"))

    def test_load_preserves_metadata(self, df_clean, temp_dir):
        """Test that load preserves pipeline metadata."""
        pipeline = Pipeline(
            [("imputer", Imputer(numerical_strategy="mean", categorical_strategy="most_frequent"))]
        )
        pipeline.fit(
            df_clean,
            numerical_cols=["age", "income"],
            categorical_cols=["category"],
        )

        save_path = os.path.join(temp_dir, "pipeline")
        pipeline.save(save_path)
        loaded_pipeline = Pipeline.load(save_path)

        assert loaded_pipeline._is_fitted
        assert loaded_pipeline._numerical_cols == ["age", "income"]
        assert loaded_pipeline._categorical_cols == ["category"]


class TestPipelineProperties:
    """Tests for Pipeline properties and methods."""

    def test_steps_property(self):
        """Test steps property returns copy of steps."""
        imputer = Imputer()
        pipeline = Pipeline([("imputer", imputer)])

        steps = pipeline.steps
        assert len(steps) == 1
        assert steps[0][0] == "imputer"
        assert steps[0][1] is imputer

    def test_named_steps_property(self):
        """Test named_steps property."""
        imputer = Imputer()
        scaler = Scaler()
        pipeline = Pipeline(
            [
                ("imputer", imputer),
                ("scaler", scaler),
            ]
        )

        named = pipeline.named_steps
        assert named["imputer"] is imputer
        assert named["scaler"] is scaler

    def test_getitem_access(self):
        """Test accessing steps by name with []."""
        imputer = Imputer()
        pipeline = Pipeline([("imputer", imputer)])

        assert pipeline["imputer"] is imputer

    def test_getitem_missing_key_raises_error(self):
        """Test that accessing missing step raises KeyError."""
        pipeline = Pipeline([("imputer", Imputer())])

        with pytest.raises(KeyError, match="nonexistent"):
            _ = pipeline["nonexistent"]

    def test_len(self):
        """Test len() on pipeline."""
        pipeline = Pipeline(
            [
                ("imputer", Imputer()),
                ("scaler", Scaler()),
            ]
        )
        assert len(pipeline) == 2

    def test_repr_not_fitted(self):
        """Test repr for unfitted pipeline."""
        pipeline = Pipeline([("imputer", Imputer())])
        repr_str = repr(pipeline)

        assert "Pipeline" in repr_str
        assert "imputer" in repr_str
        assert "Imputer" in repr_str
        assert "not fitted" in repr_str

    def test_repr_fitted(self, df_clean):
        """Test repr for fitted pipeline."""
        pipeline = Pipeline([("imputer", Imputer())])
        pipeline.fit(df_clean)
        repr_str = repr(pipeline)

        assert "Pipeline" in repr_str
        assert "fitted" in repr_str
        assert "not fitted" not in repr_str


class TestPipelinePreprocessingValidation:
    """Tests for pipeline preprocessing requirement validation."""

    def test_random_forest_without_encoder_raises_error_with_categorical_cols(self, df_clean):
        """Test that RandomForestRegressor without Encoder raises PreprocessingError when categorical_cols provided."""
        pipeline = Pipeline(
            [
                ("imputer", Imputer()),
                ("model", RandomForestRegressor()),
            ]
        )
        # Error should be raised at fit() time when categorical_cols is provided
        with pytest.raises(PreprocessingError) as exc_info:
            pipeline.fit(
                df_clean,
                label_col="target",
                numerical_cols=["age", "income"],
                categorical_cols=["category"],
            )
        assert "RandomForestRegressor" in str(exc_info.value)
        assert "Encoder" in str(exc_info.value)

    def test_random_forest_classifier_without_encoder_raises_error_with_categorical_cols(
        self, df_clean
    ):
        """Test that RandomForestClassifier without Encoder raises PreprocessingError when categorical_cols provided."""
        pipeline = Pipeline(
            [
                ("imputer", Imputer()),
                ("model", RandomForestClassifier()),
            ]
        )
        # Error should be raised at fit() time when categorical_cols is provided
        with pytest.raises(PreprocessingError) as exc_info:
            pipeline.fit(
                df_clean,
                label_col="target",
                numerical_cols=["age", "income"],
                categorical_cols=["category"],
            )
        assert "RandomForestClassifier" in str(exc_info.value)
        assert "Encoder" in str(exc_info.value)

    def test_xgboost_without_encoder_raises_error_with_categorical_cols(self, df_clean):
        """Test that XGBoostRegressor without Encoder raises PreprocessingError when categorical_cols provided."""
        pipeline = Pipeline(
            [
                ("imputer", Imputer()),
                ("scaler", Scaler()),
                ("model", XGBoostRegressor()),
            ]
        )
        # Error should be raised at fit() time when categorical_cols is provided
        with pytest.raises(PreprocessingError) as exc_info:
            pipeline.fit(
                df_clean,
                label_col="target",
                numerical_cols=["age", "income"],
                categorical_cols=["category"],
            )
        assert "XGBoostRegressor" in str(exc_info.value)
        assert "Encoder" in str(exc_info.value)

    def test_random_forest_with_encoder_succeeds(self):
        """Test that RandomForestRegressor with Encoder succeeds."""
        pipeline = Pipeline(
            [
                ("imputer", Imputer()),
                ("encoder", Encoder()),
                ("model", RandomForestRegressor()),
            ]
        )
        assert len(pipeline) == 3

    def test_random_forest_with_encoder_and_scaler_succeeds(self):
        """Test that RandomForestRegressor with Encoder and Scaler succeeds."""
        pipeline = Pipeline(
            [
                ("imputer", Imputer()),
                ("scaler", Scaler()),
                ("encoder", Encoder()),
                ("model", RandomForestRegressor()),
            ]
        )
        assert len(pipeline) == 4

    def test_catboost_without_encoder_succeeds(self):
        """Test that CatBoostRegressor works without Encoder (handles categoricals natively)."""
        pipeline = Pipeline(
            [
                ("imputer", Imputer()),
                ("scaler", Scaler()),
                ("model", CatBoostRegressor()),
            ]
        )
        assert len(pipeline) == 3

    def test_unknown_model_type_succeeds(self):
        """Test that unknown model types have no preprocessing requirements."""
        # MockModel is not in MODEL_PREPROCESSING_REQUIREMENTS, so it should succeed
        pipeline = Pipeline(
            [
                ("imputer", Imputer()),
                ("model", MockModel()),
            ]
        )
        assert len(pipeline) == 2

    def test_preprocessing_only_pipeline_succeeds(self):
        """Test that preprocessing-only pipelines have no validation issues."""
        pipeline = Pipeline(
            [
                ("imputer", Imputer()),
                ("scaler", Scaler()),
            ]
        )
        assert len(pipeline) == 2

    def test_preprocessing_error_contains_algorithm_and_step(self, df_clean):
        """Test that PreprocessingError contains algorithm and missing step info."""
        pipeline = Pipeline(
            [
                ("model", RandomForestRegressor()),
            ]
        )
        with pytest.raises(PreprocessingError) as exc_info:
            pipeline.fit(
                df_clean,
                label_col="target",
                numerical_cols=["age", "income"],
                categorical_cols=["category"],
            )
        error = exc_info.value
        assert error.algorithm == "RandomForestRegressor"
        assert error.missing_step == "Encoder"

    def test_random_forest_without_encoder_succeeds_with_numerical_only(self, df_clean):
        """Test that RandomForestRegressor without Encoder succeeds when categorical_cols is None."""
        pipeline = Pipeline(
            [
                ("imputer", Imputer(numerical_strategy="mean")),
                ("model", RandomForestRegressor()),
            ]
        )
        # Should not raise error when categorical_cols is None (numerical data only)
        pipeline.fit(
            df_clean,
            label_col="target",
            numerical_cols=["age", "income"],
            categorical_cols=None,
        )
        assert pipeline._is_fitted

    def test_xgboost_without_encoder_succeeds_with_numerical_only(self, df_clean):
        """Test that XGBoostRegressor without Encoder succeeds when categorical_cols is None."""
        pipeline = Pipeline(
            [
                ("imputer", Imputer(numerical_strategy="mean")),
                ("scaler", Scaler()),
                ("model", XGBoostRegressor()),
            ]
        )
        # Should not raise error when categorical_cols is None (numerical data only)
        pipeline.fit(
            df_clean,
            label_col="target",
            numerical_cols=["age", "income"],
            categorical_cols=None,
        )
        assert pipeline._is_fitted

    def test_random_forest_without_encoder_succeeds_with_empty_categorical_cols(self, df_clean):
        """Test that RandomForestRegressor without Encoder succeeds when categorical_cols is empty list."""
        pipeline = Pipeline(
            [
                ("imputer", Imputer(numerical_strategy="mean")),
                ("model", RandomForestRegressor()),
            ]
        )
        # Should not raise error when categorical_cols is empty list
        pipeline.fit(
            df_clean,
            label_col="target",
            numerical_cols=["age", "income"],
            categorical_cols=[],
        )
        assert pipeline._is_fitted
