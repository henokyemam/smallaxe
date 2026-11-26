"""Tests for the Pipeline class."""

import math
import os
import shutil
import tempfile
import warnings

import pytest
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from smallaxe.exceptions import ModelNotFittedError, PreprocessingError, ValidationError
from smallaxe.pipeline import Pipeline
from smallaxe.preprocessing import Encoder, Imputer, Scaler
from smallaxe.training import algorithm, classifiers, regressors
from tests.fakes import install_fake_modules


def _require(name):
    """Skip unless ``name``'s package is installed (probed lazily, not at collection)."""
    if not algorithm.get(name).available():
        pytest.skip(f"{name} is not installed")


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
            raise ModelNotFittedError("MockModel has not been fitted.")
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
        """Accessing a missing step raises ValidationError naming the steps."""
        pipeline = Pipeline([("imputer", Imputer())])

        with pytest.raises(ValidationError, match="nonexistent"):
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


@pytest.fixture
def df_mixed(spark_session):
    """Numeric features, a categorical, an unlisted string ID and an unlisted number."""
    rows = [
        (
            f"cust-{i:03d}",
            float(20 + i % 30) if i % 9 else None,
            float(40000 + i * 900),
            ["a", "b", "c"][i % 3] if i % 11 else None,
            float(i % 4),
            50.0 + i * 3 + (i % 5),
            i % 2,
        )
        for i in range(1, 61)
    ]
    return spark_session.createDataFrame(
        rows, ["customer_id", "age", "income", "category", "unlisted", "target", "label"]
    )


def _full_pipeline(model, encoder_method="onehot"):
    return Pipeline(
        [
            ("imputer", Imputer(numerical_strategy="mean", categorical_strategy="most_frequent")),
            ("scaler", Scaler()),
            ("encoder", Encoder(method=encoder_method)),
            ("model", model),
        ]
    )


def _assert_same_scores(got, want):
    """Equal metrics. Spark sums in partition order, so the last bits of a mean may vary."""
    assert set(got) == set(want)
    for key, value in want.items():
        if value is None:
            assert got[key] is None, key
        else:
            assert math.isclose(got[key], value, rel_tol=1e-12, abs_tol=1e-12), key


class DuckStep:
    """A preprocessing step smallaxe does not know: fit + transform, no predict."""

    def __init__(self):
        self.fit_kwargs = None

    def fit(self, df, numerical_cols=None, categorical_cols=None):
        self.fit_kwargs = {"numerical_cols": numerical_cols, "categorical_cols": categorical_cols}
        return self

    def transform(self, df):
        return df.withColumn("duck", F.lit(1.0))


class TestEncoderRequirement:
    """Whether a model needs an Encoder comes from its Algorithm record."""

    @pytest.mark.parametrize(
        "make", [regressors.random_forest, classifiers.random_forest], ids=["reg", "clf"]
    )
    def test_random_forest_with_categorical_cols_and_no_encoder_raises(self, df_clean, make):
        pipeline = Pipeline([("imputer", Imputer()), ("model", make())])

        with pytest.raises(PreprocessingError) as exc_info:
            pipeline.fit(
                df_clean,
                label_col="target",
                numerical_cols=["age", "income"],
                categorical_cols=["category"],
            )
        assert exc_info.value.algorithm == "random_forest"
        assert exc_info.value.missing_step == "Encoder"
        assert "Encoder" in str(exc_info.value)

    def test_xgboost_with_categorical_cols_and_no_encoder_raises(self, df_clean):
        _require("xgboost")
        pipeline = Pipeline([("scaler", Scaler()), ("model", regressors.xgboost())])

        with pytest.raises(PreprocessingError, match="xgboost requires Encoder"):
            pipeline.fit(
                df_clean,
                label_col="target",
                numerical_cols=["age", "income"],
                categorical_cols=["category"],
            )

    @pytest.mark.parametrize("name", ["lightgbm", "catboost"])
    def test_fake_algorithm_with_categorical_cols_and_no_encoder_raises(
        self, df_clean, monkeypatch, name
    ):
        record = algorithm.get(name)
        install_fake_modules(monkeypatch, record)
        assert record.accepts_raw_categoricals is False
        pipeline = Pipeline([("imputer", Imputer()), ("model", getattr(regressors, name)())])

        with pytest.raises(PreprocessingError) as exc_info:
            pipeline.fit(
                df_clean,
                label_col="target",
                numerical_cols=["age", "income"],
                categorical_cols=["category"],
            )
        assert exc_info.value.algorithm == name

    @pytest.mark.parametrize("name", ["lightgbm", "catboost"])
    def test_fake_algorithm_with_an_encoder_fits_on_the_encoded_columns(
        self, df_clean, monkeypatch, name
    ):
        install_fake_modules(monkeypatch, algorithm.get(name))
        pipeline = Pipeline([("encoder", Encoder()), ("model", getattr(regressors, name)())])

        pipeline.fit(
            df_clean,
            label_col="target",
            numerical_cols=["age", "income"],
            categorical_cols=["category"],
        )

        assert pipeline.feature_cols == ["age", "income"] + pipeline["encoder"].output_cols
        assert pipeline["model"]._feature_cols == pipeline.feature_cols

    def test_no_encoder_needed_without_categorical_cols(self, df_clean):
        for categorical_cols in (None, []):
            pipeline = Pipeline(
                [
                    ("imputer", Imputer(numerical_strategy="mean")),
                    ("model", regressors.random_forest()),
                ]
            )
            pipeline.fit(
                df_clean,
                label_col="target",
                numerical_cols=["age", "income"],
                categorical_cols=categorical_cols,
            )
            assert pipeline.feature_cols == ["age", "income"]

    def test_duck_typed_model_without_encoder_gets_the_raw_columns(self, df_clean):
        pipeline = Pipeline([("model", MockModel())])

        pipeline.fit(
            df_clean,
            label_col="target",
            numerical_cols=["age", "income"],
            categorical_cols=["category"],
        )

        assert pipeline["model"]._feature_cols == ["age", "income", "category"]
        assert pipeline.feature_cols == ["age", "income", "category"]


class TestFeatureColumns:
    """The Pipeline, not the frame, decides which columns reach the model."""

    @pytest.mark.parametrize("method", ["onehot", "label"])
    def test_model_gets_numerical_cols_and_encoder_outputs_only(self, df_mixed, method):
        pipeline = _full_pipeline(regressors.random_forest(n_estimators=5, seed=1), method)

        pipeline.fit(
            df_mixed,
            label_col="target",
            numerical_cols=["age", "income"],
            categorical_cols=["category"],
        )

        expected = ["age", "income"] + pipeline["encoder"].output_cols
        assert pipeline.feature_cols == expected
        assert pipeline["model"]._feature_cols == expected
        for unlisted in ("customer_id", "unlisted", "label", "target"):
            assert unlisted not in pipeline.feature_cols
        if method == "label":
            assert pipeline.feature_cols == ["age", "income", "category"]

    def test_predict_ignores_extra_columns_and_unseen_categories(self, df_mixed):
        pipeline = _full_pipeline(classifiers.random_forest(n_estimators=5, seed=1))
        pipeline.fit(
            df_mixed,
            label_col="label",
            numerical_cols=["age", "income"],
            categorical_cols=["category"],
        )
        shifted = df_mixed.withColumn(
            "category", F.when(F.col("age") > 40, F.lit("never-seen")).otherwise(F.col("category"))
        ).withColumn("extra", F.lit("x"))

        result = pipeline.predict(shifted)

        assert result.count() == shifted.count()
        assert result.filter(F.col("predict_label").isNull()).count() == 0
        assert "extra" in result.columns and "customer_id" in result.columns

    def test_without_column_lists_the_model_infers_its_features_and_warns_once(self, df_mixed):
        numeric = df_mixed.drop("category").fillna({"age": 30.0})
        pipeline = Pipeline([("model", regressors.random_forest(n_estimators=5, seed=1))])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pipeline.fit(numeric, label_col="target")

        inference = [w for w in caught if "Feature inference" in str(w.message)]
        assert len(inference) == 1
        assert "customer_id" in str(inference[0].message)
        assert pipeline.feature_cols == ["age", "income", "unlisted", "label"]
        assert pipeline["model"]._feature_cols == pipeline.feature_cols
        assert pipeline["model"].metadata["feature_cols_inferred"] is True

    def test_duck_typed_model_without_column_lists_gets_the_inferred_features(self, df_mixed):
        pipeline = Pipeline([("model", MockModel())])

        with pytest.warns(UserWarning, match="Feature inference"):
            pipeline.fit(df_mixed, label_col="target")

        assert pipeline["model"]._feature_cols == ["age", "income", "unlisted", "label"]

    def test_label_listed_as_a_feature_raises(self, df_clean):
        pipeline = Pipeline([("model", regressors.random_forest())])

        with pytest.raises(ValidationError, match="target"):
            pipeline.fit(df_clean, label_col="target", numerical_cols=["age", "target"])

    def test_preprocessing_only_pipeline_has_no_feature_cols(self, df_clean):
        pipeline = Pipeline([("imputer", Imputer(numerical_strategy="mean"))])
        pipeline.fit(df_clean, numerical_cols=["age"])

        assert pipeline.feature_cols is None


class TestSkippedSteps:
    """A step the pipeline has no columns for is skipped, recorded, and passed through."""

    def test_steps_without_columns_are_skipped_and_pass_data_through(self, df_clean):
        pipeline = Pipeline(
            [
                ("imputer", Imputer(numerical_strategy="mean")),
                ("scaler", Scaler()),
                ("encoder", Encoder()),
            ]
        )

        pipeline.fit(df_clean, categorical_cols=None, numerical_cols=None)

        assert pipeline.skipped_steps == ["imputer", "scaler", "encoder"]
        assert pipeline.transform(df_clean).collect() == df_clean.collect()

    def test_only_the_steps_without_columns_are_skipped(self, df_clean):
        pipeline = Pipeline(
            [
                ("imputer", Imputer(numerical_strategy="mean")),
                ("scaler", Scaler()),
                ("encoder", Encoder()),
                ("model", regressors.random_forest(n_estimators=5, seed=1)),
            ]
        )

        pipeline.fit(df_clean, label_col="target", numerical_cols=["age", "income"])

        assert pipeline.skipped_steps == ["encoder"]
        assert pipeline.feature_cols == ["age", "income"]
        assert "category" in pipeline.transform(df_clean).columns

    def test_refit_with_columns_unskips(self, df_clean):
        pipeline = Pipeline([("scaler", Scaler())])
        pipeline.fit(df_clean)
        assert pipeline.skipped_steps == ["scaler"]

        pipeline.fit(df_clean, numerical_cols=["age"])

        assert pipeline.skipped_steps == []


class TestDuckTypedSteps:
    def test_duck_step_is_fitted_with_the_column_lists_and_applied(self, df_clean):
        duck = DuckStep()
        pipeline = Pipeline([("duck", duck), ("model", MockModel())])

        pipeline.fit(df_clean, label_col="target", numerical_cols=["age"], categorical_cols=None)

        assert duck.fit_kwargs == {"numerical_cols": ["age"], "categorical_cols": None}
        assert "duck" in pipeline.transform(df_clean).columns
        assert pipeline.skipped_steps == []

    def test_duck_step_whose_fit_raises_type_error_propagates(self, df_clean):
        class NoColumnsStep(DuckStep):
            def fit(self, df):  # takes no column lists
                return self

        pipeline = Pipeline([("odd", NoColumnsStep())])

        with pytest.raises(TypeError):
            pipeline.fit(df_clean, numerical_cols=["age"])
        assert not pipeline._is_fitted

    def test_step_that_is_neither_preprocessing_nor_model_is_rejected(self):
        with pytest.raises(ValidationError, match="'thing'"):
            Pipeline([("thing", object())])

    def test_more_than_one_model_is_rejected(self):
        with pytest.raises(ValidationError, match="one model"):
            Pipeline([("a", MockModel()), ("b", MockModel())])


class TestPredictionApi:
    @pytest.fixture
    def fitted_classifier(self, df_mixed):
        pipeline = _full_pipeline(classifiers.random_forest(n_estimators=5, seed=1))
        return pipeline.fit(
            df_mixed,
            label_col="label",
            numerical_cols=["age", "income"],
            categorical_cols=["category"],
        )

    @pytest.fixture
    def fitted_regressor(self, df_mixed):
        pipeline = _full_pipeline(regressors.random_forest(n_estimators=5, seed=1))
        return pipeline.fit(
            df_mixed,
            label_col="target",
            numerical_cols=["age", "income"],
            categorical_cols=["category"],
        )

    def test_predict_output_col_defaults_like_model_predict_and_passes_through(
        self, fitted_regressor, df_mixed
    ):
        assert "predict_label" in fitted_regressor.predict(df_mixed).columns
        result = fitted_regressor.predict(df_mixed, output_col="yhat")
        assert "yhat" in result.columns and "predict_label" not in result.columns

    def test_predict_proba_delegates_to_the_model(self, fitted_classifier, df_mixed):
        model = fitted_classifier["model"]
        transformed = fitted_classifier.transform(df_mixed)

        got = fitted_classifier.predict_proba(df_mixed).select("probability").collect()
        want = model.predict_proba(transformed).select("probability").collect()

        assert got == want
        assert "p" in fitted_classifier.predict_proba(df_mixed, output_col="p").columns

    def test_evaluate_delegates_to_the_model(self, fitted_classifier, df_mixed):
        model = fitted_classifier["model"]

        scores = fitted_classifier.evaluate(df_mixed)

        _assert_same_scores(scores, model.evaluate(fitted_classifier.transform(df_mixed)))
        assert set(scores) >= {"accuracy", "auc_roc", "log_loss"}

    def test_evaluate_with_an_explicit_label(self, fitted_regressor, df_mixed):
        renamed = df_mixed.withColumnRenamed("target", "y")

        _assert_same_scores(
            fitted_regressor.evaluate(renamed, label_col="y"), fitted_regressor.evaluate(df_mixed)
        )

    def test_predict_proba_on_a_regressor_raises(self, fitted_regressor, df_mixed):
        with pytest.raises(ValidationError, match="predict_proba"):
            fitted_regressor.predict_proba(df_mixed)

    @pytest.mark.parametrize("method", ["predict", "predict_proba", "evaluate"])
    def test_without_a_model_raises(self, df_clean, method):
        pipeline = Pipeline([("imputer", Imputer(numerical_strategy="mean"))])
        pipeline.fit(df_clean, numerical_cols=["age"])

        with pytest.raises(ValidationError, match="does not contain a model"):
            getattr(pipeline, method)(df_clean)

    @pytest.mark.parametrize("method", ["predict", "predict_proba", "evaluate"])
    def test_before_fit_raises(self, df_clean, method):
        pipeline = Pipeline([("model", classifiers.random_forest())])

        with pytest.raises(ModelNotFittedError):
            getattr(pipeline, method)(df_clean)


# --- Validation through the pipeline --------------------------------------------------


@pytest.fixture
def churn(spark_session):
    rows = [
        (
            i,
            float(18 + i % 50),
            (None if i % 11 == 0 else float(1000 + 37 * i)),
            ["a", "b", "c"][i % 3],
            int((i % 7) < 3),
        )
        for i in range(120)
    ]
    return spark_session.createDataFrame(rows, ["id", "age", "income", "city", "label"])


def _validated_pipeline(model):
    return Pipeline(
        [
            ("imputer", Imputer(numerical_strategy="mean", categorical_strategy="most_frequent")),
            ("scaler", Scaler()),
            ("encoder", Encoder()),
            ("model", model),
        ]
    )


_COLUMNS = {"numerical_cols": ["age", "income"], "categorical_cols": ["city"]}


class TestPipelineValidation:
    @pytest.mark.parametrize(
        "options",
        [
            {"validation": "train_test", "test_size": 0.3},
            {"validation": "kfold", "n_folds": 3},
            {"validation": "kfold", "n_folds": 3, "refit": False},
            {"validation": "train_test", "fold_key": "id", "cache_strategy": "none"},
        ],
        ids=["train_test", "kfold", "kfold-no-refit", "fold_key"],
    )
    def test_scores_equal_the_model_fitted_on_the_preprocessed_frame(self, churn, options):
        pipeline = _validated_pipeline(classifiers.random_forest(n_estimators=5, seed=7))
        pipeline.fit(churn, label_col="label", **_COLUMNS, **options)

        direct = classifiers.random_forest(n_estimators=5, seed=7)
        direct.fit(
            pipeline.transform(churn),
            label_col="label",
            feature_cols=pipeline.feature_cols,
            **options,
        )

        assert pipeline.validation_scores is not None
        assert pipeline.validation_scores == direct.validation_scores
        assert pipeline.validation_scores["validation_type"] == options["validation"]
        model = pipeline.named_steps["model"]
        assert model.metadata["split_seed"] == direct.metadata["split_seed"] == 7
        assert model.metadata["kept_model"] == direct.metadata["kept_model"]

    def test_global_seed_gives_the_same_split(self, churn):
        import smallaxe

        smallaxe.set_seed(11)
        try:
            pipeline = _validated_pipeline(classifiers.random_forest(n_estimators=5))
            pipeline.fit(churn, label_col="label", **_COLUMNS, validation="train_test")
            direct = classifiers.random_forest(n_estimators=5)
            direct.fit(
                pipeline.transform(churn),
                label_col="label",
                feature_cols=pipeline.feature_cols,
                validation="train_test",
            )
        finally:
            smallaxe.set_seed(None)
        assert pipeline.validation_scores == direct.validation_scores

    def test_fit_transform_passes_the_options(self, churn):
        pipeline = _validated_pipeline(classifiers.random_forest(n_estimators=5, seed=7))
        transformed = pipeline.fit_transform(
            churn, label_col="label", **_COLUMNS, validation="kfold", n_folds=2
        )

        assert pipeline.validation_scores["n_folds"] == 2
        assert "predict_label" not in transformed.columns

    def test_default_is_no_validation(self, churn):
        pipeline = _validated_pipeline(classifiers.random_forest(n_estimators=5, seed=7))
        pipeline.fit(churn, label_col="label", **_COLUMNS)

        assert pipeline.validation_scores is None

    def test_validation_options_are_keyword_only(self, churn):
        pipeline = _validated_pipeline(classifiers.random_forest(n_estimators=5, seed=7))
        with pytest.raises(TypeError):
            pipeline.fit(churn, "label", ["age"], ["city"], "train_test")

    def test_invalid_option_is_the_models_validation_error(self, churn):
        pipeline = _validated_pipeline(classifiers.random_forest(n_estimators=5, seed=7))
        with pytest.raises(ValidationError, match="Invalid validation strategy"):
            pipeline.fit(churn, label_col="label", **_COLUMNS, validation="loo")
        assert not pipeline._is_fitted

    @pytest.mark.parametrize(
        "option",
        [
            {"validation": "train_test"},
            {"validation": "kfold"},
            {"test_size": 0.3},
            {"n_folds": 3},
            {"fold_key": "id"},
            {"refit": False},
            {"cache_strategy": "memory"},
            # Equal to the default but of another type: still an option given.
            {"refit": 1},
            {"n_folds": 5.0},
            {"validation": None},
        ],
        ids=lambda option: f"{next(iter(option))}={next(iter(option.values()))!r}",
    )
    def test_options_without_a_model_raise(self, churn, option):
        pipeline = Pipeline([("scaler", Scaler())])
        with pytest.raises(ValidationError, match="need a model step"):
            pipeline.fit(churn, numerical_cols=["age"], **option)
        with pytest.raises(ValidationError, match="need a model step"):
            pipeline.fit_transform(churn, numerical_cols=["age"], **option)

    def test_default_values_passed_explicitly_without_a_model_are_accepted(self, churn):
        """A default passed explicitly cannot be told from one left out: both are allowed."""
        pipeline = Pipeline([("scaler", Scaler())])
        pipeline.fit(
            churn,
            numerical_cols=["age"],
            validation="none",
            test_size=0.2,
            n_folds=5,
            fold_key=None,
            refit=True,
            cache_strategy=None,
        )
        assert pipeline.validation_scores is None

    def test_validation_scores_without_a_model_is_none(self, churn):
        pipeline = Pipeline([("scaler", Scaler())]).fit(churn, numerical_cols=["age"])
        assert pipeline.validation_scores is None
        assert Pipeline([("scaler", Scaler())]).validation_scores is None

    def test_validation_scores_before_fit_raises(self):
        pipeline = _validated_pipeline(classifiers.random_forest())
        with pytest.raises(ModelNotFittedError):
            _ = pipeline.validation_scores

    def test_scores_survive_save_and_load(self, churn, tmp_path):
        pipeline = _validated_pipeline(classifiers.random_forest(n_estimators=5, seed=7))
        pipeline.fit(churn, label_col="label", **_COLUMNS, validation="train_test")
        pipeline.save(str(tmp_path / "p"))

        assert Pipeline.load(str(tmp_path / "p")).validation_scores == pipeline.validation_scores

    def test_duck_typed_model_gets_only_the_options_given(self, churn):
        class Recording(MockModel):
            def fit(self, df, label_col, feature_cols, **options):
                self.options = options
                self.validation_scores = {"seen": sorted(options)}
                return super().fit(df, label_col, feature_cols)

        model = Recording()
        pipeline = Pipeline([("scaler", Scaler()), ("model", model)])
        pipeline.fit(churn, label_col="label", numerical_cols=["age"])
        assert model.options == {}

        pipeline.fit(churn, label_col="label", numerical_cols=["age"], validation="kfold")
        assert model.options == {"validation": "kfold"}
        assert pipeline.validation_scores == {"seen": ["validation"]}

    def test_duck_typed_model_without_scores_reports_none(self, churn):
        pipeline = Pipeline([("scaler", Scaler()), ("model", MockModel())])
        pipeline.fit(churn, label_col="label", numerical_cols=["age"])
        assert pipeline.validation_scores is None


class TestPipelineArguments:
    @pytest.mark.parametrize("value", ["age", [1, 2], {"age": 1}])
    def test_column_lists_must_be_lists_of_names(self, churn, value):
        pipeline = Pipeline([("scaler", Scaler())])
        with pytest.raises(ValidationError, match="must be a list of column names"):
            pipeline.fit(churn, numerical_cols=value)
        with pytest.raises(ValidationError, match="must be a list of column names"):
            pipeline.fit(churn, categorical_cols=value)

    def test_dataframe_must_be_a_dataframe(self):
        with pytest.raises(ValidationError, match="must be a PySpark DataFrame"):
            Pipeline([("scaler", Scaler())]).fit({"age": [1]}, numerical_cols=["age"])
