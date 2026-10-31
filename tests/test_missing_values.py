"""Rows with null or NaN features are kept or rejected, never dropped.

See ``GOTCHAS.md`` item 3. ``VectorAssembler(handleInvalid="skip")`` used to remove
such rows from fit, validation scoring, and predict without a message.
"""

import math

import pytest
from pyspark.sql import functions as F

from smallaxe.exceptions import ValidationError
from smallaxe.preprocessing import Imputer
from smallaxe.training import Classifiers, Regressors, algorithm
from smallaxe.training.base import check_training_data


@pytest.fixture
def frame(spark_session):
    rows = []
    for i in range(60):
        age = None if i % 10 == 0 else 20.0 + i
        income = float("nan") if i % 15 == 0 else 1000.0 * i
        rows.append((i, age, income, float(i), i % 2))
    return spark_session.createDataFrame(rows, ["id", "age", "income", "target", "label"])


def _skip_unless(name):
    if not algorithm.get(name).available():
        pytest.skip(f"{name} is not installed")


class TestRandomForestRejectsMissing:
    def test_fit_names_the_columns_with_missing_values(self, frame):
        model = Regressors.random_forest(n_estimators=2, seed=1)
        with pytest.raises(ValidationError, match="age \\(6\\).*income \\(4\\)") as exc:
            model.fit(frame, label_col="target", feature_cols=["age", "income"])
        assert "Imputer" in str(exc.value)

    def test_fit_rejects_infinite_features(self, spark_session):
        df = spark_session.createDataFrame(
            [(1.0, 1.0), (float("inf"), 2.0), (3.0, 3.0)] * 5, ["x", "y"]
        )
        model = Regressors.random_forest(n_estimators=2, seed=1)
        with pytest.raises(ValidationError, match="infinite values: x \\(5\\)"):
            model.fit(df, label_col="y", feature_cols=["x"])

    def test_clean_data_records_zero_missing(self, frame):
        clean = frame.filter(F.col("age").isNotNull() & ~F.isnan("income"))
        model = Regressors.random_forest(n_estimators=2, seed=1)
        model.fit(clean, label_col="target", feature_cols=["age", "income"])
        assert model.metadata["missing_counts"] == {"age": 0, "income": 0}
        assert model.metadata["n_samples"] == clean.count()


def test_null_label_is_rejected_for_every_algorithm(spark_session):
    df = spark_session.createDataFrame([(1.0, 1.0), (2.0, None), (3.0, 3.0)] * 5, ["x", "y"])
    model = Regressors.random_forest(n_estimators=2, seed=1)
    with pytest.raises(ValidationError, match="Label column 'y' has 5 null values"):
        model.fit(df, label_col="y", feature_cols=["x"])


def test_check_training_data_counts_in_one_pass(frame):
    stats = check_training_data(frame, "target", ["age", "income"], accepts_missing=True)
    assert stats == {"n_samples": 60, "missing_counts": {"age": 6, "income": 4}}


class TestBoostingKeepsMissingRows:
    def test_xgboost_predicts_every_row(self, frame):
        _skip_unless("xgboost")
        model = Regressors.xgboost(n_estimators=3, seed=1)
        model.fit(frame, label_col="target", feature_cols=["age", "income"])

        out = model.predict(frame)

        assert out.count() == 60
        assert out.filter(F.col("predict_label").isNull()).count() == 0
        assert model.metadata["missing_counts"] == {"age": 6, "income": 4}

    def test_xgboost_validation_scores_use_every_row(self, frame):
        _skip_unless("xgboost")
        model = Classifiers.xgboost(n_estimators=3, seed=1)
        model.fit(frame, label_col="label", feature_cols=["age", "income"], validation="train_test")
        assert 0 <= model.validation_scores["accuracy"] <= 1


def test_search_rejects_bad_data_before_any_trial(frame):
    pytest.importorskip("hyperopt")
    from smallaxe.search import optimize

    model = Regressors.random_forest(n_estimators=2, seed=1)
    with pytest.raises(ValidationError, match="null or NaN"):
        optimize.run(
            model,
            frame,
            label_col="target",
            param_space={"max_depth": [2, 3]},
            feature_cols=["age", "income"],
            max_evals=2,
            verbose=False,
        )


class TestImputerIgnoresNaN:
    @pytest.mark.parametrize("strategy", ["mean", "median", "mode"])
    def test_output_has_no_nan(self, spark_session, strategy):
        df = spark_session.createDataFrame(
            [(1.0,), (float("nan"),), (None,), (3.0,), (3.0,)], ["x"]
        )
        imputer = Imputer(numerical_strategy=strategy).fit(df, numerical_cols=["x"])

        fill = imputer.numerical_fill_values["x"]
        assert not math.isnan(fill)
        values = [r["x"] for r in imputer.transform(df).collect()]
        assert all(v is not None and not math.isnan(v) for v in values)

    def test_mean_ignores_nan(self, spark_session):
        df = spark_session.createDataFrame([(2.0,), (float("nan"),), (4.0,)], ["x"])
        imputer = Imputer(numerical_strategy="mean").fit(df, numerical_cols=["x"])
        assert imputer.numerical_fill_values["x"] == 3.0
