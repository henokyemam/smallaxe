"""User columns survive fit and predict untouched.

See ``GOTCHAS.md`` item 2. A user column named ``features`` used to replace the
model's own feature vector, and ``predict`` dropped it from the output.
"""

import pytest
from pyspark.ml.linalg import Vectors

from smallaxe.exceptions import ValidationError
from smallaxe.training import Classifiers, Regressors


@pytest.fixture
def frame_with_features_column(spark_session):
    rows = [
        (float(i), float(i * 2), Vectors.dense([9.0, 9.0, 9.0]), float(i * 3), i % 2)
        for i in range(40)
    ]
    return spark_session.createDataFrame(rows, ["x", "z", "features", "y", "label"])


def test_model_trains_on_feature_cols_not_the_features_column(frame_with_features_column):
    model = Regressors.random_forest(n_estimators=2, seed=1)
    model.fit(frame_with_features_column, label_col="y", feature_cols=["x", "z"])

    assert model.metadata["feature_cols"] == ["x", "z"]
    assert model._spark_model.numFeatures == 2


def test_predict_keeps_the_users_features_column(frame_with_features_column):
    model = Regressors.random_forest(n_estimators=2, seed=1)
    model.fit(frame_with_features_column, label_col="y", feature_cols=["x", "z"])

    out = model.predict(frame_with_features_column)

    assert set(out.columns) == set(frame_with_features_column.columns) | {"predict_label"}
    assert out.select("features").first()[0] == Vectors.dense([9.0, 9.0, 9.0])


def test_predict_keeps_user_prediction_and_probability_columns(spark_session):
    rows = [(float(i), float(i % 3), "mine", 0.5, i % 2) for i in range(40)]
    df = spark_session.createDataFrame(rows, ["x", "z", "prediction", "probability", "label"])
    model = Classifiers.random_forest(n_estimators=2, seed=1)
    model.fit(df, label_col="label", feature_cols=["x", "z"])

    out = model.predict(df)

    assert set(out.columns) == set(df.columns) | {"predict_label"}
    assert out.select("prediction").first()[0] == "mine"
    assert out.select("probability").first()[0] == 0.5


def test_output_col_may_not_overwrite_a_user_column(frame_with_features_column):
    model = Classifiers.random_forest(n_estimators=2, seed=1)
    model.fit(frame_with_features_column, label_col="label", feature_cols=["x", "z"])

    with pytest.raises(ValidationError, match="already exists"):
        model.predict(frame_with_features_column, output_col="y")
    with pytest.raises(ValidationError, match="already exists"):
        model.predict_proba(frame_with_features_column, output_col="x")


def test_predict_proba_output_is_the_only_new_column(frame_with_features_column):
    model = Classifiers.random_forest(n_estimators=2, seed=1)
    model.fit(frame_with_features_column, label_col="label", feature_cols=["x", "z"])

    out = model.predict_proba(frame_with_features_column, output_col="p")

    assert set(out.columns) == set(frame_with_features_column.columns) | {"p"}
