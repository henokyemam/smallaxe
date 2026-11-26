"""Tests for Model.evaluate: the metrics validation computes, on any DataFrame."""

import math

import pytest
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F

from smallaxe.exceptions import ColumnNotFoundError, ModelNotFittedError
from smallaxe.metrics.classification import (
    accuracy,
    auc_pr,
    auc_roc,
    f1_score,
    log_loss,
    multiclass_scores,
    precision,
    recall,
)
from smallaxe.metrics.regression import mae, mape, mse, r2, rmse
from smallaxe.training import classifiers, regressors

FEATURES = ["x1", "x2"]


@pytest.fixture(scope="module")
def frame(spark_session):
    rows = [
        (i, float(i % 11), float((i * 7) % 13), 3.0 + 2.0 * (i % 11) + (i % 3), i % 2, i % 3)
        for i in range(1, 81)
    ]
    return spark_session.createDataFrame(rows, ["id", "x1", "x2", "target", "binary", "multiclass"])


def _close(got, want):
    if want is None or (isinstance(want, float) and math.isnan(want)):
        return got is None
    return math.isclose(got, want, rel_tol=1e-9, abs_tol=1e-12)


def _assert_scores(got, want):
    assert set(got) == set(want)
    for key, value in want.items():
        assert _close(got[key], value), (key, got[key], value)


def _validation_keys(model, frame, label):
    """The metric keys a train_test validation stores for this model's task."""
    fitted = model.clone().fit(
        frame, label_col=label, feature_cols=FEATURES, validation="train_test"
    )
    return set(fitted.validation_scores) - {"validation_type", "test_size"}


class TestModelEvaluate:
    def test_regression_matches_the_metric_functions(self, frame):
        model = regressors.random_forest(n_estimators=5, seed=1)
        model.fit(frame, label_col="target", feature_cols=FEATURES)
        preds = model.predict(frame)

        scores = model.evaluate(frame)

        want = {
            name: fn(preds, "target", "predict_label")
            for name, fn in [("mse", mse), ("rmse", rmse), ("mae", mae), ("r2", r2), ("mape", mape)]
        }
        _assert_scores(scores, want)
        assert set(scores) == _validation_keys(model, frame, "target")

    def test_binary_matches_the_metric_functions(self, frame):
        model = classifiers.random_forest(n_estimators=5, seed=1)
        model.fit(frame, label_col="binary", feature_cols=FEATURES)
        preds = model.predict(frame).withColumn("p1", vector_to_array("probability")[1])

        scores = model.evaluate(frame)

        want = {
            "accuracy": accuracy(preds, "binary", "predict_label"),
            "precision": precision(preds, "binary", "predict_label"),
            "recall": recall(preds, "binary", "predict_label"),
            "f1_score": f1_score(preds, "binary", "predict_label"),
            "auc_roc": auc_roc(preds, "binary", "p1"),
            "auc_pr": auc_pr(preds, "binary", "p1"),
            "log_loss": log_loss(preds, "binary", "p1"),
        }
        _assert_scores(scores, want)
        assert set(scores) == _validation_keys(model, frame, "binary")

    def test_multiclass_matches_the_metric_functions(self, frame):
        model = classifiers.random_forest(task="multiclass", n_estimators=5, seed=1)
        model.fit(frame, label_col="multiclass", feature_cols=FEATURES)
        preds = model.predict(frame)

        scores = model.evaluate(frame)

        _assert_scores(scores, multiclass_scores(preds, "multiclass", "predict_label"))
        assert set(scores) == _validation_keys(model, frame, "multiclass")

    def test_undefined_metrics_are_none(self, frame):
        model = classifiers.random_forest(n_estimators=5, seed=1)
        model.fit(frame, label_col="binary", feature_cols=FEATURES)

        # One class only: AUC-ROC is undefined.
        scores = model.evaluate(frame.filter(F.col("binary") == 1))

        assert scores["auc_roc"] is None
        assert isinstance(scores["accuracy"], float)

    def test_label_col_defaults_to_the_fitted_label_and_can_be_overridden(self, frame):
        model = regressors.random_forest(n_estimators=5, seed=1)
        model.fit(frame, label_col="target", feature_cols=FEATURES)
        renamed = frame.withColumnRenamed("target", "y")

        _assert_scores(model.evaluate(renamed, label_col="y"), model.evaluate(frame))

    def test_extra_columns_are_ignored(self, frame):
        model = regressors.random_forest(n_estimators=5, seed=1)
        model.fit(frame, label_col="target", feature_cols=FEATURES)
        extra = frame.withColumn("note", F.lit("n"))

        _assert_scores(model.evaluate(extra), model.evaluate(frame))

    @pytest.mark.parametrize("task", ["regression", "binary"])
    def test_columns_named_like_model_outputs_are_ignored(self, frame, task):
        """A frame that already holds prediction-like columns can still be evaluated."""
        if task == "regression":
            model, label = regressors.random_forest(n_estimators=5, seed=1), "target"
        else:
            model, label = classifiers.random_forest(n_estimators=5, seed=1), "binary"
        model.fit(frame, label_col=label, feature_cols=FEATURES)
        clashing = frame
        for name in ("prediction", "probability", "rawPrediction", "predict_label"):
            clashing = clashing.withColumn(name, F.lit(0.0))

        _assert_scores(model.evaluate(clashing), model.evaluate(frame))

    def test_before_fit_raises(self, frame):
        with pytest.raises(ModelNotFittedError):
            regressors.random_forest().evaluate(frame)

    def test_missing_label_raises(self, frame):
        model = regressors.random_forest(n_estimators=5, seed=1)
        model.fit(frame, label_col="target", feature_cols=FEATURES)

        with pytest.raises(ColumnNotFoundError, match="target"):
            model.evaluate(frame.drop("target"))

    def test_missing_feature_raises(self, frame):
        model = regressors.random_forest(n_estimators=5, seed=1)
        model.fit(frame, label_col="target", feature_cols=FEATURES)

        with pytest.raises(ColumnNotFoundError, match="x2"):
            model.evaluate(frame.drop("x2"))

    def test_evaluate_after_load_is_identical(self, frame, tmp_path):
        from smallaxe import load_model

        model = classifiers.random_forest(n_estimators=5, seed=1)
        model.fit(frame, label_col="binary", feature_cols=FEATURES)
        model.save(str(tmp_path / "m"))

        _assert_scores(load_model(str(tmp_path / "m")).evaluate(frame), model.evaluate(frame))
