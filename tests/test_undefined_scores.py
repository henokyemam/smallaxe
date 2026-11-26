"""Undefined (NaN) metric values in validation, k-fold means, saved scores, and search.

A metric that is undefined on a validation split (AUC-ROC on a fold with one class)
is NaN from ``smallaxe.metrics``. Validation stores it as None, k-fold means skip it
with one warning, and ``optimize.run`` counts a trial scored on it as failed.
"""

import json
import math
import warnings

import pytest
from pyspark.sql import functions as F

import smallaxe
from smallaxe.exceptions import ValidationError
from smallaxe.training import classifiers
from smallaxe.training.base import Model
from smallaxe.training.mixins.validation_mixin import _split_hash

NAN = float("nan")
SEED = 11
N_FOLDS = 3


@pytest.fixture(autouse=True)
def no_global_seed():
    smallaxe.set_seed(None)
    yield
    smallaxe.set_seed(None)


@pytest.fixture(scope="module")
def one_class_fold_frame(spark_session):
    """Binary data whose last k-fold validation fold (fold_key='key') has only label 0."""
    keys = spark_session.createDataFrame([(f"k{i}",) for i in range(60)], ["key"])
    folds = {
        row["key"]: row["fold"]
        for row in keys.select(
            "key", (_split_hash(keys, SEED, "key") % N_FOLDS).alias("fold")
        ).collect()
    }
    rows = []
    for index, (key, fold) in enumerate(sorted(folds.items())):
        label = 1 if fold != N_FOLDS - 1 and index % 2 == 0 else 0
        for copy in range(3):
            rows.append((key, float(label) + 0.1 * copy, float(index % 5), label))
    df = spark_session.createDataFrame(rows, ["key", "x", "z", "label"])

    by_fold = (
        df.withColumn("fold", _split_hash(df, SEED, "key") % N_FOLDS)
        .groupBy("fold")
        .agg(F.countDistinct("label").alias("classes"))
        .collect()
    )
    assert {row["fold"]: row["classes"] for row in by_fold} == {0: 2, 1: 2, 2: 1}
    return df


def _undefined_warnings(record):
    return [w for w in record if "undefined" in str(w.message)]


class TestKFoldMean:
    def test_one_class_fold_is_skipped_with_one_warning(self, one_class_fold_frame):
        model = classifiers.random_forest(n_estimators=3, seed=SEED)
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            model.fit(
                one_class_fold_frame,
                label_col="label",
                feature_cols=["x", "z"],
                validation="kfold",
                n_folds=N_FOLDS,
                fold_key="key",
            )

        scores = model.validation_scores
        folds = scores["fold_scores"]
        assert folds[2]["auc_roc"] is None
        assert folds[2]["auc_pr"] is None
        assert scores["mean_auc_roc"] == pytest.approx(
            (folds[0]["auc_roc"] + folds[1]["auc_roc"]) / 2
        )
        assert scores["mean_accuracy"] == pytest.approx(
            sum(fold["accuracy"] for fold in folds) / N_FOLDS
        )
        undefined = _undefined_warnings(record)
        assert len(undefined) == 1
        message = str(undefined[0].message)
        assert "auc_roc" in message and "1 of 3 folds" in message
        assert undefined[0].category is UserWarning

    def test_scores_with_undefined_folds_are_valid_json(self, one_class_fold_frame, tmp_path):
        model = classifiers.random_forest(n_estimators=3, seed=SEED)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(
                one_class_fold_frame,
                label_col="label",
                feature_cols=["x", "z"],
                validation="kfold",
                n_folds=N_FOLDS,
                fold_key="key",
            )
        json.dumps(model.validation_scores, allow_nan=False)

        model.save(str(tmp_path / "model"))
        saved = (tmp_path / "model" / "metadata.json").read_text()
        assert "NaN" not in saved
        assert (
            classifiers.load(str(tmp_path / "model")).validation_scores["fold_scores"][2]["auc_roc"]
            is None
        )

    def test_every_fold_undefined_leaves_the_mean_undefined(self):
        model = classifiers.random_forest()
        folds = [
            {"accuracy": 0.5, "auc_roc": NAN, "log_loss": None, "fold": 0},
            {"accuracy": 0.7, "auc_roc": NAN, "log_loss": None, "fold": 1},
        ]
        with pytest.warns(UserWarning, match=r"auc_roc.*2 of 2 folds") as record:
            result = model._aggregate_fold_scores(folds, 2)

        assert len(_undefined_warnings(record)) == 1
        assert result["mean_auc_roc"] is None
        assert "std_auc_roc" not in result
        assert result["mean_accuracy"] == pytest.approx(0.6)
        assert [fold["auc_roc"] for fold in result["fold_scores"]] == [None, None]
        # Unavailable metrics (None from the start) stay silent and get no mean.
        assert "mean_log_loss" not in result

    def test_all_defined_gives_no_warning(self):
        model = classifiers.random_forest()
        folds = [{"auc_roc": 0.6, "fold": 0}, {"auc_roc": 0.8, "fold": 1}]
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            result = model._aggregate_fold_scores(folds, 2)
        assert result["mean_auc_roc"] == pytest.approx(0.7)


class TestTrainTest:
    def test_undefined_score_is_stored_as_none(self, monkeypatch, one_class_fold_frame):
        monkeypatch.setattr(
            Model, "_evaluate", lambda self, df, label_col: {"accuracy": 1.0, "auc_roc": NAN}
        )
        model = classifiers.random_forest(n_estimators=2, seed=SEED)
        model.fit(
            one_class_fold_frame,
            label_col="label",
            feature_cols=["x"],
            validation="train_test",
        )
        assert model.validation_scores["auc_roc"] is None
        assert model.validation_scores["accuracy"] == 1.0


class TestBinaryProbabilityMetrics:
    def test_errors_in_probability_metrics_propagate(self, monkeypatch, one_class_fold_frame):
        import smallaxe.metrics.classification as classification

        def broken(*args, **kwargs):
            raise RuntimeError("evaluator exploded")

        monkeypatch.setattr(classification, "auc_roc", broken)
        model = classifiers.random_forest(n_estimators=2, seed=SEED)
        with pytest.raises(RuntimeError, match="evaluator exploded"):
            model.fit(
                one_class_fold_frame,
                label_col="label",
                feature_cols=["x"],
                validation="train_test",
            )

    def test_model_without_probabilities_reports_none(self, spark_session):
        model = classifiers.random_forest()
        df = spark_session.createDataFrame([(0.0, 0, 0.0), (1.0, 1, 1.0)], ["x", "y", "expected"])

        class NoProbabilityModel:
            def transform(self, frame):
                return frame.withColumn(model.PREDICTION_COL, F.col("expected"))

        model._feature_cols = ["x"]
        model._spark_model = NoProbabilityModel()
        scores = model._evaluate(df, "y")

        assert scores["accuracy"] == 1.0
        assert scores["auc_roc"] is None
        assert scores["auc_pr"] is None
        assert scores["log_loss"] is None

    def test_one_class_probability_vector_is_undefined(self, spark_session):
        # A binary model trained on label 0 only writes a 1-element probability vector:
        # it has no positive-class probability, so the probability metrics are undefined.
        from pyspark.ml.linalg import Vectors, VectorUDT

        model = classifiers.random_forest()
        df = spark_session.createDataFrame([(0.0, 0), (1.0, 1), (2.0, 0)], ["x", "y"])

        class OneClassModel:
            def transform(self, frame):
                one = F.udf(lambda _: Vectors.dense([1.0]), VectorUDT())
                return frame.withColumn(model.PREDICTION_COL, F.lit(0.0)).withColumn(
                    model.PROBABILITY_COL, one(F.col("x"))
                )

        model._feature_cols = ["x"]
        model._spark_model = OneClassModel()
        scores = model._evaluate(df, "y")

        assert scores["accuracy"] == pytest.approx(2 / 3)
        assert math.isnan(scores["auc_roc"])
        assert math.isnan(scores["auc_pr"])
        assert math.isnan(scores["log_loss"])


@pytest.fixture(scope="module")
def positives_share_one_key(spark_session):
    """90 binary rows whose 6 positives all share fold_key 'pos'.

    Whichever split holds 'pos' for validation trains on label 0 only.
    """
    rows = [
        ("pos" if i < 6 else f"k{i % 9}", float(i % 7) + (5.0 if i < 6 else 0.0), 1 if i < 6 else 0)
        for i in range(90)
    ]
    return spark_session.createDataFrame(rows, ["key", "x", "label"])


class TestSingleClassTrainingSplit:
    """A training split with one class must not crash fit (Spark RF writes 1-wide vectors)."""

    def test_kfold(self, positives_share_one_key):
        model = classifiers.random_forest(n_estimators=3, seed=2)
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            model.fit(
                positives_share_one_key,
                label_col="label",
                feature_cols=["x"],
                validation="kfold",
                n_folds=3,
                fold_key="key",
            )

        folds = model.validation_scores["fold_scores"]
        assert len(folds) == 3
        # Every fold is undefined: the 'pos' fold trained on one class, the others
        # validate on one class.
        for fold in folds:
            assert fold["auc_roc"] is None
            assert fold["auc_pr"] is None
            assert fold["accuracy"] is not None
        assert model.validation_scores["mean_auc_roc"] is None
        undefined = _undefined_warnings(record)
        assert len(undefined) == 1
        assert "auc_roc in 3 of 3 folds" in str(undefined[0].message)

    def test_train_test(self, positives_share_one_key):
        keys = positives_share_one_key.select("key").distinct()
        seed = next(
            s
            for s in range(500)
            if keys.where(F.col("key") == "pos")
            .select((_split_hash(keys, s, "key") / F.lit(float(2**31))).alias("b"))
            .first()["b"]
            < 0.2
        )
        model = classifiers.random_forest(n_estimators=3, seed=seed)
        model.fit(
            positives_share_one_key,
            label_col="label",
            feature_cols=["x"],
            validation="train_test",
            test_size=0.2,
            fold_key="key",
        )

        scores = model.validation_scores
        assert scores["auc_roc"] is None
        assert scores["auc_pr"] is None
        assert scores["log_loss"] is None
        assert scores["accuracy"] is not None
        json.dumps(scores, allow_nan=False)


class TestSearch:
    @pytest.fixture(autouse=True)
    def hyperopt(self):
        pytest.importorskip("hyperopt")

    @pytest.mark.parametrize("validation", ["train_test", "kfold"])
    def test_trial_with_undefined_metric_fails(self, monkeypatch, one_class_fold_frame, validation):
        import smallaxe.metrics.classification as classification
        from smallaxe.search import optimize

        monkeypatch.setattr(classification, "auc_roc", lambda *args, **kwargs: NAN)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValidationError, match="All trials failed") as info:
                optimize.run(
                    classifiers.random_forest(n_estimators=2, seed=SEED),
                    one_class_fold_frame,
                    label_col="label",
                    param_space={"max_depth": [2, 3]},
                    feature_cols=["x"],
                    metric="auc_roc",
                    validation=validation,
                    n_folds=2,
                    max_evals=2,
                    verbose=False,
                )
        assert "undefined" in str(info.value)

    def test_extract_metric_distinguishes_undefined_from_unavailable(self):
        from smallaxe.search.optimize import _extract_metric

        with pytest.raises(ValidationError, match="'auc_roc' is undefined"):
            _extract_metric({"auc_roc": None, "accuracy": 0.5}, "auc_roc", "train_test")
        with pytest.raises(ValidationError, match="undefined on every fold.*'mean_auc_roc'"):
            _extract_metric({"mean_auc_roc": None}, "auc_roc", "kfold")
        with pytest.raises(ValidationError, match="not available"):
            _extract_metric({"accuracy": 0.5}, "auc_roc", "train_test")
        with pytest.raises(ValidationError, match="undefined"):
            _extract_metric({"auc_roc": NAN}, "auc_roc", "train_test")
        assert math.isclose(_extract_metric({"auc_roc": 0.5}, "auc_roc", "train_test"), 0.5)
