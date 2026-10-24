"""The contract every algorithm satisfies, parametrized over (algorithm, task).

Random Forest always runs. The others run wherever their packages are installed:
XGBoost in the CI ``test-xgboost`` job, LightGBM and CatBoost on Databricks
(``examples/databricks_all_algorithms_validation.py``). Translation details that
need no JVM are in ``test_algorithm.py``.
"""

import pytest
from pyspark.ml.functions import vector_to_array

from smallaxe.exceptions import ModelNotFittedError
from smallaxe.training import Classifier, Classifiers, Regressor, Regressors, algorithm

TASKS = ("simple_regression", "binary", "multiclass")
CASES = [(name, task) for name in algorithm.names() for task in TASKS]
FEATURES = ["age", "income"]
# Every record has both; keeps the twelve cases fast.
FAST = {"n_estimators": 5, "seed": 42}


@pytest.fixture(scope="module")
def frame(spark_session):
    rows = [
        (i, 20.0 + (i % 30), 40000.0 + i * 1000.0, 50.0 + i * 5 + (i % 7), i % 2, i % 3)
        for i in range(1, 121)
    ]
    return spark_session.createDataFrame(
        rows, ["id", "age", "income", "target", "binary", "multiclass"]
    )


def _label(task):
    return "target" if task == "simple_regression" else task


def _make(name, task, **params):
    if not algorithm.get(name).available():
        pytest.skip(f"{name} is not installed")
    if task == "simple_regression":
        return getattr(Regressors, name)(**params)
    return getattr(Classifiers, name)(task=task, **params)


def _fit(name, task, frame, **fit_kwargs):
    model = _make(name, task, **FAST)
    return model.fit(frame, label_col=_label(task), feature_cols=FEATURES, **fit_kwargs)


@pytest.mark.parametrize("name, task", CASES, ids=[f"{n}-{t}" for n, t in CASES])
class TestModelContract:
    def test_factory_returns_the_right_kind(self, name, task):
        model = _make(name, task)
        expected = Regressor if task == "simple_regression" else Classifier
        assert type(model) is expected
        assert model.algorithm.name == name
        assert model.task == task

    def test_fit_predict_preserves_rows(self, frame, name, task):
        model = _fit(name, task, frame)
        predictions = model.predict(frame)

        assert predictions.count() == frame.count()
        assert "predict_label" in predictions.columns
        assert "features" not in predictions.columns
        assert model._feature_cols == FEATURES

    def test_predict_proba_has_one_entry_per_class(self, frame, name, task):
        if task == "simple_regression":
            pytest.skip("regression has no probabilities")
        proba = _fit(name, task, frame).predict_proba(frame)

        assert proba.count() == frame.count()
        first = proba.select(vector_to_array("probability").alias("p")).first()["p"]
        assert len(first) == (2 if task == "binary" else 3)

    def test_train_test_validation_scores(self, frame, name, task):
        model = _fit(name, task, frame, validation="train_test", test_size=0.25)
        scores = model.validation_scores

        expected = (
            {"rmse", "r2", "mae"} if task == "simple_regression" else {"accuracy", "f1_score"}
        )
        assert expected <= set(scores)
        assert scores["validation_type"] == "train_test"

    def test_kfold_validation_scores(self, frame, name, task):
        model = _fit(name, task, frame, validation="kfold", n_folds=3)
        scores = model.validation_scores

        metric = "mean_rmse" if task == "simple_regression" else "mean_accuracy"
        assert metric in scores
        assert len(scores["fold_scores"]) == 3

    def test_feature_importances_cover_every_feature(self, frame, name, task):
        importances = _fit(name, task, frame).feature_importances

        assert set(importances) == set(FEATURES)
        assert all(isinstance(v, float) and v >= 0 for v in importances.values())

    def test_save_load_predicts_identically(self, frame, name, task, tmp_path):
        model = _fit(name, task, frame)
        path = str(tmp_path / "model")
        model.save(path)

        factory = Regressors if task == "simple_regression" else Classifiers
        loaded = factory.load(path)

        assert type(loaded) is type(model)
        assert loaded.algorithm is model.algorithm
        assert loaded.task == task
        assert loaded.get_params() == model.get_params()
        assert loaded._is_fitted
        before = model.predict(frame).select("id", "predict_label").orderBy("id").collect()
        after = loaded.predict(frame).select("id", "predict_label").orderBy("id").collect()
        assert before == after

    def test_clone_is_an_unfitted_copy(self, frame, name, task):
        model = _fit(name, task, frame)
        clone = model.clone()

        assert type(clone) is type(model)
        assert clone.algorithm is model.algorithm
        assert clone.get_params() == model.get_params()
        with pytest.raises(ModelNotFittedError):
            _ = clone.validation_scores
