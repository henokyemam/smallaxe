"""Model-side behaviour of the algorithm seam, checked with fake estimators.

How ``Regressor`` / ``Classifier`` drive an algorithm record: hook ordering, the
fit context, importances alignment, what ``metadata.json`` records, and how
``load`` resolves new and 0.8.x artifacts. The real algorithms are exercised in
``test_model_contract.py``.
"""

import dataclasses
import json
import sys
from contextlib import contextmanager

import pytest

from smallaxe.exceptions import DependencyError, ModelNotFittedError, ValidationError
from smallaxe.training import (
    Classifiers,
    RandomForestClassifier,
    Regressors,
    XGBoostClassifier,
    XGBoostRegressor,
    algorithm,
)
from smallaxe.training import catboost as catboost_module
from smallaxe.training.algorithm import ALL_TASKS, Algorithm, Param
from smallaxe.training.base import Classifier, Model, Regressor
from tests.fakes import FakeEstimator, FakeModel, install_fake_modules

LEGACY_STATE = {
    "__class__": "XGBoostClassifier",
    "__module__": "smallaxe.training.xgboost",
    "task": "binary",
    "params": {"max_depth": 3},
    "feature_cols": ["a", "b"],
    "label_col": "y",
    "exclude_cols": [],
    "is_fitted": True,
    "metadata": {},
    "validation_scores": None,
}


@pytest.fixture
def df(spark_session):
    rows = [(1.0, 2.0, 0), (2.0, 3.0, 1), (3.0, 5.0, 0), (4.0, 7.0, 1)]
    return spark_session.createDataFrame(rows, ["a", "b", "y"])


def _record(**overrides) -> Algorithm:
    fields = dict(
        name="fake",
        params=(
            Param("n_estimators", "trees", 10, spark="numTrees"),
            Param("seed", "seed", None, setter="setSeed"),
        ),
        estimators={task: "fake_pkg.Estimator" for task in ALL_TASKS},
        models={task: "fake_pkg.Model" for task in ALL_TASKS},
        cols={
            "features": "featuresCol",
            "label": "labelCol",
            "prediction": "predictionCol",
            "probability": "probabilityCol",
            "raw_prediction": "rawPredictionCol",
        },
    )
    fields.update(overrides)
    return Algorithm(**fields)


@pytest.fixture
def fake_xgboost(monkeypatch):
    """The registered xgboost record, backed by fake estimator modules."""
    record = algorithm.get("xgboost")
    install_fake_modules(monkeypatch, record)
    return record


def _write_legacy_artifact(path, state=LEGACY_STATE):
    (path / "spark_model").mkdir(parents=True)
    (path / "metadata.json").write_text(json.dumps(state))
    return str(path)


class TestFitWiring:
    def test_fit_assembles_features_and_builds_the_estimator(self, monkeypatch, df):
        record = _record()
        install_fake_modules(monkeypatch, record)
        model = Regressor(record).set_param({"seed": 5})

        model._fit_spark_model(df, "y", ["a", "b"])

        estimator = model._spark_model.estimator
        assert estimator.kwargs["labelCol"] == "y"
        assert estimator.kwargs["featuresCol"] == "features"
        assert "probabilityCol" not in estimator.kwargs
        assert ("setSeed", 5) in estimator.calls
        assert "features" in estimator.fitted_on.columns
        assert model._feature_cols == ["a", "b"]
        assert model._label_col == "y"

    def test_classifier_wires_probability_columns(self, monkeypatch, df):
        record = _record()
        install_fake_modules(monkeypatch, record)
        model = Classifier(record, "binary")

        model._fit_spark_model(df, "y", ["a", "b"])

        kwargs = model._spark_model.estimator.kwargs
        assert kwargs["probabilityCol"] == "probability"
        assert kwargs["rawPredictionCol"] == "rawPrediction"

    def test_prepare_runs_before_the_estimator_is_built(self, monkeypatch, df):
        events = []
        record = _record(prepare=lambda: events.append("prepare"))
        install_fake_modules(monkeypatch, record)

        class Recording(FakeEstimator):
            def __init__(self, **kwargs):
                events.append("build")
                super().__init__(**kwargs)

        sys.modules["fake_pkg"].Estimator = Recording

        Regressor(record)._fit_spark_model(df, "y", ["a", "b"])

        assert events == ["prepare", "build"]

    def test_fit_context_adds_kwargs_and_always_exits(self, monkeypatch, df):
        events = []

        @contextmanager
        def training_context(values):
            events.append(("enter", values["n_estimators"]))
            try:
                yield {"trainDir": "/scratch/catboost"}
            finally:
                events.append("exit")

        record = _record(fit_context=training_context)
        install_fake_modules(monkeypatch, record)
        model = Regressor(record).set_param({"n_estimators": 3})

        model._fit_spark_model(df, "y", ["a", "b"])
        assert model._spark_model.estimator.kwargs["trainDir"] == "/scratch/catboost"
        assert events == [("enter", 3), "exit"]

        class Exploding(FakeEstimator):
            def fit(self, df):
                raise RuntimeError("boom")

        sys.modules["fake_pkg"].Estimator = Exploding
        events.clear()
        with pytest.raises(RuntimeError, match="boom"):
            model._fit_spark_model(df, "y", ["a", "b"])
        assert events == [("enter", 3), "exit"]

    def test_public_fit_marks_the_model_fitted(self, monkeypatch, df):
        record = _record()
        install_fake_modules(monkeypatch, record)
        model = Classifier(record, "binary")

        model.fit(df, label_col="y", feature_cols=["a", "b"])

        assert model._is_fitted
        assert isinstance(model._spark_model, FakeModel)
        assert model.metadata["task"] == "binary"


class TestImportancesWiring:
    def test_scores_are_zipped_with_feature_cols(self, monkeypatch, df):
        record = _record(importances=lambda model, n: [float(i) for i in range(n)])
        install_fake_modules(monkeypatch, record)
        model = Regressor(record)
        model._fit_spark_model(df, "y", ["a", "b"])

        assert model.feature_importances == {"a": 0.0, "b": 1.0}

    def test_none_when_the_record_has_no_importances(self, monkeypatch, df):
        record = _record()
        install_fake_modules(monkeypatch, record)
        model = Regressor(record)
        model._fit_spark_model(df, "y", ["a", "b"])

        assert model.feature_importances is None

    def test_unfitted_model_raises(self, monkeypatch):
        record = _record()
        install_fake_modules(monkeypatch, record)
        with pytest.raises(ModelNotFittedError):
            _ = Regressor(record).feature_importances


class TestPersistence:
    def test_metadata_records_algorithm_and_task(self, fake_xgboost, df, tmp_path):
        model = Classifiers.xgboost(task="multiclass", n_estimators=7)
        model.fit(df, label_col="y", feature_cols=["a", "b"])
        path = str(tmp_path / "model")

        model.save(path)

        state = json.loads((tmp_path / "model" / "metadata.json").read_text())
        assert state["algorithm"] == "xgboost"
        assert state["task"] == "multiclass"
        assert state["__class__"] == "Classifier"
        assert state["params"] == {"n_estimators": 7}
        assert model._spark_model.saved_to.endswith("spark_model")

    def test_round_trip_through_model_load(self, fake_xgboost, df, tmp_path):
        model = Classifiers.xgboost(task="multiclass", n_estimators=7)
        model.fit(df, label_col="y", feature_cols=["a", "b"])
        path = str(tmp_path / "model")
        model.save(path)

        loaded = Model.load(path)

        assert type(loaded) is Classifier
        assert loaded.algorithm is fake_xgboost
        assert loaded.task == "multiclass"
        assert loaded.get_param("n_estimators") == 7
        assert loaded._feature_cols == ["a", "b"]
        assert loaded._is_fitted
        assert isinstance(loaded._spark_model, FakeModel)
        assert loaded._spark_model.loaded_from.endswith("spark_model")

    def test_prepare_runs_on_load(self, monkeypatch, df, tmp_path):
        events = []
        record = dataclasses.replace(
            algorithm.get("catboost"), prepare=lambda: events.append("prepare")
        )
        install_fake_modules(monkeypatch, record)
        monkeypatch.setattr(catboost_module, "ALGORITHM", record)
        model = Regressors.catboost()
        model.fit(df, label_col="y", feature_cols=["a", "b"])
        path = str(tmp_path / "model")
        model.save(path)
        events.clear()

        Regressors.load(path)

        assert events == ["prepare"]

    def test_legacy_metadata_loads_by_class_name(self, fake_xgboost, tmp_path, spark_session):
        path = _write_legacy_artifact(tmp_path / "legacy")

        loaded = Classifiers.load(path)

        assert type(loaded) is Classifier
        assert loaded.algorithm.name == "xgboost"
        assert loaded.task == "binary"
        assert loaded.get_param("max_depth") == 3
        assert loaded._feature_cols == ["a", "b"]
        assert isinstance(loaded._spark_model, FakeModel)

    def test_legacy_metadata_keeps_the_kind(self, fake_xgboost, tmp_path, spark_session):
        path = _write_legacy_artifact(tmp_path / "legacy")

        with pytest.raises(ValidationError, match="not a supported regressor"):
            Regressors.load(path)

    def test_alias_load_checks_kind_and_algorithm(self, fake_xgboost, tmp_path, spark_session):
        path = _write_legacy_artifact(tmp_path / "legacy")

        with pytest.raises(ValidationError, match="Model type mismatch"):
            XGBoostRegressor.load(path)
        with pytest.raises(ValidationError, match="Model type mismatch"):
            RandomForestClassifier.load(path)
        assert type(XGBoostClassifier.load(path)) is Classifier

    def test_unknown_metadata_raises(self, tmp_path, spark_session):
        path = _write_legacy_artifact(
            tmp_path / "mystery", {**LEGACY_STATE, "__class__": "MysteryModel"}
        )

        with pytest.raises(ValidationError, match="does not contain 'algorithm'"):
            Model.load(path)

    def test_missing_dependency_raises_dependency_error(self, monkeypatch, tmp_path, spark_session):
        path = _write_legacy_artifact(tmp_path / "legacy")
        monkeypatch.setitem(sys.modules, "xgboost.spark", None)

        with pytest.raises(DependencyError, match="xgboost is not installed"):
            Model.load(path)


class TestPipelineEncoderRequirement:
    def test_algorithm_record_decides_whether_an_encoder_is_required(self, monkeypatch):
        from smallaxe.exceptions import PreprocessingError
        from smallaxe.pipeline import Pipeline

        install_fake_modules(monkeypatch, algorithm.get("catboost"))

        # CatBoost accepts raw categoricals, so a model-only pipeline is valid.
        steps = [("model", Regressors.catboost())]
        Pipeline(steps)._validate_preprocessing_requirements(steps, categorical_cols=["city"])

        steps = [("model", Regressors.random_forest())]
        with pytest.raises(PreprocessingError):
            Pipeline(steps)._validate_preprocessing_requirements(steps, categorical_cols=["city"])


class TestOptimizeUsesClone:
    def test_search_builds_candidates_from_clone(self, monkeypatch, df):
        pytest.importorskip("hyperopt")
        from hyperopt import hp

        from smallaxe.search import optimize

        record = _record(importances=None)
        install_fake_modules(monkeypatch, record)
        model = Regressor(record).set_param({"seed": 1})
        clones = []
        original_clone = Regressor.clone

        def spy(self):
            clone = original_clone(self)
            clones.append(clone)
            return clone

        monkeypatch.setattr(Regressor, "clone", spy)
        monkeypatch.setattr(
            Model, "_evaluate", lambda self, frame, label_col: {"rmse": 1.0, "r2": 0.0}
        )

        result = optimize.run(
            model,
            df,
            label_col="y",
            param_space={"n_estimators": hp.quniform("n_estimators", 2, 4, 1)},
            metric="rmse",
            validation="train_test",
            max_evals=2,
            seed=0,
        )

        assert len(clones) >= 2
        assert all(c.algorithm is record and c.get_param("seed") == 1 for c in clones)
        assert result.best_model.algorithm is record
