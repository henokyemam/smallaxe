"""The algorithm seam, tested without Spark or any optional package.

The translation rule, the registry, the 0.8.x class-name map, and each
algorithm record are checked against fake estimators, so LightGBM and CatBoost
translation runs on machines without their JVM packages.
"""

import os
import sys

import pytest

from smallaxe.exceptions import DependencyError, ValidationError
from smallaxe.training import algorithm
from smallaxe.training.algorithm import (
    ALL_TASKS,
    CLASSIFICATION_TASKS,
    Algorithm,
    Dependency,
    Param,
    estimator_kwargs,
)
from smallaxe.training.base import Classifier, Model, Regressor
from tests.fakes import FakeEstimator, columns_for, install_fake_modules

TASKS = sorted(ALL_TASKS)


def _record(**overrides) -> Algorithm:
    """A small record with one of everything, for testing the translation rule."""
    fields = dict(
        name="fake",
        params=(
            Param("n_estimators", "trees", 10, spark="numTrees"),
            Param("max_depth", "depth", 5),
            Param("seed", "seed", None, setter="setSeed"),
            Param("weight", "class weight", None, spark="classWeight", tasks=CLASSIFICATION_TASKS),
        ),
        estimators={task: "fake_pkg.Estimator" for task in ALL_TASKS},
        models={task: "fake_pkg.Model" for task in ALL_TASKS},
        cols={"features": "featuresCol", "label": "labelCol", "prediction": "predictionCol"},
        task_params={"binary": {"objective": "binary"}},
    )
    fields.update(overrides)
    return Algorithm(**fields)


class TestEstimatorKwargs:
    def test_defaults_translate_to_spark_names(self):
        ctor, setters = estimator_kwargs(_record(), "simple_regression", {}, {})
        assert ctor == {"numTrees": 10, "max_depth": 5}
        assert setters == []

    def test_user_values_replace_defaults(self):
        ctor, _ = estimator_kwargs(_record(), "simple_regression", {"n_estimators": 50}, {})
        assert ctor["numTrees"] == 50

    def test_none_is_never_forwarded(self):
        ctor, setters = estimator_kwargs(_record(), "binary", {"max_depth": None}, {})
        assert "max_depth" not in ctor
        assert "classWeight" not in ctor
        assert setters == []

    def test_setter_params_are_returned_separately(self):
        ctor, setters = estimator_kwargs(_record(), "simple_regression", {"seed": 7}, {})
        assert "seed" not in ctor
        assert setters == [("setSeed", 7)]

    def test_params_are_visible_per_task(self):
        ctor, _ = estimator_kwargs(_record(), "binary", {"weight": 2.0}, {})
        assert ctor["classWeight"] == 2.0
        ctor, _ = estimator_kwargs(_record(), "simple_regression", {"weight": 2.0}, {})
        assert "classWeight" not in ctor

    def test_task_params_override_user_params(self):
        record = _record(params=(Param("objective", "loss", "regression"),))
        ctor, _ = estimator_kwargs(record, "binary", {"objective": "poisson"}, {})
        assert ctor["objective"] == "binary"

    def test_columns_win_over_everything(self):
        record = _record(params=(Param("label", "label", "wrong", spark="labelCol"),))
        ctor, _ = estimator_kwargs(record, "simple_regression", {}, {"label": "y"})
        assert ctor["labelCol"] == "y"

    def test_unmapped_column_roles_are_ignored(self):
        ctor, _ = estimator_kwargs(_record(), "binary", {}, columns_for("binary"))
        assert ctor["featuresCol"] == "features"
        assert "probability" not in ctor.values()

    def test_cols_none_ignores_columns(self):
        ctor, _ = estimator_kwargs(_record(cols=None), "binary", {}, columns_for("binary"))
        assert "features" not in ctor.values()


class TestAlgorithmRecord:
    def test_tasks_come_from_estimators(self):
        assert _record().tasks == ALL_TASKS

    def test_visible_params_keep_table_order(self):
        names = [p.name for p in _record().visible_params("binary")]
        assert names == ["n_estimators", "max_depth", "seed", "weight"]
        assert [p.name for p in _record().visible_params("simple_regression")] == names[:-1]

    def test_available_reprobes_each_call(self, monkeypatch):
        record = _record()
        assert not record.available()
        install_fake_modules(monkeypatch, record)
        assert record.available()
        monkeypatch.setitem(sys.modules, "fake_pkg", None)
        assert not record.available()

    def test_require_raises_with_install_hint(self):
        record = _record(dependency=Dependency("fake_pkg", "pip install fake"))
        with pytest.raises(DependencyError, match="fake_pkg is not installed") as info:
            record.require()
        assert info.value.install_command == "pip install fake"

    def test_classes_resolve_lazily(self, monkeypatch):
        record = _record()
        install_fake_modules(monkeypatch, record)
        assert record.estimator_class("binary") is FakeEstimator


class TestRegistry:
    def test_names_in_factory_order(self):
        assert algorithm.names() == ["random_forest", "xgboost", "lightgbm", "catboost"]

    def test_get_unknown_raises(self):
        with pytest.raises(ValidationError, match="Unknown algorithm 'nope'"):
            algorithm.get("nope")

    @pytest.mark.parametrize("name", algorithm.names())
    def test_every_record_covers_every_task(self, name):
        record = algorithm.get(name)
        assert record.name == name
        assert set(record.estimators) == ALL_TASKS
        assert set(record.models) == ALL_TASKS

    @pytest.mark.parametrize("name", algorithm.names())
    @pytest.mark.parametrize("task_type", ["regression", "classification"])
    def test_legacy_names_round_trip(self, name, task_type):
        legacy = algorithm.legacy_class_name(name, task_type)
        assert algorithm.LEGACY_CLASS_NAMES[legacy] == (name, task_type)

    def test_legacy_map_covers_all_eight_classes(self):
        assert len(algorithm.LEGACY_CLASS_NAMES) == 2 * len(algorithm.names())


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("name", algorithm.names())
class TestRecordTranslation:
    """Every record, every task, against a fake estimator."""

    @pytest.fixture
    def record(self, monkeypatch, name):
        record = algorithm.get(name)
        install_fake_modules(monkeypatch, record)
        return record

    def test_defaults_reach_the_estimator_under_spark_names(self, record, task):
        ctor, setters = estimator_kwargs(record, task, {}, columns_for(task))
        forwarded = set(ctor) | {setter for setter, _ in setters}
        for param in record.visible_params(task):
            key = param.setter or param.spark or param.name
            if param.default is None:
                assert key not in forwarded, (record.name, param.name)
            else:
                assert key in forwarded, (record.name, param.name)

    def test_column_names_are_wired(self, record, task):
        ctor, _ = estimator_kwargs(record, task, {}, columns_for(task))
        if record.cols is None:
            return
        for role, column in columns_for(task).items():
            assert ctor[record.cols[role]] == column

    def test_task_params_are_present(self, record, task):
        ctor, _ = estimator_kwargs(record, task, {}, columns_for(task))
        for key, value in record.task_params.get(task, {}).items():
            assert ctor[key] == value

    def test_model_builds_the_estimator(self, record, task):
        kind = Classifier if task in CLASSIFICATION_TASKS else Regressor
        model = kind(record, task)
        model.set_param({"n_estimators": 3, "seed": 11})
        estimator = model._create_spark_estimator(label_col="y")

        assert isinstance(estimator, FakeEstimator)
        ctor, setters = estimator_kwargs(record, task, model.get_params(), model._column_names("y"))
        assert estimator.kwargs == ctor
        assert all(call in estimator.calls for call in setters)
        if record.cols is None:
            assert ("setLabelCol", "y") in estimator.calls
            assert ("setFeaturesCol", "features") in estimator.calls
            assert ("setPredictionCol", "prediction") in estimator.calls


class TestLightGBM:
    @pytest.fixture(autouse=True)
    def fake(self, monkeypatch):
        install_fake_modules(monkeypatch, algorithm.get("lightgbm"))

    @pytest.mark.parametrize(
        "task, objective", [("binary", "binary"), ("multiclass", "multiclass")]
    )
    def test_classifier_objective_matches_task(self, task, objective):
        """SynapseML defaults to 'binary', so multiclass must be requested explicitly."""
        estimator = Classifier("lightgbm", task)._create_spark_estimator(label_col="y")
        assert estimator.kwargs["objective"] == objective

    def test_regressor_sets_no_objective(self):
        estimator = Regressor("lightgbm")._create_spark_estimator(label_col="y")
        assert "objective" not in estimator.kwargs

    def test_seed_goes_through_setter(self):
        estimator = Regressor("lightgbm").set_param({"seed": 3})._create_spark_estimator("y")
        assert ("setSeed", 3) in estimator.calls
        assert "seed" not in estimator.kwargs


class TestCatBoost:
    @pytest.fixture(autouse=True)
    def fake(self, monkeypatch):
        install_fake_modules(monkeypatch, algorithm.get("catboost"))

    @pytest.mark.parametrize(
        "task, loss",
        [("simple_regression", "RMSE"), ("binary", "Logloss"), ("multiclass", "MultiClass")],
    )
    def test_loss_function_follows_task(self, task, loss):
        kind = Classifier if task in CLASSIFICATION_TASKS else Regressor
        estimator = kind("catboost", task)._create_spark_estimator(label_col="y")
        assert estimator.kwargs["lossFunction"] == loss

    def test_scale_pos_weight_only_for_classification(self):
        assert "scale_pos_weight" in Classifier("catboost").params
        assert "scale_pos_weight" not in Regressor("catboost").params
        with pytest.raises(ValidationError, match="Invalid parameter"):
            Regressor("catboost").set_param({"scale_pos_weight": 2.0})

    def test_optional_params_are_not_forwarded_when_unset(self):
        estimator = Regressor("catboost")._create_spark_estimator(label_col="y")
        for key in ("subsample", "oneHotMaxSize", "randomSeed", "trainDir"):
            assert key not in estimator.kwargs
        assert estimator.kwargs["allowWritingFiles"] is False

    def test_training_dir_is_temporary_unless_configured(self):
        from smallaxe.training.catboost import _training_dir

        with _training_dir({"train_dir": None}) as extra:
            assert os.path.isdir(extra["trainDir"])
        assert not os.path.exists(extra["trainDir"])

        with _training_dir({"train_dir": "/configured"}) as extra:
            assert extra == {}

    def test_accepts_raw_categoricals_and_prepares_session(self):
        record = algorithm.get("catboost")
        assert record.accepts_raw_categoricals
        assert record.prepare.__name__ == "_ensure_synapseml_compat"


class TestImportances:
    def test_random_forest_reads_the_vector(self):
        class Vector:
            def toArray(self):
                return [0.25, 0.75]

        class Fitted:
            featureImportances = Vector()

        assert algorithm.get("random_forest").importances(Fitted(), 2) == [0.25, 0.75]

    def test_xgboost_fills_unused_features_with_zero(self):
        class Booster:
            def get_score(self, importance_type):
                assert importance_type == "gain"
                return {"f0": 3.0, "f2": 1.5}

        class Fitted:
            def get_booster(self):
                return Booster()

        assert algorithm.get("xgboost").importances(Fitted(), 3) == [3.0, 0.0, 1.5]

    def test_lightgbm_and_catboost_read_lists(self):
        class LightGBMFitted:
            def getFeatureImportances(self):
                return [4, 1]

        class CatBoostFitted:
            def getFeatureImportance(self):
                return [60.0, 40.0]

        assert algorithm.get("lightgbm").importances(LightGBMFitted(), 2) == [4.0, 1.0]
        assert algorithm.get("catboost").importances(CatBoostFitted(), 2) == [60.0, 40.0]


class TestModelConstruction:
    def test_unavailable_algorithm_raises_dependency_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "xgboost.spark", None)
        with pytest.raises(DependencyError, match=r"xgboost is not installed.*smallaxe\[xgboost\]"):
            Regressor("xgboost")

    def test_task_must_match_kind(self):
        with pytest.raises(ValidationError, match="Invalid regression task"):
            Regressor("random_forest", task="binary")
        with pytest.raises(ValidationError, match="Invalid classification task"):
            Classifier("random_forest", task="simple_regression")

    def test_params_follow_the_record(self):
        record = algorithm.get("random_forest")
        model = Regressor(record)
        assert model.algorithm is record
        assert set(model.params) == {p.name for p in record.params}
        assert model.default_params["n_estimators"] == 20

    def test_clone_copies_algorithm_task_and_params(self):
        model = Classifier("random_forest", "multiclass").set_param({"max_depth": 9})
        clone = model.clone()
        assert type(clone) is Classifier
        assert clone.algorithm is model.algorithm
        assert clone.task == "multiclass"
        assert clone.get_params() == model.get_params()
        assert not clone._is_fitted

    def test_model_kinds_are_base_classes(self):
        from smallaxe.training.base import BaseClassifier, BaseModel, BaseRegressor

        assert issubclass(Regressor, (Model, BaseRegressor, BaseModel))
        assert issubclass(Classifier, (Model, BaseClassifier, BaseModel))
        assert not hasattr(Regressor("random_forest"), "predict_proba")


class TestDeprecatedAliases:
    @pytest.mark.parametrize(
        "legacy, factory",
        [
            (name, "Regressors" if kind == "regression" else "Classifiers")
            for name, (_, kind) in algorithm.LEGACY_CLASS_NAMES.items()
        ],
    )
    def test_alias_warns_and_binds_its_algorithm(self, monkeypatch, legacy, factory):
        from smallaxe import training

        algorithm_name, task_type = algorithm.LEGACY_CLASS_NAMES[legacy]
        install_fake_modules(monkeypatch, algorithm.get(algorithm_name))
        alias = getattr(training, legacy)

        with pytest.warns(DeprecationWarning, match=f"{legacy} is deprecated.*{factory}"):
            model = alias()

        expected_kind = Regressor if task_type == "regression" else Classifier
        assert isinstance(model, expected_kind)
        assert model.algorithm.name == algorithm_name
        assert type(model.clone()) is expected_kind
