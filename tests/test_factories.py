"""The regressors / classifiers functions: construction, availability, and loading.

The deprecated Regressors / Classifiers classes are tested in test_deprecations.py.
"""

import os
import sys
import tempfile

import pytest

from smallaxe.exceptions import DependencyError, ValidationError
from smallaxe.training import Classifier, Regressor, classifiers, regressors

# (factory method, estimator module, dependency package, extra) for the optional algorithms.
OPTIONAL = [
    ("xgboost", "xgboost.spark", "xgboost", "smallaxe[xgboost]"),
    ("lightgbm", "synapse.ml.lightgbm", "synapseml", "smallaxe[lightgbm]"),
    ("catboost", "catboost_spark", "catboost_spark", "smallaxe[catboost]"),
]


@pytest.fixture
def regression_df(spark_session):
    data = [(i, 20.0 + (i % 30), 40000.0 + (i * 1000), 50.0 + (i * 5)) for i in range(1, 101)]
    return spark_session.createDataFrame(data, ["id", "age", "income", "target"])


@pytest.fixture
def classification_df(spark_session):
    data = [(i, 20.0 + (i % 30), 40000.0 + (i * 1000), i % 2) for i in range(1, 101)]
    return spark_session.createDataFrame(data, ["id", "age", "income", "label"])


@pytest.fixture
def without_optional_algorithms(monkeypatch):
    """Make every optional estimator module unimportable, whatever is installed."""
    for _, module, _, _ in OPTIONAL:
        monkeypatch.setitem(sys.modules, module, None)


class TestRegressorsFactory:
    def test_random_forest_creates_regressor(self):
        model = regressors.random_forest()
        assert isinstance(model, Regressor)
        assert model.algorithm.name == "random_forest"
        assert model.task == "simple_regression"
        assert model.task_type == "regression"

    def test_random_forest_with_params(self):
        model = regressors.random_forest(n_estimators=100, max_depth=10)
        assert model.get_param("n_estimators") == 100
        assert model.get_param("max_depth") == 10

    def test_random_forest_default_params(self):
        model = regressors.random_forest()
        assert model.get_param("n_estimators") == 20
        assert model.get_param("max_depth") == 5

    def test_invalid_param_raises(self):
        with pytest.raises(ValidationError, match="Invalid parameter"):
            regressors.random_forest(not_a_param=1)

    def test_random_forest_fit_predict(self, regression_df):
        model = regressors.random_forest(n_estimators=10, seed=42)
        model.fit(regression_df, label_col="target", feature_cols=["age", "income"])

        predictions = model.predict(regression_df)
        assert "predict_label" in predictions.columns
        assert predictions.count() == regression_df.count()

    def test_algorithms(self):
        assert regressors.algorithms() == ["random_forest", "xgboost", "lightgbm", "catboost"]

    def test_available_models_reports_optional_regressors(self):
        models = regressors.available_models()

        assert list(models) == ["random_forest", "xgboost", "lightgbm", "catboost"]
        assert models["random_forest"] == {
            "available": True,
            "dependency": None,
            "install_hint": None,
        }
        assert models["xgboost"]["dependency"] == "xgboost"
        assert models["lightgbm"]["dependency"] == "synapseml"
        assert models["catboost"]["dependency"] == "catboost_spark"
        for name in ("xgboost", "lightgbm", "catboost"):
            assert isinstance(models[name]["available"], bool)
            assert (models[name]["install_hint"] is None) == models[name]["available"]

    @pytest.mark.parametrize("name, module, package, extra", OPTIONAL)
    def test_missing_dependency_raises_actionable_error(
        self, monkeypatch, name, module, package, extra
    ):
        monkeypatch.setitem(sys.modules, module, None)

        with pytest.raises(DependencyError, match=f"{package} is not installed") as info:
            getattr(regressors, name)()
        assert extra in info.value.install_command

    def test_available_models_includes_install_hints_for_missing(self, without_optional_algorithms):
        models = regressors.available_models()

        assert models["random_forest"]["available"] is True
        for name, _, package, extra in OPTIONAL:
            assert models[name]["available"] is False
            assert models[name]["dependency"] == package
            assert extra in models[name]["install_hint"]

    def test_load_regressor(self, regression_df):
        model = regressors.random_forest(n_estimators=10, seed=42)
        model.fit(regression_df, label_col="target", feature_cols=["age", "income"])

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model")
            model.save(save_path)

            loaded = regressors.load(save_path)
            assert isinstance(loaded, Regressor)
            assert loaded.algorithm.name == "random_forest"
            assert loaded.get_param("n_estimators") == 10
            assert loaded._is_fitted

    def test_load_regressor_can_predict(self, regression_df):
        model = regressors.random_forest(seed=42)
        model.fit(regression_df, label_col="target", feature_cols=["age", "income"])

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model")
            model.save(save_path)

            predictions = regressors.load(save_path).predict(regression_df)
            assert "predict_label" in predictions.columns
            assert predictions.count() == regression_df.count()

    def test_load_nonexistent_path_raises_error(self):
        with pytest.raises(ValidationError, match="'/nonexistent/path' does not exist"):
            regressors.load("/nonexistent/path")

    def test_load_directory_without_metadata_raises_error(self, tmp_path):
        with pytest.raises(ValidationError, match="expected metadata.json"):
            regressors.load(str(tmp_path))

    def test_load_classifier_as_regressor_raises_error(self, classification_df):
        model = classifiers.random_forest()
        model.fit(classification_df, label_col="label", feature_cols=["age", "income"])

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model")
            model.save(save_path)

            with pytest.raises(ValidationError, match="not a supported regressor"):
                regressors.load(save_path)


class TestClassifiersFactory:
    def test_random_forest_creates_classifier(self):
        model = classifiers.random_forest()
        assert isinstance(model, Classifier)
        assert model.algorithm.name == "random_forest"
        assert model.task == "binary"
        assert model.task_type == "classification"

    def test_random_forest_multiclass(self):
        assert classifiers.random_forest(task="multiclass").task == "multiclass"

    def test_invalid_task_raises(self):
        with pytest.raises(ValidationError, match="Invalid classification task"):
            classifiers.random_forest(task="simple_regression")

    def test_random_forest_with_params(self):
        model = classifiers.random_forest(n_estimators=100, max_depth=10)
        assert model.get_param("n_estimators") == 100
        assert model.get_param("max_depth") == 10

    def test_random_forest_fit_predict_proba(self, classification_df):
        model = classifiers.random_forest(n_estimators=10, seed=42)
        model.fit(classification_df, label_col="label", feature_cols=["age", "income"])

        predictions = model.predict(classification_df)
        assert "predict_label" in predictions.columns
        proba = model.predict_proba(classification_df)
        assert "probability" in proba.columns
        assert proba.count() == classification_df.count()

    def test_algorithms(self):
        assert classifiers.algorithms() == ["random_forest", "xgboost", "lightgbm", "catboost"]

    def test_available_models_reports_optional_classifiers(self):
        models = classifiers.available_models()

        assert list(models) == ["random_forest", "xgboost", "lightgbm", "catboost"]
        assert models["random_forest"] == {
            "available": True,
            "dependency": None,
            "install_hint": None,
        }

    @pytest.mark.parametrize("name, module, package, extra", OPTIONAL)
    def test_missing_dependency_raises_actionable_error(
        self, monkeypatch, name, module, package, extra
    ):
        monkeypatch.setitem(sys.modules, module, None)

        with pytest.raises(DependencyError, match=f"{package} is not installed") as info:
            getattr(classifiers, name)(task="binary")
        assert extra in info.value.install_command

    def test_available_models_includes_install_hints_for_missing(self, without_optional_algorithms):
        models = classifiers.available_models()

        for name, _, _, extra in OPTIONAL:
            assert models[name]["available"] is False
            assert extra in models[name]["install_hint"]

    def test_load_classifier(self, classification_df):
        model = classifiers.random_forest(task="binary", n_estimators=10, seed=42)
        model.fit(classification_df, label_col="label", feature_cols=["age", "income"])

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model")
            model.save(save_path)

            loaded = classifiers.load(save_path)
            assert isinstance(loaded, Classifier)
            assert loaded.algorithm.name == "random_forest"
            assert loaded.task == "binary"

            predictions = loaded.predict(classification_df)
            proba = loaded.predict_proba(classification_df)
            assert predictions.count() == classification_df.count()
            assert "probability" in proba.columns

    def test_load_nonexistent_path_raises_error(self):
        with pytest.raises(ValidationError, match="'/nonexistent/path' does not exist"):
            classifiers.load("/nonexistent/path")

    def test_load_directory_without_metadata_raises_error(self, tmp_path):
        with pytest.raises(ValidationError, match="expected metadata.json"):
            classifiers.load(str(tmp_path))

    def test_load_regressor_as_classifier_raises_error(self, regression_df):
        model = regressors.random_forest()
        model.fit(regression_df, label_col="target", feature_cols=["age", "income"])

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model")
            model.save(save_path)

            with pytest.raises(ValidationError, match="not a supported classifier"):
                classifiers.load(save_path)


class TestFactoryIntegration:
    def test_regressor_full_workflow(self, regression_df):
        model = regressors.random_forest(n_estimators=15, max_depth=6, seed=42)
        model.fit(
            regression_df,
            label_col="target",
            feature_cols=["age", "income"],
            validation="train_test",
            test_size=0.2,
        )

        assert model._is_fitted
        assert "rmse" in model.validation_scores
        original_count = model.predict(regression_df).count()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model")
            model.save(save_path)
            loaded = regressors.load(save_path)

            assert loaded.predict(regression_df).count() == original_count
            assert loaded.get_param("n_estimators") == 15
            assert loaded.get_param("max_depth") == 6

    def test_classifier_full_workflow(self, classification_df):
        model = classifiers.random_forest(task="binary", n_estimators=15, max_depth=6, seed=42)
        model.fit(
            classification_df,
            label_col="label",
            feature_cols=["age", "income"],
            validation="train_test",
            test_size=0.2,
        )

        assert model._is_fitted
        assert "accuracy" in model.validation_scores
        original_count = model.predict(classification_df).count()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model")
            model.save(save_path)
            loaded = classifiers.load(save_path)

            assert loaded.predict(classification_df).count() == original_count
            assert loaded.predict_proba(classification_df).count() == original_count
            assert loaded.get_param("n_estimators") == 15
            assert loaded.task == "binary"

    def test_model_interoperability(self, regression_df, classification_df):
        reg_model = regressors.random_forest(seed=42)
        clf_model = classifiers.random_forest(seed=42)
        reg_model.fit(regression_df, label_col="target", feature_cols=["age", "income"])
        clf_model.fit(classification_df, label_col="label", feature_cols=["age", "income"])

        with tempfile.TemporaryDirectory() as tmpdir:
            reg_path = os.path.join(tmpdir, "reg_model")
            clf_path = os.path.join(tmpdir, "clf_model")
            reg_model.save(reg_path)
            clf_model.save(clf_path)

            assert isinstance(regressors.load(reg_path), Regressor)
            assert isinstance(classifiers.load(clf_path), Classifier)

            with pytest.raises(ValidationError):
                regressors.load(clf_path)
            with pytest.raises(ValidationError):
                classifiers.load(reg_path)
