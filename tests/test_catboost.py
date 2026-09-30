"""Tests for CatBoostRegressor and CatBoostClassifier."""

import pytest

from smallaxe.exceptions import DependencyError, ValidationError

try:
    import catboost_spark  # noqa: F401

    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


@pytest.mark.skipif(not CATBOOST_AVAILABLE, reason="catboost_spark not installed")
class TestCatBoostRegressorInit:
    """Tests for CatBoostRegressor initialization."""

    def test_default_task(self):
        """Test that default task is 'simple_regression'."""
        from smallaxe.training.catboost import CatBoostRegressor

        model = CatBoostRegressor()
        assert model.task == "simple_regression"
        assert model.task_type == "regression"

    def test_invalid_task_raises_error(self):
        """Test that invalid task raises ValidationError."""
        from smallaxe.training.catboost import CatBoostRegressor

        with pytest.raises(ValidationError, match="Invalid regression task"):
            CatBoostRegressor(task="binary")


@pytest.mark.skipif(not CATBOOST_AVAILABLE, reason="catboost_spark not installed")
class TestCatBoostRegressorParams:
    """Tests for CatBoostRegressor params."""

    def test_params_dict(self):
        """Test that params returns parameter descriptions."""
        from smallaxe.training.catboost import CatBoostRegressor

        model = CatBoostRegressor()
        params = model.params

        assert "n_estimators" in params
        assert "max_depth" in params
        assert "learning_rate" in params
        assert "subsample" in params
        assert "l2_leaf_reg" in params
        assert "random_strength" in params
        assert "one_hot_max_size" in params
        assert "allow_writing_files" in params
        assert "train_dir" in params
        assert "seed" in params

    def test_default_params_dict(self):
        """Test that default_params returns default values."""
        from smallaxe.training.catboost import CatBoostRegressor

        model = CatBoostRegressor()
        defaults = model.default_params

        assert defaults["n_estimators"] == 100
        assert defaults["max_depth"] == 6
        assert defaults["learning_rate"] == 0.03
        assert defaults["subsample"] is None
        assert defaults["l2_leaf_reg"] == 3.0
        assert defaults["random_strength"] == 1.0
        assert defaults["one_hot_max_size"] is None
        assert defaults["allow_writing_files"] is False
        assert defaults["train_dir"] is None
        assert defaults["seed"] is None

    def test_set_param_multiple(self):
        """Test setting multiple parameters."""
        from smallaxe.training.catboost import CatBoostRegressor

        model = CatBoostRegressor()
        model.set_param({"n_estimators": 50, "max_depth": 4, "learning_rate": 0.1})

        assert model.get_param("n_estimators") == 50
        assert model.get_param("max_depth") == 4
        assert model.get_param("learning_rate") == 0.1

    def test_set_param_invalid_key(self):
        """Test that invalid parameter key raises ValidationError."""
        from smallaxe.training.catboost import CatBoostRegressor

        model = CatBoostRegressor()
        with pytest.raises(ValidationError, match="Invalid parameter"):
            model.set_param({"invalid_param": 10})


@pytest.mark.skipif(not CATBOOST_AVAILABLE, reason="catboost_spark not installed")
class TestCatBoostClassifierInit:
    """Tests for CatBoostClassifier initialization."""

    def test_default_task(self):
        """Test that default task is 'binary'."""
        from smallaxe.training.catboost import CatBoostClassifier

        model = CatBoostClassifier()
        assert model.task == "binary"
        assert model.task_type == "classification"

    def test_multiclass_task(self):
        """Test that multiclass task is accepted."""
        from smallaxe.training.catboost import CatBoostClassifier

        model = CatBoostClassifier(task="multiclass")
        assert model.task == "multiclass"

    def test_invalid_task_raises_error(self):
        """Test that invalid task raises ValidationError."""
        from smallaxe.training.catboost import CatBoostClassifier

        with pytest.raises(ValidationError, match="Invalid classification task"):
            CatBoostClassifier(task="simple_regression")


@pytest.mark.skipif(not CATBOOST_AVAILABLE, reason="catboost_spark not installed")
class TestCatBoostClassifierParams:
    """Tests for CatBoostClassifier params."""

    def test_params_dict(self):
        """Test that params returns parameter descriptions."""
        from smallaxe.training.catboost import CatBoostClassifier

        model = CatBoostClassifier()
        params = model.params

        assert "n_estimators" in params
        assert "max_depth" in params
        assert "learning_rate" in params
        assert "subsample" in params
        assert "l2_leaf_reg" in params
        assert "random_strength" in params
        assert "one_hot_max_size" in params
        assert "scale_pos_weight" in params
        assert "allow_writing_files" in params
        assert "train_dir" in params
        assert "seed" in params

    def test_default_params_dict(self):
        """Test that default_params returns default values."""
        from smallaxe.training.catboost import CatBoostClassifier

        model = CatBoostClassifier()
        defaults = model.default_params

        assert defaults["n_estimators"] == 100
        assert defaults["max_depth"] == 6
        assert defaults["learning_rate"] == 0.03
        assert defaults["subsample"] is None
        assert defaults["l2_leaf_reg"] == 3.0
        assert defaults["random_strength"] == 1.0
        assert defaults["one_hot_max_size"] is None
        assert defaults["scale_pos_weight"] is None
        assert defaults["allow_writing_files"] is False
        assert defaults["train_dir"] is None
        assert defaults["seed"] is None

    def test_set_param_multiple(self):
        """Test setting multiple parameters."""
        from smallaxe.training.catboost import CatBoostClassifier

        model = CatBoostClassifier()
        model.set_param({"n_estimators": 50, "max_depth": 4, "learning_rate": 0.1})

        assert model.get_param("n_estimators") == 50
        assert model.get_param("max_depth") == 4
        assert model.get_param("learning_rate") == 0.1


# =============================================================================
# DependencyError Tests (run always, even without CatBoost)
# =============================================================================


@pytest.mark.skipif(
    CATBOOST_AVAILABLE,
    reason="Test only runs when catboost_spark is NOT installed",
)
class TestCatBoostDependencyError:
    """Tests for DependencyError when catboost_spark is not installed."""

    def test_regressor_raises_dependency_error(self):
        """Test that CatBoostRegressor raises DependencyError when unavailable."""
        from smallaxe.training.catboost import CatBoostRegressor

        with pytest.raises(DependencyError, match="catboost_spark is not installed"):
            CatBoostRegressor()

    def test_classifier_raises_dependency_error(self):
        """Test that CatBoostClassifier raises DependencyError when unavailable."""
        from smallaxe.training.catboost import CatBoostClassifier

        with pytest.raises(DependencyError, match="catboost_spark is not installed"):
            CatBoostClassifier()


# =============================================================================
# SynapseML Compatibility Tests (no JVM packages needed)
# =============================================================================


class _FakeJavaParam:
    def w(self, value):
        return ("java", value)


class _FakeJavaObj:
    def getParam(self, name):
        return _FakeJavaParam()


class _FakeParams:
    _java_obj = _FakeJavaObj()

    def _resolveParam(self, param):
        class _P:
            name = param

        return _P()


class _FakeStage:
    def __init__(self, class_name):
        self._class_name = class_name

    def getClass(self):
        stage = self

        class _C:
            def getName(self):
                return stage._class_name

        return _C()


def _synapseml_style(fn):
    fn.__module__ = "synapse.ml.core.serialize.java_params_patch"
    return fn


class TestCatBoostSynapseMLCompat:
    """SynapseML's JavaParams patches must not break CatBoost params or model loading."""

    @pytest.fixture
    def synapseml_patched(self, monkeypatch):
        from pyspark.ml import wrapper

        from smallaxe.training import catboost as catboost_module

        calls = []

        @_synapseml_style
        def synapse_make_pair(self, param, value):
            calls.append(("synapse_make_pair", value))
            return ("synapse", value)

        @_synapseml_style
        def synapse_from_java(java_stage):
            calls.append(("synapse_from_java", java_stage._class_name))
            return "synapse-stage"

        monkeypatch.setattr(wrapper.JavaParams, "_make_java_param_pair", synapse_make_pair)
        monkeypatch.setattr(wrapper.JavaParams, "_from_java", staticmethod(synapse_from_java))
        monkeypatch.setattr(wrapper, "_py2java", lambda sc, value: ("catboost_py2java", value))
        monkeypatch.setattr(
            catboost_module, "_catboost_from_java", lambda: lambda stage: "catboost-stage"
        )
        return wrapper.JavaParams, calls

    def test_timedelta_params_use_catboost_converter(self, synapseml_patched):
        import datetime

        from smallaxe.training.catboost import _ensure_synapseml_compat

        java_params, calls = synapseml_patched
        _ensure_synapseml_compat()
        timeout = datetime.timedelta(seconds=60)

        result = java_params._make_java_param_pair(_FakeParams(), "connectTimeout", timeout)

        assert result == ("java", ("catboost_py2java", timeout))
        assert calls == []

    def test_other_params_still_use_synapseml_converter(self, synapseml_patched):
        from smallaxe.training.catboost import _ensure_synapseml_compat

        java_params, calls = synapseml_patched
        _ensure_synapseml_compat()

        assert java_params._make_java_param_pair(_FakeParams(), "depth", 6) == ("synapse", 6)
        assert calls == [("synapse_make_pair", 6)]

    def test_catboost_stages_load_with_catboost_loader(self, synapseml_patched):
        from smallaxe.training.catboost import _ensure_synapseml_compat

        java_params, calls = synapseml_patched
        _ensure_synapseml_compat()

        catboost_stage = _FakeStage("ai.catboost.spark.CatBoostRegressionModel")
        lightgbm_stage = _FakeStage(
            "com.microsoft.azure.synapse.ml.lightgbm.LightGBMRegressionModel"
        )
        assert java_params._from_java(catboost_stage) == "catboost-stage"
        assert java_params._from_java(lightgbm_stage) == "synapse-stage"
        assert calls == [("synapse_from_java", lightgbm_stage._class_name)]

    def test_is_idempotent(self, synapseml_patched):
        from smallaxe.training.catboost import _ensure_synapseml_compat

        java_params, _ = synapseml_patched
        _ensure_synapseml_compat()
        make_pair, from_java = java_params._make_java_param_pair, java_params._from_java
        _ensure_synapseml_compat()

        assert java_params._make_java_param_pair is make_pair
        assert java_params._from_java is from_java

    def test_noop_without_synapseml(self):
        from pyspark.ml.wrapper import JavaParams

        from smallaxe.training.catboost import _ensure_synapseml_compat

        make_pair, from_java = JavaParams._make_java_param_pair, JavaParams._from_java
        _ensure_synapseml_compat()

        assert JavaParams._make_java_param_pair is make_pair
        assert JavaParams._from_java is from_java
