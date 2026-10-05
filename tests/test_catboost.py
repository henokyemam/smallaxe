"""CatBoost specifics: the SynapseML JavaParams compatibility shim (no JVM packages needed).

Translation, availability, and the temporary training directory are covered by
``test_algorithm.py``; the end-to-end contract by ``test_model_contract.py``.
"""

import datetime

import pytest

from smallaxe.training import catboost as catboost_module
from smallaxe.training.catboost import _ensure_synapseml_compat


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
        java_params, calls = synapseml_patched
        _ensure_synapseml_compat()
        timeout = datetime.timedelta(seconds=60)

        result = java_params._make_java_param_pair(_FakeParams(), "connectTimeout", timeout)

        assert result == ("java", ("catboost_py2java", timeout))
        assert calls == []

    def test_other_params_still_use_synapseml_converter(self, synapseml_patched):
        java_params, calls = synapseml_patched
        _ensure_synapseml_compat()

        assert java_params._make_java_param_pair(_FakeParams(), "depth", 6) == ("synapse", 6)
        assert calls == [("synapse_make_pair", 6)]

    def test_catboost_stages_load_with_catboost_loader(self, synapseml_patched):
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
        java_params, _ = synapseml_patched
        _ensure_synapseml_compat()
        make_pair, from_java = java_params._make_java_param_pair, java_params._from_java
        _ensure_synapseml_compat()

        assert java_params._make_java_param_pair is make_pair
        assert java_params._from_java is from_java

    def test_noop_without_synapseml(self):
        from pyspark.ml.wrapper import JavaParams

        make_pair, from_java = JavaParams._make_java_param_pair, JavaParams._from_java
        _ensure_synapseml_compat()

        assert JavaParams._make_java_param_pair is make_pair
        assert JavaParams._from_java is from_java

    def test_shim_is_the_record_prepare_hook(self):
        assert catboost_module.ALGORITHM.prepare is _ensure_synapseml_compat
