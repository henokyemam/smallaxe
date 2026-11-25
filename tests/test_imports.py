"""Importing smallaxe imports no optional package; JVM packages load in a fixed order.

``xgboost.spark`` pulls in scikit-learn, and SynapseML and catboost_spark patch
``pyspark.ml.wrapper`` process-wide, so none of them may load before an algorithm
that needs them is used. When one of the JVM packages is first needed, catboost_spark
is imported before ``synapse.ml.lightgbm``, because the CatBoost compatibility shim
assumes SynapseML's patches were applied last. The real JVM packages cannot be
installed here, so the order is checked with fake modules served by an import hook.
"""

import importlib.abc
import importlib.machinery
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

from smallaxe.exceptions import DependencyError
from smallaxe.training import algorithm
from smallaxe.training.algorithm import ALL_TASKS, Algorithm, Dependency, Param
from tests.fakes import FakeEstimator, FakeModel

ROOT = Path(__file__).resolve().parent.parent
OPTIONAL_PACKAGES = ["xgboost", "sklearn", "synapse", "catboost_spark", "lightgbm", "catboost"]


def test_importing_smallaxe_imports_no_optional_package():
    code = (
        "import json, sys\n"
        "import smallaxe, smallaxe.training, smallaxe.search, smallaxe.pipeline, "
        "smallaxe.preprocessing, smallaxe.metrics\n"
        f"packages = {OPTIONAL_PACKAGES!r}\n"
        "loaded = sorted(m for m in sys.modules if m.split('.')[0] in packages)\n"
        "print(json.dumps(loaded))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=str(ROOT), capture_output=True, text=True, check=True
    )
    assert json.loads(result.stdout.strip().splitlines()[-1]) == []


@pytest.mark.parametrize(
    "module, flag, name",
    [
        ("smallaxe.training.xgboost", "XGBOOST_AVAILABLE", "xgboost"),
        ("smallaxe.training.lightgbm", "LIGHTGBM_AVAILABLE", "lightgbm"),
        ("smallaxe.training.catboost", "CATBOOST_AVAILABLE", "catboost"),
    ],
)
def test_availability_flags_are_importable_and_current(monkeypatch, module, flag, name):
    namespace = {}
    exec(f"from {module} import {flag}", namespace)
    assert namespace[flag] is algorithm.get(name).available()

    record = algorithm.get(name)
    estimator_module = next(iter(record.estimators.values())).rpartition(".")[0]
    monkeypatch.setitem(sys.modules, estimator_module, None)
    assert getattr(importlib.import_module(module), flag) is False


def test_unknown_module_attribute_still_raises():
    from smallaxe.training import xgboost

    with pytest.raises(AttributeError, match="NOT_A_NAME"):
        _ = xgboost.NOT_A_NAME


# --- First-use import order of the JVM packages ------------------------------------

_FAKE_PACKAGES = {
    "catboost_spark": {
        "CatBoostRegressor": FakeEstimator,
        "CatBoostClassifier": FakeEstimator,
        "CatBoostRegressionModel": FakeModel,
        "CatBoostClassificationModel": FakeModel,
    },
    "synapse": {},
    "synapse.ml": {},
    "synapse.ml.lightgbm": {
        "LightGBMRegressor": FakeEstimator,
        "LightGBMClassifier": FakeEstimator,
        "LightGBMRegressionModel": FakeModel,
        "LightGBMClassificationModel": FakeModel,
    },
}


class _RecordingFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Serves fake JVM packages and records the order in which they are executed."""

    def __init__(self, available):
        self.available = set(available)
        self.executed = []

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in _FAKE_PACKAGES:
            return None
        root = fullname.split(".")[0]
        if root not in self.available and fullname not in self.available:
            self.executed.append(f"missing:{fullname}")
            return None
        is_package = fullname in ("synapse", "synapse.ml")
        return importlib.machinery.ModuleSpec(fullname, self, is_package=is_package)

    def create_module(self, spec):
        return types.ModuleType(spec.name)

    def exec_module(self, module):
        if module.__name__ in ("synapse", "synapse.ml"):
            module.__path__ = []
        for attr, value in _FAKE_PACKAGES[module.__name__].items():
            setattr(module, attr, value)
        self.executed.append(module.__name__)


@pytest.fixture
def jvm_imports(monkeypatch):
    """A clean slate: no JVM package imported, and the hook first on the meta path."""

    def install(available=("catboost_spark", "synapse")):
        for name in _FAKE_PACKAGES:
            # setitem first so teardown removes whatever the test imported.
            monkeypatch.setitem(sys.modules, name, None)
            monkeypatch.delitem(sys.modules, name)
        finder = _RecordingFinder(available)
        monkeypatch.setattr(sys, "meta_path", [finder] + sys.meta_path)
        return finder

    return install


def _jvm_order(finder):
    return [name for name in finder.executed if name in ("catboost_spark", "synapse.ml.lightgbm")]


@pytest.mark.parametrize("first", ["lightgbm", "catboost"])
def test_catboost_spark_is_imported_before_synapseml(jvm_imports, first):
    finder = jvm_imports()

    assert algorithm.get(first).available()

    assert _jvm_order(finder) == ["catboost_spark", "synapse.ml.lightgbm"]


@pytest.mark.parametrize("first", ["lightgbm", "catboost"])
def test_order_holds_when_resolving_a_class_first(jvm_imports, first):
    finder = jvm_imports()

    estimator = algorithm.get(first).estimator_class("binary")

    assert estimator is FakeEstimator
    assert _jvm_order(finder) == ["catboost_spark", "synapse.ml.lightgbm"]


def test_a_missing_jvm_package_does_not_block_the_other(jvm_imports):
    finder = jvm_imports(available=("synapse",))

    assert algorithm.get("lightgbm").available()
    assert not algorithm.get("catboost").available()
    assert "missing:catboost_spark" in finder.executed
    assert _jvm_order(finder) == ["synapse.ml.lightgbm"]


def test_other_algorithms_do_not_import_jvm_packages(jvm_imports):
    finder = jvm_imports()

    algorithm.get("random_forest").available()

    assert finder.executed == []


# --- require() and broken installs -------------------------------------------------


def test_require_reports_a_plain_missing_package_with_the_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "xgboost.spark", None)
    with pytest.raises(DependencyError, match=r"xgboost is not installed.*smallaxe\[xgboost\]"):
        algorithm.get("xgboost").require()


class _BrokenFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Finds ``brokenpkg`` but fails to execute it, like a package missing its own deps."""

    def __init__(self, error):
        self.error = error

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "brokenpkg":
            return importlib.machinery.ModuleSpec(fullname, self)
        return None

    def create_module(self, spec):
        return types.ModuleType(spec.name)

    def exec_module(self, module):
        raise self.error


def _broken_record():
    return Algorithm(
        name="broken",
        params=(Param("n_estimators", "trees", 10),),
        estimators={task: "brokenpkg.Estimator" for task in ALL_TASKS},
        models={task: "brokenpkg.Model" for task in ALL_TASKS},
        dependency=Dependency("brokenpkg", "pip install brokenpkg"),
    )


@pytest.mark.parametrize(
    "error",
    [
        ModuleNotFoundError("No module named 'pyarrow'", name="pyarrow"),
        ImportError("cannot import name 'thing' from 'pyspark.ml'"),
    ],
    ids=["missing-transitive-dependency", "import-error"],
)
def test_require_includes_the_error_of_a_broken_install(monkeypatch, error):
    monkeypatch.delitem(sys.modules, "brokenpkg", raising=False)
    monkeypatch.setattr(sys, "meta_path", [_BrokenFinder(error)] + sys.meta_path)
    record = _broken_record()

    assert not record.available()
    with pytest.raises(DependencyError, match="brokenpkg could not be imported") as info:
        record.require()
    assert str(error) in str(info.value)
    assert "pip install brokenpkg" in str(info.value)
    assert info.value.package == "brokenpkg"
    assert info.value.install_command == "pip install brokenpkg"
    assert info.value.__cause__ is error


def test_require_keeps_the_short_message_when_the_package_itself_is_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "brokenpkg", None)
    with pytest.raises(DependencyError) as info:
        _broken_record().require()
    assert str(info.value) == "brokenpkg is not installed. Install with: pip install brokenpkg"
