"""Fake Spark estimators for testing the algorithm seam without JVM packages.

``install_fake_modules`` puts :class:`FakeEstimator` / :class:`FakeModel` at every
dotted path an :class:`~smallaxe.training.algorithm.Algorithm` names, so a record's
translation can be checked on a machine without xgboost, SynapseML, or
catboost_spark installed.
"""

import json
import os
import sys
import types
from typing import Any, Dict, List, Tuple

from smallaxe.training.algorithm import Algorithm


class _FakeWriter:
    def __init__(self, path_holder: "FakeModel") -> None:
        self._model = path_holder

    def overwrite(self) -> "_FakeWriter":
        return self

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "fake_model.json"), "w") as handle:
            json.dump({"fake": True}, handle)
        self._model.saved_to = path


class FakeModel:
    """Stands in for a fitted Spark model, including ``write()`` / ``load()``."""

    def __init__(self, estimator: "FakeEstimator" = None) -> None:
        self.estimator = estimator

    def write(self) -> _FakeWriter:
        return _FakeWriter(self)

    @classmethod
    def load(cls, path: str) -> "FakeModel":
        model = cls()
        model.loaded_from = path
        return model


class FakeEstimator:
    """Records constructor kwargs and every ``set*`` call; ``fit`` returns a FakeModel."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs: Dict[str, Any] = dict(kwargs)
        self.calls: List[Tuple[str, Any]] = []

    def __getattr__(self, name: str) -> Any:
        if name.startswith("set"):

            def setter(value: Any) -> "FakeEstimator":
                self.calls.append((name, value))
                return self

            return setter
        raise AttributeError(name)

    def fit(self, df: Any) -> FakeModel:
        self.fitted_on = df
        return FakeModel(self)


def install_fake_modules(monkeypatch, record: Algorithm) -> Dict[str, types.ModuleType]:
    """Register fake modules for ``record``'s estimator and model paths in ``sys.modules``."""
    modules: Dict[str, types.ModuleType] = {}
    for path in list(record.estimators.values()) + list(record.models.values()):
        module_name, _, attr = path.rpartition(".")
        module = modules.get(module_name)
        if module is None:
            module = types.ModuleType(module_name)
            modules[module_name] = module
            monkeypatch.setitem(sys.modules, module_name, module)
        is_estimator = path in record.estimators.values()
        setattr(module, attr, FakeEstimator if is_estimator else FakeModel)
    return modules


def columns_for(task: str) -> Dict[str, str]:
    """The column-role mapping a Model hands to ``estimator_kwargs`` for ``task``."""
    columns = {"features": "features", "label": "y", "prediction": "prediction"}
    if task != "simple_regression":
        columns["probability"] = "probability"
        columns["raw_prediction"] = "rawPrediction"
    return columns
