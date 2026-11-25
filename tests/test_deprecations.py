"""The 0.11 deprecations: every old spelling works and warns once; the new ones are silent.

Each old spelling must keep its 0.10 result, emit exactly one ``DeprecationWarning``
that names its replacement and says it is removed in smallaxe 1.0, and attribute it
to the caller's line (this file).
"""

import warnings
from contextlib import contextmanager
from typing import Iterator, List

import pytest

from smallaxe.exceptions import ValidationError
from smallaxe.training import Regressor


@contextmanager
def warns_once(*fragments: str) -> Iterator[List[warnings.WarningMessage]]:
    """Expect exactly one DeprecationWarning, from this file, naming each fragment."""
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        yield record
    deprecations = [w for w in record if issubclass(w.category, DeprecationWarning)]
    messages = [str(w.message) for w in deprecations]
    assert len(deprecations) == 1, messages
    assert "removed in smallaxe 1.0" in messages[0]
    for fragment in fragments:
        assert fragment in messages[0], (fragment, messages[0])
    assert deprecations[0].filename == __file__, deprecations[0].filename


@contextmanager
def no_deprecation() -> Iterator[None]:
    """Fail on any DeprecationWarning raised inside the block."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        yield


# --- 1. set_param(dict) -> set_params(**params) ---------------------------------------


class TestSetParams:
    def test_set_params_returns_the_model_and_does_not_warn(self):
        model = Regressor("random_forest")
        with no_deprecation():
            assert model.set_params(n_estimators=7, max_depth=3) is model
        assert model.get_params()["n_estimators"] == 7
        assert model.get_param("max_depth") == 3

    def test_set_params_validates_like_set_param(self):
        model = Regressor("random_forest")
        with pytest.raises(ValidationError, match="Invalid parameter 'nope'"):
            model.set_params(nope=1)
        with pytest.raises(ValidationError, match="must be of type int"):
            model.set_params(n_estimators="many")

    def test_set_param_warns_once_and_delegates(self):
        model = Regressor("random_forest")
        with warns_once("set_param", "set_params(**params)"):
            assert model.set_param({"n_estimators": 9}) is model
        assert model.get_param("n_estimators") == 9

    def test_set_param_keeps_its_errors(self):
        model = Regressor("random_forest")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with pytest.raises(ValidationError, match="Invalid parameter '1'"):
                model.set_param({1: 1})
