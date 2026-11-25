"""The private deprecation helpers in smallaxe._deprecation."""

import inspect
import types
import warnings

import pytest

from smallaxe import _deprecation
from smallaxe.exceptions import ValidationError


@_deprecation.df_keyword
def _function(dataframe, label_col="label"):
    """Doc of _function."""
    return dataframe, label_col


class _Thing:
    @_deprecation.df_keyword
    def method(self, dataframe, flag=False):
        return self, dataframe, flag

    @classmethod
    @_deprecation.df_keyword
    def klass(cls, dataframe):
        return cls, dataframe

    @_deprecation.df_keyword
    @staticmethod
    def static(dataframe):
        return dataframe


@_deprecation.keyword_only("eps")
def _metric(dataframe, label_col="label", *, eps=1.0):
    return dataframe, label_col, eps


def _one_deprecation(record):
    deprecations = [w for w in record if issubclass(w.category, DeprecationWarning)]
    assert len(deprecations) == 1, [str(w.message) for w in deprecations]
    warning = deprecations[0]
    assert "removed in smallaxe 1.0" in str(warning.message)
    assert warning.filename == __file__
    return str(warning.message)


class TestDfKeyword:
    def test_keeps_name_doc_and_signature(self):
        assert _function.__name__ == "_function"
        assert _function.__doc__ == "Doc of _function."
        assert list(inspect.signature(_function).parameters) == ["dataframe", "label_col"]
        assert list(inspect.signature(_Thing.method).parameters) == ["self", "dataframe", "flag"]
        assert list(inspect.signature(_Thing.klass).parameters) == ["dataframe"]
        assert list(inspect.signature(_Thing.static).parameters) == ["dataframe"]

    @pytest.mark.parametrize(
        "call",
        [
            lambda: _function("x", label_col="y"),
            lambda: _function(dataframe="x", label_col="y"),
        ],
    )
    def test_new_spelling_does_not_warn(self, call):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert call() == ("x", "y")

    def test_df_keyword_warns_once_at_the_caller(self):
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            assert _function(df="x", label_col="y") == ("x", "y")
        message = _one_deprecation(record)
        assert "df=" in message and "dataframe=" in message and "_function" in message

    def test_method_classmethod_staticmethod(self):
        thing = _Thing()
        for call, expected in (
            (lambda: thing.method(df="x", flag=True), (thing, "x", True)),
            (lambda: _Thing.klass(df="x"), (_Thing, "x")),
            (lambda: _Thing.static(df="x"), "x"),
        ):
            with warnings.catch_warnings(record=True) as record:
                warnings.simplefilter("always")
                assert call() == expected
            _one_deprecation(record)

    @pytest.mark.parametrize(
        "call",
        [lambda: _function("x", df="y"), lambda: _function(dataframe="x", df="y")],
    )
    def test_df_and_dataframe_raise(self, call):
        with pytest.raises(ValidationError, match="both df= and dataframe"):
            call()

    def test_neither_is_a_missing_argument(self):
        with pytest.raises(TypeError, match="dataframe"):
            _function()


class TestKeywordOnly:
    def test_keyword_does_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert _metric("x", "y", eps=2.0) == ("x", "y", 2.0)
        assert str(inspect.signature(_metric)) == "(dataframe, label_col='label', *, eps=1.0)"

    def test_positional_warns_once(self):
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            assert _metric("x", "y", 2.0) == ("x", "y", 2.0)
        assert "eps=" in _one_deprecation(record)

    def test_too_many_or_duplicate_positional_values(self):
        with pytest.raises(TypeError, match="at most 3 positional"):
            _metric("x", "y", 2.0, 3.0)
        with pytest.raises(TypeError, match="multiple values"):
            _metric("x", "y", 2.0, eps=3.0)


class TestModuleGetattr:
    @pytest.fixture
    def module(self, monkeypatch):
        module = types.ModuleType("smallaxe_fake_module")
        module.__path__ = []  # a package, so ``from ... import`` goes through _handle_fromlist
        module.__getattr__ = _deprecation.module_getattr(
            "smallaxe_fake_module", {"OLD": ("Use NEW instead.", _deprecation.renamed(42))}
        )
        module.__dir__ = _deprecation.module_dir(vars(module), ["OLD"])
        monkeypatch.setitem(__import__("sys").modules, "smallaxe_fake_module", module)
        return module

    def test_old_name_warns_once_and_returns_value(self, module):
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            assert module.OLD == 42
        assert "Use NEW instead" in _one_deprecation(record)

    def test_from_import_warns_once(self, module):
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            from smallaxe_fake_module import OLD  # noqa: F401
        _one_deprecation(record)

    def test_unknown_name_and_dir(self, module):
        with pytest.raises(AttributeError, match="no attribute 'MISSING'"):
            module.MISSING  # noqa: B018
        assert "OLD" in dir(module)
