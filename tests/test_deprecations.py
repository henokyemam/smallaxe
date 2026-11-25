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


# --- 2. The DataFrame argument is called ``dataframe`` ---------------------------------


def _public_callables():
    """(qualified name, callable) for every public function and public method in smallaxe."""
    import importlib
    import inspect
    import pkgutil

    import smallaxe

    found = []
    modules = [smallaxe] + [
        importlib.import_module(info.name)
        for info in pkgutil.walk_packages(smallaxe.__path__, "smallaxe.")
        if not any(part.startswith("_") for part in info.name.split("."))
    ]
    for module in modules:
        for name, obj in vars(module).items():
            if name.startswith("_") or getattr(obj, "__module__", None) != module.__name__:
                continue
            if inspect.isfunction(obj):
                found.append((f"{module.__name__}.{name}", obj))
            elif inspect.isclass(obj):
                for attr, member in vars(obj).items():
                    if attr.startswith("_") and attr != "__init__":
                        continue
                    if isinstance(member, (classmethod, staticmethod)):
                        member = member.__func__
                    if inspect.isfunction(member):
                        found.append((f"{module.__name__}.{name}.{attr}", member))
    return found


def test_no_public_callable_has_a_df_parameter():
    import inspect

    callables = _public_callables()
    offenders = [name for name, func in callables if "df" in inspect.signature(func).parameters]
    assert offenders == []
    renamed = {
        name for name, func in callables if "dataframe" in inspect.signature(func).parameters
    }
    for expected in (
        "smallaxe.training.base.BaseModel.fit",
        "smallaxe.training.base.BaseModel.predict",
        "smallaxe.training.base.BaseModel.evaluate",
        "smallaxe.training.base.BaseClassifier.predict_proba",
        "smallaxe.pipeline.pipeline.Pipeline.fit",
        "smallaxe.pipeline.pipeline.Pipeline.transform",
        "smallaxe.pipeline.pipeline.Pipeline.fit_transform",
        "smallaxe.pipeline.pipeline.Pipeline.predict",
        "smallaxe.pipeline.pipeline.Pipeline.predict_proba",
        "smallaxe.pipeline.pipeline.Pipeline.evaluate",
        "smallaxe.preprocessing.imputer.Imputer.fit",
        "smallaxe.preprocessing.scaler.Scaler.transform",
        "smallaxe.preprocessing.encoder.Encoder.fit_transform",
        "smallaxe.metrics.regression.mse",
        "smallaxe.metrics.classification.log_loss",
        "smallaxe.search.optimize.run",
    ):
        assert expected in renamed


@pytest.fixture
def frames(spark_session):
    regression = spark_session.createDataFrame(
        [(float(i), float(i % 5), "a" if i % 3 else "b", 2.0 * i) for i in range(40)],
        ["x", "z", "c", "y"],
    )
    classification = spark_session.createDataFrame(
        [(float(i), float(i % 5), i % 2) for i in range(40)], ["x", "z", "label"]
    )
    scored = spark_session.createDataFrame(
        [(1, 1.0, 0.9), (0, 0.0, 0.2), (1, 0.0, 0.4), (0, 1.0, 0.7)],
        ["label", "predict_label", "probability"],
    )
    return {"regression": regression, "classification": classification, "scored": scored}


def _calls(frames):
    """name -> (callable taking the DataFrame first, the DataFrame, other kwargs)."""
    from smallaxe import metrics
    from smallaxe.pipeline import Pipeline
    from smallaxe.preprocessing import Encoder, Imputer, Scaler
    from smallaxe.training import Classifier

    reg, cls, scored = frames["regression"], frames["classification"], frames["scored"]
    model = (
        Regressor("random_forest")
        .set_params(n_estimators=3, seed=1)
        .fit(reg, "y", feature_cols=["x"])
    )
    clf = Classifier("random_forest").set_params(n_estimators=3, seed=1).fit(cls, "label")
    pipeline = Pipeline(
        [("scaler", Scaler()), ("model", Classifier("random_forest").set_params(seed=1))]
    )
    pipeline.fit(cls, label_col="label", numerical_cols=["x", "z"])
    fitted_scaler = Scaler().fit(reg, numerical_cols=["x"])
    return {
        "Model.fit": (
            Regressor("random_forest").set_params(seed=1).fit,
            reg,
            {"label_col": "y", "feature_cols": ["x"]},
        ),
        "Model.predict": (model.predict, reg, {}),
        "Model.evaluate": (model.evaluate, reg, {}),
        "Classifier.predict_proba": (clf.predict_proba, cls, {}),
        "Pipeline.fit": (
            Pipeline([("scaler", Scaler())]).fit,
            reg,
            {"numerical_cols": ["x"]},
        ),
        "Pipeline.fit_transform": (
            Pipeline([("scaler", Scaler())]).fit_transform,
            reg,
            {"numerical_cols": ["x"]},
        ),
        "Pipeline.transform": (pipeline.transform, cls, {}),
        "Pipeline.predict": (pipeline.predict, cls, {}),
        "Pipeline.predict_proba": (pipeline.predict_proba, cls, {}),
        "Pipeline.evaluate": (pipeline.evaluate, cls, {}),
        "Imputer.fit": (Imputer(numerical_strategy="mean").fit, reg, {"numerical_cols": ["x"]}),
        "Imputer.fit_transform": (
            Imputer(numerical_strategy="mean").fit_transform,
            reg,
            {"numerical_cols": ["x"]},
        ),
        "Scaler.fit": (Scaler().fit, reg, {"numerical_cols": ["x"]}),
        "Scaler.transform": (fitted_scaler.transform, reg, {}),
        "Encoder.fit": (Encoder().fit, reg, {"categorical_cols": ["c"]}),
        "Encoder.fit_transform": (Encoder().fit_transform, reg, {"categorical_cols": ["c"]}),
        "metrics.mse": (metrics.mse, scored, {}),
        "metrics.accuracy": (metrics.accuracy, scored, {}),
        "metrics.multiclass_scores": (metrics.multiclass_scores, scored, {}),
        "metrics.auc_roc": (metrics.auc_roc, scored, {}),
        "metrics.log_loss": (metrics.log_loss, scored, {}),
    }


def _result(value):
    """Something comparable for a call's result."""
    from pyspark.sql import DataFrame

    if isinstance(value, DataFrame):
        return value.columns, value.count()
    if hasattr(value, "_is_fitted"):
        return type(value).__name__, value._is_fitted
    return value


def test_dataframe_rename_contract(frames):
    for name, (func, dataframe, kwargs) in _calls(frames).items():
        with no_deprecation():
            positional = _result(func(dataframe, **kwargs))
            assert _result(func(dataframe=dataframe, **kwargs)) == positional, name
        with warns_once("df=", "dataframe="):
            assert _result(func(df=dataframe, **kwargs)) == positional, name
        with pytest.raises(ValidationError, match="both df= and dataframe"):
            func(dataframe, df=dataframe, **kwargs)
        with pytest.raises(ValidationError, match="both df= and dataframe"):
            func(dataframe=dataframe, df=dataframe, **kwargs)
        with pytest.raises(TypeError, match="dataframe"):
            func(**kwargs)


# --- 3. fit(stratified=...) and optimize.run(stratified=...) have no effect -----------


class TestStratified:
    @pytest.mark.parametrize("validation", ["train_test", "kfold"])
    @pytest.mark.parametrize("kind", ["regression", "classification"])
    def test_stratified_selects_nothing(self, frames, kind, validation):
        """Splits and scores are the same for stratified None, True and False."""
        from smallaxe.training import Classifier

        dataframe = frames[kind]
        label = "y" if kind == "regression" else "label"
        make = Regressor if kind == "regression" else Classifier
        results = []
        for stratified in (None, True, False):
            model = make("random_forest").set_params(n_estimators=3, seed=3)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                model.fit(
                    dataframe,
                    label,
                    feature_cols=["x", "z"],
                    validation=validation,
                    n_folds=3,
                    stratified=stratified,
                )
            results.append((model.validation_scores, model.metadata["split_seed"]))
        assert results[1] == results[0]
        assert results[2] == results[0]

    def test_default_does_not_warn(self, frames):
        with no_deprecation():
            Regressor("random_forest").set_params(n_estimators=2).fit(
                frames["regression"], "y", feature_cols=["x"], validation="train_test"
            )
            Regressor("random_forest").set_params(n_estimators=2).fit(
                frames["regression"], "y", feature_cols=["x"], stratified=None
            )

    @pytest.mark.parametrize("value", [True, False])
    def test_explicit_value_warns(self, frames, value):
        with warns_once("stratified", "no effect"):
            Regressor("random_forest").set_params(n_estimators=2).fit(
                frames["regression"], "y", feature_cols=["x"], stratified=value
            )

    def test_optimize_run_warns_once(self, frames):
        pytest.importorskip("hyperopt")
        from smallaxe.search import optimize

        with warns_once("stratified", "no effect"):
            optimize.run(
                Regressor("random_forest").set_params(n_estimators=2, seed=1),
                frames["regression"],
                label_col="y",
                param_space={"max_depth": [2, 3]},
                feature_cols=["x"],
                max_evals=2,
                stratified=True,
                seed=1,
                verbose=False,
            )


# --- 4. datasets.dataset_info returns a dict ------------------------------------------


class TestDatasetInfo:
    @pytest.mark.parametrize("name", ["regression", "classification"])
    def test_returns_dict_and_prints_with_a_warning(self, capsys, name):
        from smallaxe.datasets import dataset_info

        with warns_once("dataset_info", "verbose="):
            info = dataset_info(name)
        printed = capsys.readouterr().out
        assert "Columns:" in printed and info["label"] in printed
        assert info["name"] == name
        assert set(info) >= {"title", "columns", "numerical_features", "label", "loader"}

    @pytest.mark.parametrize("verbose", [True, False])
    def test_verbose_does_not_warn(self, capsys, verbose):
        from smallaxe.datasets import dataset_info

        with no_deprecation():
            info = dataset_info("regression", verbose=verbose)
        assert info["label"] == "price"
        assert info["numerical_features"] == ["bedrooms", "bathrooms", "sqft", "age"]
        assert info["categorical_features"] == ["location", "condition"]
        assert info["loader"] == "load_sample_regression"
        assert bool(capsys.readouterr().out) is verbose

    def test_unknown_name_is_a_validation_error_and_still_a_value_error(self):
        from smallaxe.datasets import dataset_info

        for caught in (ValidationError, ValueError):
            with pytest.raises(caught, match="Unknown dataset: 'nope'"):
                dataset_info("nope", verbose=False)
