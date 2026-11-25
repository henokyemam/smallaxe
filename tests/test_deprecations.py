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


# --- 5. Regressors / Classifiers -> the regressors / classifiers module functions ------

_CONSTRUCTORS = ["random_forest", "xgboost", "lightgbm", "catboost"]


def _installed(name):
    from smallaxe.training import algorithm

    return algorithm.get(name).available()


class TestFactoryFunctions:
    def test_module_functions_and_all(self):
        from smallaxe.training import classifiers, regressors

        expected = _CONSTRUCTORS + ["load", "algorithms", "available_models"]
        for module in (regressors, classifiers):
            assert module.__all__ == expected
            assert all(callable(getattr(module, name)) for name in expected)
            assert not hasattr(module, "list_models")

    def test_constructors_do_not_warn(self):
        from smallaxe.training import Classifier, classifiers, regressors

        with no_deprecation():
            model = regressors.random_forest(max_depth=3)
            clf = classifiers.random_forest(task="multiclass", max_depth=3)
            assert regressors.algorithms() == _CONSTRUCTORS
            assert classifiers.algorithms() == _CONSTRUCTORS
            reg_report = regressors.available_models()
            cls_report = classifiers.available_models()
        assert isinstance(model, Regressor) and model.get_param("max_depth") == 3
        assert isinstance(clf, Classifier) and clf.task == "multiclass"
        for report in (reg_report, cls_report):
            assert list(report) == _CONSTRUCTORS
            assert report["random_forest"] == {
                "available": True,
                "dependency": None,
                "install_hint": None,
            }

    def test_load_does_not_warn(self, tmp_path, frames):
        from smallaxe.training import classifiers, regressors

        path = str(tmp_path / "model")
        regressors.random_forest(n_estimators=3, seed=1).fit(
            frames["regression"], "y", feature_cols=["x"]
        ).save(path)
        with no_deprecation():
            loaded = regressors.load(path)
        assert loaded.algorithm.name == "random_forest"
        with pytest.raises(ValidationError, match="not a supported classifier.*regressors.load"):
            classifiers.load(path)


class TestDeprecatedFactoryClasses:
    def test_importing_the_classes_does_not_warn(self):
        with no_deprecation():
            from smallaxe.training import Classifiers, Regressors  # noqa: F401

    @pytest.mark.parametrize("name", _CONSTRUCTORS)
    def test_regressors_constructors(self, name):
        from smallaxe.training import Regressors, regressors

        if not _installed(name):
            pytest.skip(f"{name} is not installed")
        with warns_once(f"Regressors.{name}()", f"regressors.{name}()"):
            model = getattr(Regressors, name)(max_depth=4)
        expected = getattr(regressors, name)(max_depth=4)
        assert type(model) is type(expected)
        assert model.algorithm is expected.algorithm
        assert model.get_params() == expected.get_params()

    @pytest.mark.parametrize("name", _CONSTRUCTORS)
    def test_classifiers_constructors(self, name):
        from smallaxe.training import Classifiers, classifiers

        if not _installed(name):
            pytest.skip(f"{name} is not installed")
        with warns_once(f"Classifiers.{name}()", f"classifiers.{name}()"):
            model = getattr(Classifiers, name)(task="multiclass", max_depth=4)
        expected = getattr(classifiers, name)(task="multiclass", max_depth=4)
        assert model.task == "multiclass"
        assert model.get_params() == expected.get_params()

    @pytest.mark.parametrize("name", ["lightgbm", "catboost"])
    def test_missing_dependency_still_raises(self, monkeypatch, name):
        import sys

        from smallaxe.exceptions import DependencyError
        from smallaxe.training import Classifiers, Regressors, algorithm

        if _installed(name):
            pytest.skip(f"{name} is installed")
        monkeypatch.setitem(sys.modules, algorithm.get(name).module, None)
        for factory in (Regressors, Classifiers):
            with warns_once(f"{factory.__name__}.{name}()"):
                with pytest.raises(DependencyError):
                    getattr(factory, name)()

    def test_load(self, tmp_path, frames):
        from smallaxe.training import Classifiers, Regressors, classifiers

        path = str(tmp_path / "model")
        model = Regressor("random_forest").set_params(n_estimators=3, seed=1)
        model.fit(frames["regression"], "y", feature_cols=["x"]).save(path)
        with warns_once("Regressors.load()", "regressors.load()"):
            loaded = Regressors.load(path)
        assert isinstance(loaded, Regressor)
        assert loaded.get_param("n_estimators") == 3

        clf_path = str(tmp_path / "clf")
        classifiers.random_forest(n_estimators=3, seed=1).fit(
            frames["classification"], "label"
        ).save(clf_path)
        with warns_once("Classifiers.load()", "classifiers.load()"):
            assert Classifiers.load(clf_path).task == "binary"

    def test_list_models(self):
        from smallaxe.training import Classifiers, Regressors

        for factory, suffix in ((Regressors, "Regressor"), (Classifiers, "Classifier")):
            with warns_once(f"{factory.__name__}.list_models()", "algorithms()"):
                names = factory.list_models()
            assert f"RandomForest{suffix}" in names
            assert all(name.endswith(suffix) for name in names)

    def test_available_models_keeps_class_name(self):
        from smallaxe.training import Classifiers, Regressors, classifiers, regressors

        for factory, module, suffix in (
            (Regressors, regressors, "Regressor"),
            (Classifiers, classifiers, "Classifier"),
        ):
            with warns_once(f"{factory.__name__}.available_models()"):
                report = factory.available_models()
            assert list(report) == _CONSTRUCTORS
            assert report["random_forest"] == {
                "class_name": f"RandomForest{suffix}",
                "available": True,
                "dependency": None,
                "install_hint": None,
            }
            assert report["xgboost"]["class_name"] == f"XGBoost{suffix}"
            assert report["catboost"]["class_name"] == f"CatBoost{suffix}"
            assert report["lightgbm"]["class_name"] == f"LightGBM{suffix}"
            assert {
                name: {k: v for k, v in entry.items() if k != "class_name"}
                for name, entry in report.items()
            } == module.available_models()


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


# --- 6. log_loss eps is keyword-only --------------------------------------------------


class TestLogLossEps:
    def test_keyword_eps_does_not_warn(self, frames):
        from smallaxe.metrics import log_loss

        with no_deprecation():
            keyword = log_loss(frames["scored"], "label", "probability", eps=1e-3)
        assert keyword == pytest.approx(log_loss(frames["scored"], eps=1e-3))

    def test_positional_eps_warns_and_gives_the_same_result(self, frames):
        from smallaxe.metrics import log_loss

        with warns_once("eps", "log_loss"):
            positional = log_loss(frames["scored"], "label", "probability", 0.3)
        assert positional == log_loss(frames["scored"], "label", "probability", eps=0.3)
        assert positional != log_loss(frames["scored"], "label", "probability")


# --- 7. Names that become private in 1.0 ----------------------------------------------

# (module, old public name, its private implementation or None, a fragment of the hint)
_PRIVATE = [
    ("smallaxe.training", "BaseModel", None, "smallaxe.training.Model"),
    ("smallaxe.training", "BaseRegressor", None, "smallaxe.training.Regressor"),
    ("smallaxe.training", "BaseClassifier", None, "smallaxe.training.Classifier"),
    ("smallaxe.training.base", "infer_feature_cols", "_infer_feature_cols", "fit()"),
    ("smallaxe.training.base", "warn_ignored_cols", "_warn_ignored_cols", "fit()"),
    ("smallaxe.training.base", "check_training_data", "_check_training_data", "fit()"),
    (
        "smallaxe.training.mixins.validation_mixin",
        "hashable_columns",
        "_hashable_columns",
        "fold_key",
    ),
    ("smallaxe.training.mixins.validation_mixin", "split_hash", "_split_hash", "fold_key"),
    (
        "smallaxe.training.mixins.validation_mixin",
        "resolve_split_seed",
        "_resolve_split_seed",
        "set_seed",
    ),
    ("smallaxe.training.algorithm", "resolve", "_resolve", "get("),
    ("smallaxe.training.algorithm", "estimator_kwargs", "_estimator_kwargs", "set_params"),
    ("smallaxe.training.algorithm", "legacy_class_name", "_legacy_class_name", "get("),
    ("smallaxe.training.algorithm", "package_version", "_package_version", "metadata"),
    ("smallaxe.training.algorithm", "COLUMN_ROLES", "_COLUMN_ROLES", "get("),
    ("smallaxe.training.algorithm", "LEGACY_CLASS_NAMES", "_LEGACY_CLASS_NAMES", "load"),
    ("smallaxe.search.optimize", "HYPEROPT_INSTALL_HINT", "_HYPEROPT_INSTALL_HINT", "hyperopt"),
    ("smallaxe.search.optimize", "MAXIMIZE_METRICS", "_MAXIMIZE_METRICS", "TASK_METRICS"),
    ("smallaxe.search.optimize", "MINIMIZE_METRICS", "_MINIMIZE_METRICS", "TASK_METRICS"),
    ("smallaxe.search.optimize", "VALID_METRICS", "_VALID_METRICS", "TASK_METRICS"),
    ("smallaxe.search.optimize", "HYPEROPT_AVAILABLE", "_HYPEROPT_AVAILABLE", "hyperopt"),
    ("smallaxe.search", "HYPEROPT_AVAILABLE", None, "hyperopt"),
    ("smallaxe.training.catboost", "is_catboost_available", "_is_catboost_available", "get("),
    ("smallaxe.training.catboost", "catboost_install_hint", "_catboost_install_hint", "get("),
    ("smallaxe.training.xgboost", "XGBOOST_AVAILABLE", None, "available_models()"),
    ("smallaxe.training.lightgbm", "LIGHTGBM_AVAILABLE", None, "available_models()"),
    ("smallaxe.training.catboost", "CATBOOST_AVAILABLE", None, "available_models()"),
    ("smallaxe.preprocessing.encoder", "VALID_METHODS", "_VALID_METHODS", "Encoder"),
    ("smallaxe.preprocessing.encoder", "VALID_HANDLE_RARE", "_VALID_HANDLE_RARE", "Encoder"),
    ("smallaxe.preprocessing.scaler", "VALID_METHODS", "_VALID_METHODS", "Scaler"),
]

# The value each old name had in 0.10, by (module, name), where it is not the private one.
_EXPECTED = {
    ("smallaxe.training", "BaseModel"): ("smallaxe.training.base", "BaseModel"),
    ("smallaxe.training", "BaseRegressor"): ("smallaxe.training.base", "BaseRegressor"),
    ("smallaxe.training", "BaseClassifier"): ("smallaxe.training.base", "BaseClassifier"),
    ("smallaxe.search", "HYPEROPT_AVAILABLE"): ("smallaxe.search.optimize", "_HYPEROPT_AVAILABLE"),
}


def _expected_value(module_name, name, private):
    import importlib

    from smallaxe.training import algorithm

    flags = {"XGBOOST_AVAILABLE": "xgboost", "LIGHTGBM_AVAILABLE": "lightgbm"}
    flags["CATBOOST_AVAILABLE"] = "catboost"
    if name in flags:
        return algorithm.get(flags[name]).available()
    source, attr = _EXPECTED.get((module_name, name), (module_name, private))
    return getattr(importlib.import_module(source), attr)


@pytest.mark.parametrize(
    "module_name, name, private, hint", _PRIVATE, ids=[f"{m}.{n}" for m, n, _, _ in _PRIVATE]
)
def test_private_name_keeps_old_spelling(module_name, name, private, hint):
    import importlib

    module = importlib.import_module(module_name)
    if private is not None:
        with no_deprecation():
            getattr(module, private)
    assert name in dir(module)
    with warns_once(f"{module_name}.{name}", hint):
        value = getattr(module, name)
    with warns_once(f"{module_name}.{name}"):
        namespace = {}
        code = compile(f"from {module_name} import {name}", __file__, "exec")
        exec(code, {"__name__": "user_code"}, namespace)
    assert namespace[name] is value or namespace[name] == value
    expected = _expected_value(module_name, name, private)
    if callable(expected) and not isinstance(expected, type):
        assert value.__wrapped__ is expected if hasattr(value, "__wrapped__") else value is expected
    else:
        assert value is expected or value == expected


def test_private_helpers_still_accept_df_by_old_name(frames):
    from smallaxe.training import base
    from smallaxe.training.mixins import validation_mixin

    dataframe = frames["regression"]
    expected = base._infer_feature_cols(dataframe, "y", [])
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        assert base.infer_feature_cols(df=dataframe, label_col="y", exclude_cols=[]) == expected
        assert validation_mixin.hashable_columns(df=dataframe) == ["x", "z", "c", "y"]
    assert len([w for w in record if issubclass(w.category, DeprecationWarning)]) == 4


def test_deprecated_names_are_not_in_all():
    import smallaxe.search
    import smallaxe.training

    assert not {"BaseModel", "BaseRegressor", "BaseClassifier"} & set(smallaxe.training.__all__)
    assert "HYPEROPT_AVAILABLE" not in smallaxe.search.__all__
    for name in ("Model", "Regressor", "Classifier", "Regressors", "Classifiers"):
        assert name in smallaxe.training.__all__
    assert {"regressors", "classifiers"} <= set(smallaxe.training.__all__)


# --- The library never calls its own deprecated spellings -----------------------------


def test_smallaxe_does_not_trigger_its_own_deprecations(spark_session, tmp_path):
    """Import everything, fit, search, run a pipeline and load data with them as errors."""
    import importlib
    import pkgutil

    pytest.importorskip("hyperopt")

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error", message=r"(?s).*removed in smallaxe 1\.0", category=DeprecationWarning
        )
        import smallaxe
        from smallaxe import metrics
        from smallaxe.datasets import load_sample_classification, load_sample_regression
        from smallaxe.pipeline import Pipeline
        from smallaxe.preprocessing import Encoder, Imputer, Scaler
        from smallaxe.search import optimize
        from smallaxe.training import classifiers, regressors

        for info in pkgutil.walk_packages(smallaxe.__path__, "smallaxe."):
            importlib.import_module(info.name)

        houses = load_sample_regression(spark_session, n_rows=150)
        churn = load_sample_classification(spark_session, n_rows=150)

        model = regressors.random_forest(n_estimators=3, seed=1)
        model.fit(houses, "price", exclude_cols=["location", "condition"], validation="kfold")
        model.evaluate(houses)
        model.save(str(tmp_path / "model"))
        regressors.load(str(tmp_path / "model"))
        smallaxe.load_model(str(tmp_path / "model"))

        clf = classifiers.random_forest(n_estimators=3, seed=1)
        clf.fit(churn, "churn", feature_cols=["tenure", "monthly_charges"], validation="train_test")
        clf.predict_proba(churn)
        metrics.accuracy(clf.predict(churn), "churn")
        clf.evaluate(churn)
        metrics.log_loss(churn.selectExpr("churn", "0.5 AS p"), "churn", "p", eps=1e-6)

        result = optimize.run(
            regressors.random_forest(n_estimators=3),
            houses,
            label_col="price",
            param_space={"max_depth": [2, 3]},
            feature_cols=["sqft", "age"],
            max_evals=2,
            validation="kfold",
            n_folds=2,
            seed=1,
            verbose=False,
        )
        result.best_model.predict(houses)

        pipeline = Pipeline(
            [
                (
                    "imputer",
                    Imputer(numerical_strategy="mean", categorical_strategy="most_frequent"),
                ),
                ("scaler", Scaler()),
                ("encoder", Encoder()),
                ("model", classifiers.random_forest(n_estimators=3, seed=1)),
            ]
        )
        pipeline.fit(
            churn,
            label_col="churn",
            numerical_cols=["tenure", "monthly_charges"],
            categorical_cols=["contract"],
        )
        pipeline.save(str(tmp_path / "pipeline"))
        loaded = smallaxe.load_pipeline(str(tmp_path / "pipeline"))
        loaded.predict(churn)
        loaded.predict_proba(churn)
        loaded.evaluate(churn)
        smallaxe.load(str(tmp_path / "pipeline"))
