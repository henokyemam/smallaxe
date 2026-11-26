"""Everything deprecated in 0.x is gone in 1.0, and the library emits no DeprecationWarning.

Each removed spelling fails the way Python fails for a name or argument that does not
exist: ``ImportError`` for ``from ... import name``, ``AttributeError`` for
``module.name`` / ``obj.name``, ``TypeError`` for an unknown keyword or an extra
positional argument. The CHANGELOG's "Upgrading from 0.x" list maps each one to its
replacement.
"""

import importlib
import inspect
import warnings
from pathlib import Path

import pytest

import smallaxe
from smallaxe.exceptions import ValidationError

# The eight 0.8.x class names: never importable, never in a message or return value.
OLD_CLASS_NAMES = [
    "RandomForestRegressor",
    "RandomForestClassifier",
    "XGBoostRegressor",
    "XGBoostClassifier",
    "LightGBMRegressor",
    "LightGBMClassifier",
    "CatBoostRegressor",
    "CatBoostClassifier",
]

# (module, removed name): every module attribute that warned in 0.11 or earlier.
REMOVED_NAMES = [("smallaxe.training", name) for name in OLD_CLASS_NAMES] + [
    ("smallaxe.training.random_forest", "RandomForestRegressor"),
    ("smallaxe.training.random_forest", "RandomForestClassifier"),
    ("smallaxe.training.xgboost", "XGBoostRegressor"),
    ("smallaxe.training.xgboost", "XGBoostClassifier"),
    ("smallaxe.training.lightgbm", "LightGBMRegressor"),
    ("smallaxe.training.lightgbm", "LightGBMClassifier"),
    ("smallaxe.training.catboost", "CatBoostRegressor"),
    ("smallaxe.training.catboost", "CatBoostClassifier"),
    ("smallaxe", "set_cache_strategy"),
    ("smallaxe", "get_cache_strategy"),
    ("smallaxe", "set_spark_session"),
    ("smallaxe", "get_spark_session"),
    ("smallaxe", "warnings"),
    ("smallaxe.training", "Regressors"),
    ("smallaxe.training", "Classifiers"),
    ("smallaxe.training.regressors", "Regressors"),
    ("smallaxe.training.classifiers", "Classifiers"),
    ("smallaxe.training", "BaseModel"),
    ("smallaxe.training", "BaseRegressor"),
    ("smallaxe.training", "BaseClassifier"),
    ("smallaxe.training.base", "infer_feature_cols"),
    ("smallaxe.training.base", "warn_ignored_cols"),
    ("smallaxe.training.base", "check_training_data"),
    ("smallaxe.training.base", "resolve_split_seed"),
    ("smallaxe.training.mixins.validation_mixin", "hashable_columns"),
    ("smallaxe.training.mixins.validation_mixin", "split_hash"),
    ("smallaxe.training.mixins.validation_mixin", "resolve_split_seed"),
    ("smallaxe.training.algorithm", "resolve"),
    ("smallaxe.training.algorithm", "estimator_kwargs"),
    ("smallaxe.training.algorithm", "legacy_class_name"),
    ("smallaxe.training.algorithm", "package_version"),
    ("smallaxe.training.algorithm", "COLUMN_ROLES"),
    ("smallaxe.training.algorithm", "LEGACY_CLASS_NAMES"),
    ("smallaxe.search", "HYPEROPT_AVAILABLE"),
    ("smallaxe.search.optimize", "HYPEROPT_AVAILABLE"),
    ("smallaxe.search.optimize", "HYPEROPT_INSTALL_HINT"),
    ("smallaxe.search.optimize", "MAXIMIZE_METRICS"),
    ("smallaxe.search.optimize", "MINIMIZE_METRICS"),
    ("smallaxe.search.optimize", "VALID_METRICS"),
    ("smallaxe.search.optimize", "infer_feature_cols"),
    ("smallaxe.search.optimize", "warn_ignored_cols"),
    ("smallaxe.search.optimize", "check_training_data"),
    ("smallaxe.search.optimize", "resolve_split_seed"),
    ("smallaxe.training.xgboost", "XGBOOST_AVAILABLE"),
    ("smallaxe.training.lightgbm", "LIGHTGBM_AVAILABLE"),
    ("smallaxe.training.catboost", "CATBOOST_AVAILABLE"),
    ("smallaxe.training.catboost", "is_catboost_available"),
    ("smallaxe.training.catboost", "catboost_install_hint"),
    ("smallaxe.preprocessing.encoder", "VALID_METHODS"),
    ("smallaxe.preprocessing.encoder", "VALID_HANDLE_RARE"),
    ("smallaxe.preprocessing.scaler", "VALID_METHODS"),
    ("smallaxe.metrics.classification", "MULTICLASS_AVERAGED_METRICS"),
]

REMOVED_MODULES = ["smallaxe._deprecation"]


@pytest.mark.parametrize("module_name, name", REMOVED_NAMES, ids=lambda v: str(v))
def test_removed_name_is_gone(module_name, name):
    module = importlib.import_module(module_name)
    assert not hasattr(module, name)
    with pytest.raises(AttributeError):
        getattr(module, name)
    with pytest.raises(ImportError):
        exec(f"from {module_name} import {name}", {})
    assert name not in getattr(module, "__all__", [])
    assert name not in dir(module)


@pytest.mark.parametrize("module_name", REMOVED_MODULES)
def test_removed_module_is_gone(module_name):
    with pytest.raises(ImportError):
        importlib.import_module(module_name)


def test_set_param_is_gone():
    from smallaxe.training import regressors

    model = regressors.random_forest()
    assert not hasattr(model, "set_param")
    with pytest.raises(AttributeError):
        model.set_param({"n_estimators": 9})


@pytest.fixture(scope="module")
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


def _df_calls(frames):
    """name -> (callable taking the DataFrame first, the DataFrame, other kwargs)."""
    from smallaxe import metrics
    from smallaxe.pipeline import Pipeline
    from smallaxe.preprocessing import Encoder, Imputer, Scaler
    from smallaxe.search import optimize
    from smallaxe.training import classifiers, regressors

    reg, cls, scored = frames["regression"], frames["classification"], frames["scored"]
    model = regressors.random_forest(n_estimators=3, seed=1).fit(reg, "y", feature_cols=["x"])
    clf = classifiers.random_forest(n_estimators=3, seed=1).fit(cls, "label")
    pipeline = Pipeline([("scaler", Scaler()), ("model", classifiers.random_forest(seed=1))])
    pipeline.fit(cls, label_col="label", numerical_cols=["x", "z"])
    fitted_scaler = Scaler().fit(reg, numerical_cols=["x"])
    fitted_imputer = Imputer(numerical_strategy="mean").fit(reg, numerical_cols=["x"])
    fitted_encoder = Encoder().fit(reg, categorical_cols=["c"])
    return {
        "Model.fit": (regressors.random_forest().fit, reg, {"label_col": "y"}),
        "Model.predict": (model.predict, reg, {}),
        "Model.evaluate": (model.evaluate, reg, {}),
        "Classifier.predict_proba": (clf.predict_proba, cls, {}),
        "Pipeline.fit": (Pipeline([("scaler", Scaler())]).fit, reg, {}),
        "Pipeline.fit_transform": (Pipeline([("scaler", Scaler())]).fit_transform, reg, {}),
        "Pipeline.transform": (pipeline.transform, cls, {}),
        "Pipeline.predict": (pipeline.predict, cls, {}),
        "Pipeline.predict_proba": (pipeline.predict_proba, cls, {}),
        "Pipeline.evaluate": (pipeline.evaluate, cls, {}),
        "Imputer.fit": (Imputer(numerical_strategy="mean").fit, reg, {}),
        "Imputer.transform": (fitted_imputer.transform, reg, {}),
        "Imputer.fit_transform": (Imputer(numerical_strategy="mean").fit_transform, reg, {}),
        "Scaler.fit": (Scaler().fit, reg, {"numerical_cols": ["x"]}),
        "Scaler.transform": (fitted_scaler.transform, reg, {}),
        "Scaler.fit_transform": (Scaler().fit_transform, reg, {"numerical_cols": ["x"]}),
        "Encoder.fit": (Encoder().fit, reg, {"categorical_cols": ["c"]}),
        "Encoder.transform": (fitted_encoder.transform, reg, {}),
        "Encoder.fit_transform": (Encoder().fit_transform, reg, {"categorical_cols": ["c"]}),
        "metrics.mse": (metrics.mse, scored, {}),
        "metrics.rmse": (metrics.rmse, scored, {}),
        "metrics.mae": (metrics.mae, scored, {}),
        "metrics.r2": (metrics.r2, scored, {}),
        "metrics.mape": (metrics.mape, scored, {}),
        "metrics.accuracy": (metrics.accuracy, scored, {}),
        "metrics.precision": (metrics.precision, scored, {}),
        "metrics.recall": (metrics.recall, scored, {}),
        "metrics.f1_score": (metrics.f1_score, scored, {}),
        "metrics.multiclass_scores": (metrics.multiclass_scores, scored, {}),
        "metrics.auc_roc": (metrics.auc_roc, scored, {}),
        "metrics.auc_pr": (metrics.auc_pr, scored, {}),
        "metrics.log_loss": (metrics.log_loss, scored, {}),
        "optimize.run": (
            optimize.run,
            reg,
            {"model": regressors.random_forest(), "label_col": "y", "param_space": {}},
        ),
    }


def test_df_keyword_is_a_type_error(frames):
    for name, (func, dataframe, kwargs) in _df_calls(frames).items():
        assert "df" not in inspect.signature(func).parameters, name
        with pytest.raises(TypeError, match="df"):
            func(df=dataframe, **kwargs)


def test_no_public_callable_is_wrapped_or_takes_df():
    """No public function or method of smallaxe keeps a compatibility wrapper or ``df``."""
    from tests.test_public_api import public_callables

    for name, func in public_callables():
        # contextlib.contextmanager (smallaxe.verbosity) is the one wrapper allowed.
        wrapper_file = getattr(getattr(func, "__code__", None), "co_filename", "")
        if not wrapper_file.endswith("contextlib.py"):
            assert not hasattr(func, "__wrapped__"), name
        assert "df" not in inspect.signature(func).parameters, name


@pytest.mark.parametrize("kind", ["regressors", "classifiers"])
def test_stratified_argument_is_a_type_error(frames, kind):
    from smallaxe.search import optimize
    from smallaxe.training import classifiers, regressors

    factory = regressors if kind == "regressors" else classifiers
    frame = frames["regression"] if kind == "regressors" else frames["classification"]
    label = "y" if kind == "regressors" else "label"
    with pytest.raises(TypeError, match="stratified"):
        factory.random_forest().fit(frame, label, validation="kfold", stratified=True)
    with pytest.raises(TypeError, match="stratified"):
        optimize.run(
            factory.random_forest(),
            frame,
            label_col=label,
            param_space={"max_depth": [2]},
            stratified=True,
        )


def test_split_shims_are_gone():
    from smallaxe.training.mixins import ValidationMixin

    for name in ("_stratified_train_test_split", "_simple_kfold_split", "_stratified_kfold_split"):
        assert not hasattr(ValidationMixin, name)
    for method in (ValidationMixin._train_test_split, ValidationMixin._kfold_split):
        parameters = inspect.signature(method).parameters
        assert "stratified" not in parameters and "label_col" not in parameters


def test_positional_eps_is_a_type_error(frames):
    from smallaxe.metrics import log_loss

    with pytest.raises(TypeError):
        log_loss(frames["scored"], "label", "probability", 0.3)
    assert log_loss(frames["scored"], "label", "probability", eps=0.3) != log_loss(
        frames["scored"], "label", "probability"
    )


def test_dataset_info_only_returns_the_dict(capsys):
    from smallaxe.datasets import dataset_info

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        info = dataset_info("regression")
    assert info["label"] == "price"
    assert capsys.readouterr().out == ""


def test_dataset_info_unknown_name_is_only_a_validation_error():
    from smallaxe.datasets import dataset_info

    with pytest.raises(ValidationError) as excinfo:
        dataset_info("nope")
    assert not isinstance(excinfo.value, ValueError)


@pytest.mark.parametrize("kind", ["regressors", "classifiers"])
def test_list_models_and_class_name_are_gone(kind):
    module = importlib.import_module(f"smallaxe.training.{kind}")

    assert not hasattr(module, "list_models")
    report = module.available_models()
    assert set(report) == set(module.algorithms())
    for entry in report.values():
        assert set(entry) == {"available", "dependency", "install_hint"}


def test_no_old_class_name_in_available_models():
    from smallaxe.training import classifiers, regressors

    text = repr(regressors.available_models()) + repr(classifiers.available_models())
    for name in OLD_CLASS_NAMES:
        assert name not in text


def test_old_class_names_only_in_the_private_legacy_map():
    """The old names live only in the map that reads 0.8.x artifacts."""
    from smallaxe.training import algorithm

    assert set(algorithm._LEGACY_CLASS_NAMES) == set(OLD_CLASS_NAMES)
    assert set(algorithm._LEGACY_CLASS_NAMES.values()) == set(algorithm.names())
    assert not hasattr(algorithm, "_legacy_class_name")
    assert not hasattr(algorithm, "_LEGACY_BY_ALGORITHM")


def test_hyperopt_is_imported_normally():
    from smallaxe.search import optimize

    assert not hasattr(optimize, "_HYPEROPT_AVAILABLE")
    assert not hasattr(optimize, "_check_hyperopt_available")


def test_config_has_no_cache_or_session_state():
    _config = importlib.import_module("smallaxe._config")

    assert not hasattr(_config, "VALID_CACHE_STRATEGIES")
    assert not hasattr(_config, "DEFAULT_CACHE_STRATEGY")
    assert not hasattr(_config._config, "_cache_strategy")
    assert not hasattr(_config._config, "_spark_session")


def test_smallaxe_emits_no_deprecation_warning(spark_session, tmp_path):
    """Import everything, fit, search, run a pipeline and load data, deprecations as errors."""
    import pkgutil

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
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
            validation="train_test",
        )
        pipeline.save(str(tmp_path / "pipeline"))
        loaded = smallaxe.load_pipeline(str(tmp_path / "pipeline"))
        loaded.predict(churn)
        loaded.predict_proba(churn)
        loaded.evaluate(churn)
        smallaxe.load(str(tmp_path / "pipeline"))

    # smallaxe attributes its warnings to the caller, so one of its own would point here
    # or inside the package; third-party DeprecationWarnings (PySpark's) point elsewhere.
    package = str(Path(smallaxe.__file__).parent)
    ours = [
        str(w.message)
        for w in record
        if issubclass(w.category, DeprecationWarning)
        and (
            w.filename == __file__ or w.filename.startswith(package) or "smallaxe" in str(w.message)
        )
    ]
    assert ours == []
