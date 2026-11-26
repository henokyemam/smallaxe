"""Public methods raise smallaxe exceptions, never a raw ValueError / KeyError / TypeError,
for user errors: bad arguments, missing columns, unfitted objects, bad paths.

Each case is a call a user could plausibly get wrong, and the smallaxe exception it must
raise. (A wrong *number* of arguments or an unknown keyword is a Python ``TypeError`` by
design, raised before smallaxe runs.)
"""

from pathlib import Path

import pytest
from pyspark.sql import functions as F

import smallaxe
from smallaxe import metrics
from smallaxe.auto import AutomatedTraining
from smallaxe.datasets import dataset_info, load_sample_regression
from smallaxe.exceptions import (
    ColumnNotFoundError,
    ConfigurationError,
    ModelNotFittedError,
    SmallaxeError,
    ValidationError,
)
from smallaxe.pipeline import Pipeline
from smallaxe.preprocessing import Encoder, Imputer, Scaler
from smallaxe.search import optimize
from smallaxe.training import Classifier, Model, Regressor, algorithm, classifiers, regressors


@pytest.fixture(scope="module")
def frame(spark_session):
    rows = [(i, float(i % 7), float(i % 5), "a" if i % 2 else "b", float(2 * i)) for i in range(30)]
    return spark_session.createDataFrame(rows, ["id", "x", "z", "c", "y"])


@pytest.fixture(scope="module")
def fitted(frame):
    return regressors.random_forest(n_estimators=2, seed=1).fit(frame, "y", feature_cols=["x"])


@pytest.fixture(scope="module")
def fitted_classifier(frame):
    binary = frame.withColumn("b", (F.col("id") % 2).cast("double"))
    return classifiers.random_forest(n_estimators=2, seed=1).fit(binary, "b", feature_cols=["x"])


def _cases(frame, fitted, fitted_classifier, tmp_path):
    """name -> (zero-argument call, expected exception)."""
    scaler = Scaler().fit(frame, numerical_cols=["x"])
    binary = frame.withColumn("b", (F.col("id") % 2).cast("double"))
    scored = binary.withColumn("p", F.col("x") / 7.0)
    string_x = frame.withColumn("x", F.lit("a"))
    rf = regressors.random_forest
    return {
        # Global settings
        "set_seed(bool)": (lambda: smallaxe.set_seed(True), ConfigurationError),
        "set_seed(str)": (lambda: smallaxe.set_seed("1"), ConfigurationError),
        "set_verbosity('loud')": (lambda: smallaxe.set_verbosity("loud"), ConfigurationError),
        "set_verbosity(None)": (lambda: smallaxe.set_verbosity(None), ConfigurationError),
        # Loading
        "load_model(missing)": (lambda: smallaxe.load_model(str(tmp_path / "x")), ValidationError),
        "load_model(Path)": (lambda: smallaxe.load_model(tmp_path), ValidationError),
        "load_model(None)": (lambda: smallaxe.load_model(None), ValidationError),
        "load_pipeline(int)": (lambda: smallaxe.load_pipeline(3), ValidationError),
        "load(missing)": (lambda: smallaxe.load(str(tmp_path / "x")), ValidationError),
        "Imputer.load(Path)": (lambda: Imputer.load(Path(tmp_path)), ValidationError),
        "Scaler.load(missing)": (lambda: Scaler.load(str(tmp_path / "x")), ValidationError),
        "Encoder.load(None)": (lambda: Encoder.load(None), ValidationError),
        "regressors.load(missing)": (
            lambda: regressors.load(str(tmp_path / "x")),
            ValidationError,
        ),
        # Construction
        "Regressor('nope')": (lambda: Regressor("nope"), ValidationError),
        "Regressor(3)": (lambda: Regressor(3), ValidationError),
        "Model(None, task)": (lambda: Model(None, "binary"), ValidationError),
        "Regressor(task='binary')": (lambda: Regressor("random_forest", "binary"), ValidationError),
        "Classifier(task='ranking')": (
            lambda: Classifier("random_forest", "ranking"),
            ValidationError,
        ),
        "classifiers.random_forest(task=1)": (
            lambda: classifiers.random_forest(task=1),
            ValidationError,
        ),
        "algorithm.get('nope')": (lambda: algorithm.get("nope"), ValidationError),
        "Imputer(numerical_strategy=[])": (
            lambda: Imputer(numerical_strategy=[]),
            ValidationError,
        ),
        "Scaler(method='z')": (lambda: Scaler(method="z"), ValidationError),
        "Encoder(max_categories=0)": (lambda: Encoder(max_categories=0), ValidationError),
        "Encoder(handle_rare='drop')": (lambda: Encoder(handle_rare="drop"), ValidationError),
        "Pipeline([])": (lambda: Pipeline([]), ValidationError),
        "Pipeline(tuple)": (lambda: Pipeline((("s", Scaler()),)), ValidationError),
        "Pipeline([object])": (lambda: Pipeline([("o", object())]), ValidationError),
        # Parameters
        "set_params(nope=1)": (
            lambda: regressors.random_forest().set_params(nope=1),
            ValidationError,
        ),
        "set_params(n_estimators='5')": (
            lambda: regressors.random_forest().set_params(n_estimators="5"),
            ValidationError,
        ),
        "set_params(n_estimators=True)": (
            lambda: regressors.random_forest().set_params(n_estimators=True),
            ValidationError,
        ),
        "set_params(seed='1')": (
            lambda: regressors.random_forest().set_params(seed="1"),
            ValidationError,
        ),
        "factory(nope=1)": (lambda: regressors.random_forest(nope=1), ValidationError),
        "get_param('nope')": (
            lambda: regressors.random_forest().get_param("nope"),
            ValidationError,
        ),
        # Model.fit
        "fit(list)": (lambda: regressors.random_forest().fit([1, 2], "y"), ValidationError),
        "fit(label_col=None)": (
            lambda: regressors.random_forest().fit(frame, None),
            ValidationError,
        ),
        "fit(missing label)": (
            lambda: regressors.random_forest().fit(frame, "nope"),
            ColumnNotFoundError,
        ),
        "fit(feature_cols='x')": (
            lambda: regressors.random_forest().fit(frame, "y", feature_cols="x"),
            ValidationError,
        ),
        "fit(missing feature)": (
            lambda: regressors.random_forest().fit(frame, "y", feature_cols=["nope"]),
            ColumnNotFoundError,
        ),
        "fit(string feature)": (
            lambda: regressors.random_forest().fit(frame, "y", feature_cols=["c"]),
            ValidationError,
        ),
        "fit(validation='loo')": (
            lambda: regressors.random_forest().fit(frame, "y", validation="loo"),
            ValidationError,
        ),
        "fit(test_size=2)": (
            lambda: regressors.random_forest().fit(
                frame, "y", validation="train_test", test_size=2
            ),
            ValidationError,
        ),
        "fit(test_size='0.2')": (
            lambda: regressors.random_forest().fit(
                frame, "y", validation="train_test", test_size="0.2"
            ),
            ValidationError,
        ),
        "fit(n_folds=1)": (
            lambda: regressors.random_forest().fit(frame, "y", validation="kfold", n_folds=1),
            ValidationError,
        ),
        "fit(n_folds=2.5)": (
            lambda: regressors.random_forest().fit(frame, "y", validation="kfold", n_folds=2.5),
            ValidationError,
        ),
        "fit(cache_strategy='ram')": (
            lambda: regressors.random_forest().fit(frame, "y", cache_strategy="ram"),
            ValidationError,
        ),
        "fit(fold_key missing)": (
            lambda: regressors.random_forest().fit(frame, "y", validation="kfold", fold_key="nope"),
            ColumnNotFoundError,
        ),
        "fit(fold_key=3)": (
            lambda: regressors.random_forest().fit(frame, "y", validation="kfold", fold_key=3),
            ValidationError,
        ),
        "fit(refit='yes')": (
            lambda: regressors.random_forest().fit(frame, "y", validation="kfold", refit="yes"),
            ValidationError,
        ),
        # Using a fitted or unfitted model
        "predict before fit": (
            lambda: regressors.random_forest().predict(frame),
            ModelNotFittedError,
        ),
        "evaluate before fit": (
            lambda: regressors.random_forest().evaluate(frame),
            ModelNotFittedError,
        ),
        "save before fit": (
            lambda: regressors.random_forest().save(str(tmp_path / "m")),
            ModelNotFittedError,
        ),
        "metadata before fit": (lambda: regressors.random_forest().metadata, ModelNotFittedError),
        "importances before fit": (
            lambda: regressors.random_forest().feature_importances,
            ModelNotFittedError,
        ),
        "predict(missing feature)": (lambda: fitted.predict(frame.drop("x")), ColumnNotFoundError),
        "predict(list)": (lambda: fitted.predict([1]), ValidationError),
        "predict(output_col taken)": (
            lambda: fitted.predict(frame, output_col="x"),
            ValidationError,
        ),
        "evaluate(missing label)": (lambda: fitted.evaluate(frame.drop("y")), ColumnNotFoundError),
        "save('')": (lambda: fitted.save(""), ValidationError),
        "save(None)": (lambda: fitted.save(None), ValidationError),
        # Pipeline
        "pipeline transform before fit": (
            lambda: Pipeline([("s", Scaler())]).transform(frame),
            ModelNotFittedError,
        ),
        "pipeline predict without model": (
            lambda: Pipeline([("s", Scaler())]).fit(frame, numerical_cols=["x"]).predict(frame),
            ValidationError,
        ),
        "pipeline['nope']": (lambda: Pipeline([("s", Scaler())])["nope"], ValidationError),
        "pipeline without label": (
            lambda: Pipeline([("m", regressors.random_forest())]).fit(frame),
            ValidationError,
        ),
        "pipeline validation without model": (
            lambda: Pipeline([("s", Scaler())]).fit(frame, validation="kfold"),
            ValidationError,
        ),
        "pipeline save('')": (
            lambda: Pipeline([("s", Scaler())]).fit(frame, numerical_cols=["x"]).save(""),
            ValidationError,
        ),
        # Preprocessing
        "Imputer.fit no columns": (
            lambda: Imputer(numerical_strategy="mean").fit(frame),
            ValidationError,
        ),
        "Imputer.fit missing column": (
            lambda: Imputer(numerical_strategy="mean").fit(frame, numerical_cols=["nope"]),
            ColumnNotFoundError,
        ),
        "Imputer.fit(str cols)": (
            lambda: Imputer(numerical_strategy="mean").fit(frame, numerical_cols="x"),
            ValidationError,
        ),
        "Imputer.transform before fit": (
            lambda: Imputer(numerical_strategy="mean").transform(frame),
            ModelNotFittedError,
        ),
        "Scaler.fit(list)": (lambda: Scaler().fit([1], numerical_cols=["x"]), ValidationError),
        "Scaler.fit(str cols)": (lambda: Scaler().fit(frame, numerical_cols="x"), ValidationError),
        "Scaler.fit string column": (
            lambda: Scaler().fit(frame, numerical_cols=["c"]),
            ValidationError,
        ),
        "Scaler.transform missing column": (
            lambda: scaler.transform(frame.drop("x")),
            ColumnNotFoundError,
        ),
        "Encoder.fit(str cols)": (
            lambda: Encoder().fit(frame, categorical_cols="c"),
            ValidationError,
        ),
        "Encoder.fit missing column": (
            lambda: Encoder().fit(frame, categorical_cols=["nope"]),
            ColumnNotFoundError,
        ),
        "Encoder.save before fit": (
            lambda: Encoder().save(str(tmp_path / "e")),
            ModelNotFittedError,
        ),
        # Metrics
        "mse(list)": (lambda: metrics.mse([1], "y", "x"), ValidationError),
        "mse missing column": (lambda: metrics.mse(frame, "y", "nope"), ColumnNotFoundError),
        "mse empty": (lambda: metrics.mse(frame.limit(0), "y", "x"), ValidationError),
        "accuracy(label_col=None)": (lambda: metrics.accuracy(frame, None, "x"), ValidationError),
        "precision(average='micro')": (
            lambda: metrics.precision(frame, "y", "x", average="micro"),
            ValidationError,
        ),
        "mape string column": (lambda: metrics.mape(frame, "c", "x"), ValidationError),
        "log_loss string column": (lambda: metrics.log_loss(frame, "y", "c"), ValidationError),
        "multiclass_scores string": (
            lambda: metrics.multiclass_scores(frame, "c", "x"),
            ValidationError,
        ),
        # Search
        "optimize.run(model=None)": (
            lambda: optimize.run(None, frame, "y", {"max_depth": [2]}),
            ValidationError,
        ),
        "optimize.run(space={})": (
            lambda: optimize.run(regressors.random_forest(), frame, "y", {}),
            ValidationError,
        ),
        "optimize.run(space=list)": (
            lambda: optimize.run(regressors.random_forest(), frame, "y", [("max_depth", [2])]),
            ValidationError,
        ),
        "optimize.run(metric='auc_roc')": (
            lambda: optimize.run(
                regressors.random_forest(), frame, "y", {"max_depth": [2]}, metric="auc_roc"
            ),
            ValidationError,
        ),
        "optimize.run(validation='none')": (
            lambda: optimize.run(
                regressors.random_forest(), frame, "y", {"max_depth": [2]}, validation="none"
            ),
            ValidationError,
        ),
        "optimize.run(missing label)": (
            lambda: optimize.run(regressors.random_forest(), frame, "nope", {"max_depth": [2]}),
            ColumnNotFoundError,
        ),
        # Task names that are not strings
        "Classifier(task=['binary'])": (
            lambda: Classifier("random_forest", task=["binary"]),
            ValidationError,
        ),
        "Regressor(task=['simple_regression'])": (
            lambda: Regressor("random_forest", task=["simple_regression"]),
            ValidationError,
        ),
        "Model(task={})": (lambda: Model("random_forest", task={}), ValidationError),
        "classifiers.random_forest(task=['binary'])": (
            lambda: classifiers.random_forest(task=["binary"]),
            ValidationError,
        ),
        # Labels and features of the wrong type
        "fit(string label)": (lambda: rf().fit(frame, "c", feature_cols=["x"]), ValidationError),
        "fit(boolean label)": (
            lambda: rf().fit(frame.withColumn("t", F.col("x") > 3), "t", feature_cols=["x"]),
            ValidationError,
        ),
        "classifier fit(string label)": (
            lambda: classifiers.random_forest().fit(frame, "c", feature_cols=["x"]),
            ValidationError,
        ),
        "evaluate(string label)": (lambda: fitted.evaluate(frame, "c"), ValidationError),
        "predict(string feature)": (lambda: fitted.predict(string_x), ValidationError),
        "predict_proba(string feature)": (
            lambda: fitted_classifier.predict_proba(string_x),
            ValidationError,
        ),
        # Prediction column names
        "predict(output_col=None)": (
            lambda: fitted.predict(frame, output_col=None),
            ValidationError,
        ),
        "predict(output_col=1)": (lambda: fitted.predict(frame, output_col=1), ValidationError),
        "predict(output_col='')": (lambda: fitted.predict(frame, output_col=""), ValidationError),
        "predict_proba(output_col=1)": (
            lambda: fitted_classifier.predict_proba(frame, output_col=1),
            ValidationError,
        ),
        # Parameter values the estimator rejects
        "set_params(max_depth=-3).fit": (
            lambda: rf().set_params(max_depth=-3).fit(frame, "y", feature_cols=["x"]),
            ValidationError,
        ),
        "set_params(subsampling_rate=5.0).fit": (
            lambda: rf().set_params(subsampling_rate=5.0).fit(frame, "y", feature_cols=["x"]),
            ValidationError,
        ),
        "set_params(feature_subset_strategy='bogus').fit": (
            lambda: rf()
            .set_params(feature_subset_strategy="bogus")
            .fit(frame, "y", feature_cols=["x"]),
            ValidationError,
        ),
        # Metrics on string columns
        "mse string label": (lambda: metrics.mse(frame, "c", "x"), ValidationError),
        "rmse string prediction": (lambda: metrics.rmse(frame, "y", "c"), ValidationError),
        "r2 string label": (lambda: metrics.r2(frame, "c", "x"), ValidationError),
        "mae string label": (lambda: metrics.mae(frame, "c", "x"), ValidationError),
        "auc_roc string label": (lambda: metrics.auc_roc(scored, "c", "p"), ValidationError),
        "auc_pr string probability": (lambda: metrics.auc_pr(scored, "b", "c"), ValidationError),
        "auc_roc string probability": (lambda: metrics.auc_roc(scored, "b", "c"), ValidationError),
        # Search arguments
        "optimize.run(space={'max_depth': 'ab'})": (
            lambda: optimize.run(rf(), frame, "y", {"max_depth": "ab"}, feature_cols=["x"]),
            ValidationError,
        ),
        "optimize.run(space={'max_depth': ['a', 'b']})": (
            lambda: optimize.run(rf(), frame, "y", {"max_depth": ["a", "b"]}, feature_cols=["x"]),
            ValidationError,
        ),
        "optimize.run(space={1: [2, 3]})": (
            lambda: optimize.run(rf(), frame, "y", {1: [2, 3]}, feature_cols=["x"]),
            ValidationError,
        ),
        "optimize.run(validation=[])": (
            lambda: optimize.run(rf(), frame, "y", {"max_depth": [2]}, validation=[]),
            ValidationError,
        ),
        "optimize.run(verbose='y')": (
            lambda: optimize.run(rf(), frame, "y", {"max_depth": [2]}, verbose="y"),
            ValidationError,
        ),
        "optimize.run(refit='y')": (
            lambda: optimize.run(rf(), frame, "y", {"max_depth": [2]}, refit="y"),
            ValidationError,
        ),
        "optimize.run(early_stopping='y')": (
            lambda: optimize.run(rf(), frame, "y", {"max_depth": [2]}, early_stopping="y"),
            ValidationError,
        ),
        "optimize.run(test_size=2)": (
            lambda: optimize.run(rf(), frame, "y", {"max_depth": [2]}, test_size=2),
            ValidationError,
        ),
        "optimize.run(n_folds=1)": (
            lambda: optimize.run(
                rf(), frame, "y", {"max_depth": [2]}, validation="kfold", n_folds=1
            ),
            ValidationError,
        ),
        "optimize.run(string label)": (
            lambda: optimize.run(rf(), frame, "c", {"max_depth": [2]}, feature_cols=["x"]),
            ValidationError,
        ),
        # AutomatedTraining
        "AutomatedTraining('multiclass')": (
            lambda: AutomatedTraining("multiclass"),
            ValidationError,
        ),
        "AutomatedTraining(task=None)": (lambda: AutomatedTraining(None), ValidationError),
        "AutomatedTraining(algorithms=['nope'])": (
            lambda: AutomatedTraining("binary", algorithms=["nope"]),
            ValidationError,
        ),
        "AutomatedTraining(algorithms={1: 2})": (
            lambda: AutomatedTraining("binary", algorithms={1: 2}),
            ValidationError,
        ),
        "AutomatedTraining(metric='auc_roc') for regression": (
            lambda: AutomatedTraining("simple_regression", metric="auc_roc"),
            ValidationError,
        ),
        "AutomatedTraining(metric=['rmse'])": (
            lambda: AutomatedTraining("simple_regression", metric=["rmse"]),
            ValidationError,
        ),
        "AutomatedTraining(params for unrequested)": (
            lambda: AutomatedTraining(
                "binary", algorithms=["random_forest"], params={"xgboost": {"max_depth": 2}}
            ),
            ValidationError,
        ),
        "AutomatedTraining(params={'xgboost': None})": (
            lambda: AutomatedTraining("binary", params={"xgboost": None}),
            ValidationError,
        ),
        "AutomatedTraining(param_spaces=[])": (
            lambda: AutomatedTraining("binary", param_spaces=[]),
            ValidationError,
        ),
        "AutomatedTraining(max_evals='5')": (
            lambda: AutomatedTraining("binary", max_evals="5"),
            ValidationError,
        ),
        "AutomatedTraining(seed=1.5)": (
            lambda: AutomatedTraining("binary", seed=1.5),
            ValidationError,
        ),
        "AutomatedTraining.fit(list)": (
            lambda: AutomatedTraining("simple_regression").fit([1], "y"),
            ValidationError,
        ),
        "AutomatedTraining.fit(validation='none')": (
            lambda: AutomatedTraining("simple_regression").fit(frame, "y", validation="none"),
            ValidationError,
        ),
        "AutomatedTraining.fit(missing label)": (
            lambda: AutomatedTraining("simple_regression").fit(frame, "nope"),
            ColumnNotFoundError,
        ),
        "AutomatedTraining.fit(label_col=3)": (
            lambda: AutomatedTraining("simple_regression").fit(frame, 3),
            ValidationError,
        ),
        "AutomatedTraining.fit(string label)": (
            lambda: AutomatedTraining("simple_regression").fit(frame, "c", feature_cols=["x"]),
            ValidationError,
        ),
        "AutomatedTraining.fit(feature_cols='x')": (
            lambda: AutomatedTraining("simple_regression").fit(frame, "y", feature_cols="x"),
            ValidationError,
        ),
        "AutomatedTraining.fit(test_size='a')": (
            lambda: AutomatedTraining("simple_regression").fit(frame, "y", test_size="a"),
            ValidationError,
        ),
        "AutomatedTraining.fit(fold_key=[])": (
            lambda: AutomatedTraining("simple_regression").fit(frame, "y", fold_key=[]),
            ValidationError,
        ),
        "AutomatedTraining.fit(cache_strategy=[])": (
            lambda: AutomatedTraining("simple_regression").fit(frame, "y", cache_strategy=[]),
            ValidationError,
        ),
        "AutomatedTraining.best_model unfitted": (
            lambda: AutomatedTraining("binary").best_model,
            ModelNotFittedError,
        ),
        "AutomatedTraining.predict unfitted": (
            lambda: AutomatedTraining("binary").predict(frame),
            ModelNotFittedError,
        ),
        "AutomatedTraining.predict_proba for regression": (
            lambda: AutomatedTraining("simple_regression").predict_proba(frame),
            ValidationError,
        ),
        # Pipeline validation options of the wrong type on a pipeline without a model
        "pipeline refit=1 without model": (
            lambda: Pipeline([("s", Scaler())]).fit(frame, numerical_cols=["x"], refit=1),
            ValidationError,
        ),
        "pipeline n_folds=5.0 without model": (
            lambda: Pipeline([("s", Scaler())]).fit(frame, numerical_cols=["x"], n_folds=5.0),
            ValidationError,
        ),
        # Datasets
        "dataset_info('nope')": (lambda: dataset_info("nope"), ValidationError),
        "load_sample_regression(n_rows=0)": (
            lambda: load_sample_regression(n_rows=0),
            ValidationError,
        ),
        "load_sample_regression(spark='x')": (
            lambda: load_sample_regression("x"),
            ValidationError,
        ),
    }


def test_user_errors_raise_smallaxe_exceptions(frame, fitted, fitted_classifier, tmp_path):
    failures = []
    for name, (call, expected) in _cases(frame, fitted, fitted_classifier, tmp_path).items():
        try:
            call()
        except expected:
            continue
        except Exception as exc:  # noqa: BLE001 - collected and reported below
            failures.append(f"{name}: {type(exc).__name__}: {exc}"[:300])
        else:
            failures.append(f"{name}: raised nothing")
        finally:
            smallaxe.set_seed(None)
            smallaxe.set_verbosity("normal")
    assert failures == []


def test_every_expected_exception_is_a_smallaxe_error():
    for expected in (ColumnNotFoundError, ConfigurationError, ModelNotFittedError, ValidationError):
        assert issubclass(expected, SmallaxeError)
