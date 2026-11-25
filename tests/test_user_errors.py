"""Public methods raise smallaxe exceptions, never a raw ValueError / KeyError / TypeError,
for user errors: bad arguments, missing columns, unfitted objects, bad paths.

Each case is a call a user could plausibly get wrong, and the smallaxe exception it must
raise. (A wrong *number* of arguments or an unknown keyword is a Python ``TypeError`` by
design, raised before smallaxe runs.)
"""

from pathlib import Path

import pytest

import smallaxe
from smallaxe import metrics
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


def _cases(frame, fitted, tmp_path):
    """name -> (zero-argument call, expected exception)."""
    scaler = Scaler().fit(frame, numerical_cols=["x"])
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


def test_user_errors_raise_smallaxe_exceptions(frame, fitted, tmp_path):
    failures = []
    for name, (call, expected) in _cases(frame, fitted, tmp_path).items():
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
