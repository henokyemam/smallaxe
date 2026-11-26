"""AutomatedTraining: every available algorithm on one split, compared on one metric.

Real runs use Random Forest and XGBoost on small data with few trees. The ranking,
failure, skip, verbosity and caching rules are checked with ``Model.fit`` replaced by a
fake that returns scores the test chooses (``fake_fits``), so they need no training.
LightGBM and CatBoost are made unavailable (or, for fake fits, importable through
``tests.fakes``) so the tests behave the same wherever they run.
"""

import math
import warnings
from typing import Any, Dict, List

import pandas as pd
import pytest
from pyspark.sql import functions as F

import smallaxe
from smallaxe import _fs, auto
from smallaxe.auto import AutomatedTraining
from smallaxe.exceptions import (
    ColumnNotFoundError,
    DependencyError,
    ModelNotFittedError,
    ValidationError,
)
from smallaxe.search import optimize
from smallaxe.training import Classifier, Model, Regressor, algorithm
from smallaxe.training.algorithm import Algorithm
from smallaxe.training.base import BaseModel
from smallaxe.training.mixins.validation_mixin import ValidationMixin
from tests.fakes import install_fake_modules

NAN = float("nan")
ALL = ["random_forest", "xgboost", "lightgbm", "catboost"]
REGRESSION_COLUMNS = ["algorithm", "status", "rmse", "mae", "mape", "mse", "r2"]
REGRESSION_COLUMNS += ["fit_seconds", "params", "error"]
SMALL = {"random_forest": {"n_estimators": 5, "max_depth": 3}, "xgboost": {"n_estimators": 5}}

needs_xgboost = pytest.mark.skipif(
    not algorithm.get("xgboost").available(), reason="xgboost is not installed"
)


@pytest.fixture(autouse=True)
def clean_globals():
    smallaxe.set_seed(None)
    smallaxe.set_verbosity("normal")
    yield
    smallaxe.set_seed(None)
    smallaxe.set_verbosity("normal")


@pytest.fixture(scope="module")
def reg_df(spark_session):
    rows = [
        (i, float(i % 10), float((i * 7) % 13), 3.0 * (i % 10) + 0.5 * ((i * 7) % 13) + (i % 3))
        for i in range(120)
    ]
    return spark_session.createDataFrame(rows, ["id", "x", "z", "y"])


@pytest.fixture(scope="module")
def bin_df(spark_session):
    rows = [
        (
            i,
            float(i % 10),
            float((i * 7) % 13),
            1 if (i % 10) + ((i * 7) % 13) / 2 + (i % 3) > 8 else 0,
        )
        for i in range(120)
    ]
    return spark_session.createDataFrame(rows, ["id", "x", "z", "label"])


def _only_installed(names):
    """Patch the availability probe so only ``names`` (of the real algorithms) are usable."""
    original = Algorithm.available

    def available(self):
        return self.name in names and original(self)

    return available


def _skip_warnings(record) -> List[Any]:
    return [w for w in record if "AutomatedTraining skips" in str(w.message)]


def _default_run(task, frame, label):
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(Algorithm, "available", _only_installed({"random_forest", "xgboost"}))
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            trained = AutomatedTraining(task).fit(frame, label, exclude_cols=["id"])
    return trained, record


@pytest.fixture(scope="module")
def regression_run(reg_df):
    if not algorithm.get("xgboost").available():
        pytest.skip("xgboost is not installed")
    return _default_run("simple_regression", reg_df, "y")


@pytest.fixture(scope="module")
def binary_run(bin_df):
    if not algorithm.get("xgboost").available():
        pytest.skip("xgboost is not installed")
    return _default_run("binary", bin_df, "label")


# --- Fake fits: Model.fit returns scores the test chooses -----------------------------


class FakeFits:
    """What each algorithm's fake ``fit`` does, and the calls it saw."""

    def __init__(self) -> None:
        self.outcomes: Dict[str, Any] = {}
        self.calls: List[Dict[str, Any]] = []


@pytest.fixture
def fake_fits(monkeypatch):
    """Replace ``Model.fit``: no Spark training; scores (or an error) per algorithm.

    LightGBM and CatBoost get fake estimator modules, so all four algorithms are
    available and constructible.
    """
    for name in ("lightgbm", "catboost"):
        install_fake_modules(monkeypatch, algorithm.get(name))
    fakes = FakeFits()

    def fit(self, dataframe, label_col, feature_cols=None, exclude_cols=None, **options):
        name = self.algorithm.name
        fakes.calls.append(
            {"algorithm": name, "split_seed_fallback": self._split_seed_fallback, **options}
        )
        outcome = fakes.outcomes[name]
        if isinstance(outcome, BaseException):
            raise outcome
        if callable(outcome):
            outcome(dataframe)  # raises
        validation = options["validation"]
        prefix = "mean_" if validation == "kfold" else ""
        self._validation_scores = {prefix + metric: value for metric, value in outcome.items()}
        self._validation_scores["validation_type"] = validation
        self._label_col = label_col
        self._feature_cols = list(feature_cols or [])
        self._is_fitted = True
        return self

    monkeypatch.setattr(Model, "fit", fit)
    return fakes


# --- Default runs --------------------------------------------------------------------


class TestDefaultRuns:
    def test_regression_fits_rf_and_xgboost_and_skips_the_rest(self, regression_run, reg_df):
        trained, record = regression_run
        table = trained.comparison
        status = dict(zip(table["algorithm"], table["status"]))
        assert status == {
            "random_forest": "ok",
            "xgboost": "ok",
            "lightgbm": "skipped",
            "catboost": "skipped",
        }
        assert list(table.columns) == REGRESSION_COLUMNS
        assert sorted(trained.models) == ["random_forest", "xgboost"]
        assert isinstance(trained.best_model, Regressor)
        assert trained.best_model.task == "simple_regression"
        assert "predict_label" in trained.predict(reg_df).columns
        assert set(trained.evaluate(reg_df)) >= {"rmse", "r2"}

    def test_one_skip_warning_per_missing_algorithm_at_the_callers_line(self, regression_run):
        _, record = regression_run
        skips = _skip_warnings(record)
        assert sorted(str(w.message).split("'")[1] for w in skips) == ["catboost", "lightgbm"]
        for warning in skips:
            assert warning.category is UserWarning
            assert warning.filename == __file__
            assert "pip install smallaxe[" in str(warning.message)

    def test_regression_best_is_the_lowest_rmse(self, regression_run):
        trained, _ = regression_run
        table = trained.comparison
        ok = table[table["status"] == "ok"]
        assert trained.metric == "rmse"
        assert trained.best_algorithm == table.iloc[0]["algorithm"]
        assert trained.best_score == ok["rmse"].min()
        assert list(ok["rmse"]) == sorted(ok["rmse"])
        assert trained.best_model is trained.models[trained.best_algorithm]

    def test_every_algorithm_split_on_one_seed(self, regression_run):
        trained, _ = regression_run
        seeds = {model.metadata["split_seed"] for model in trained.models.values()}
        assert seeds == {trained._split_seed}
        for model in trained.models.values():
            assert model.metadata["kept_model"] == "full"  # refitted on every row
            assert model._split_seed_fallback is None  # not carried into later fits

    def test_binary_fits_rf_and_xgboost_best_is_the_highest_auc(self, binary_run, bin_df):
        trained, record = binary_run
        table = trained.comparison
        assert len(_skip_warnings(record)) == 2
        assert list(table["status"]) == ["ok", "ok", "skipped", "skipped"]
        assert trained.metric == "auc_roc"
        assert trained.best_score == table[table["status"] == "ok"]["auc_roc"].max()
        assert trained.best_algorithm == table.iloc[0]["algorithm"]
        assert isinstance(trained.best_model, Classifier)
        assert "probability" in trained.predict_proba(bin_df).columns
        assert "predict_label" in trained.predict(bin_df).columns

    def test_repr_names_task_metric_and_winner(self, binary_run):
        trained, _ = binary_run
        text = repr(trained)
        assert "task='binary'" in text and "metric='auc_roc'" in text
        assert f"best_algorithm='{trained.best_algorithm}'" in text
        assert repr(AutomatedTraining("binary")) == (
            "AutomatedTraining(task='binary', metric='auc_roc')"
        )

    def test_comparison_is_a_fresh_copy(self, regression_run):
        trained, _ = regression_run
        table = trained.comparison
        table.loc[0, "status"] = "changed"
        table.loc[0, "params"]["n_estimators"] = 999
        again = trained.comparison
        assert again.loc[0, "status"] == "ok"
        assert "n_estimators" not in again.loc[0, "params"]
        assert isinstance(again, pd.DataFrame)


# --- One split, reproducible ---------------------------------------------------------


@needs_xgboost
class TestSameRowsAndSeed:
    def _run(self, frame, monkeypatch, **options):
        splits: List[Any] = []
        original = ValidationMixin._train_test_split

        def recording(model, dataframe, *args, **kwargs):
            train, test = original(model, dataframe, *args, **kwargs)
            splits.append((model.algorithm.name, model._split_seed, test.count()))
            return train, test

        monkeypatch.setattr(ValidationMixin, "_train_test_split", recording)
        trained = AutomatedTraining(
            "simple_regression", algorithms=["random_forest", "xgboost"], params=SMALL, **options
        ).fit(frame, "y", feature_cols=["x", "z"])
        return trained, splits

    def test_both_algorithms_score_on_the_same_rows_unseeded(self, reg_df, monkeypatch):
        trained, splits = self._run(reg_df, monkeypatch)
        assert {name for name, _, _ in splits} == {"random_forest", "xgboost"}
        assert len({(seed, count) for _, seed, count in splits}) == 1
        assert splits[0][1] == trained._split_seed

    def test_seed_makes_the_comparison_reproducible(self, reg_df, monkeypatch):
        first, splits = self._run(reg_df, monkeypatch, seed=7)
        second, _ = self._run(reg_df, monkeypatch, seed=7)
        assert {seed for _, seed, _ in splits} == {7}
        columns = ["algorithm", "status", "rmse", "mae", "mape", "mse", "r2", "params"]
        pd.testing.assert_frame_equal(first.comparison[columns], second.comparison[columns])
        assert first.comparison.loc[0, "params"]["seed"] == 7
        for model in first.models.values():
            assert model.metadata["split_seed"] == 7
            assert model.get_params()["seed"] == 7

    def test_global_seed_is_the_run_seed(self, reg_df, monkeypatch):
        smallaxe.set_seed(5)
        trained, splits = self._run(reg_df, monkeypatch)
        assert {seed for _, seed, _ in splits} == {5}
        assert trained._split_seed == 5

    def test_a_search_scores_its_trials_on_the_run_split(self, reg_df, monkeypatch):
        spied: List[Any] = []
        original_run = optimize.run

        def spy(model, *args, **kwargs):
            spied.append(model.algorithm.name)
            return original_run(model, *args, **kwargs)

        monkeypatch.setattr(optimize, "run", spy)
        trained, splits = self._run(
            reg_df, monkeypatch, param_spaces={"random_forest": {"max_depth": [2, 3]}}, max_evals=2
        )
        assert spied == ["random_forest"]
        # Two trials plus xgboost's one validation fit, all on one seed and one test set.
        assert [name for name, _, _ in splits].count("random_forest") == 2
        assert len({(seed, count) for _, seed, count in splits}) == 1
        table = trained.comparison.set_index("algorithm")
        assert table.loc["random_forest", "params"]["max_depth"] in (2, 3)
        assert table.loc["random_forest", "params"]["n_estimators"] == 5
        assert table.loc["xgboost", "params"] == {"n_estimators": 5}
        # optimize.run scores trials on the compared metric only.
        assert not math.isnan(table.loc["random_forest", "rmse"])
        assert math.isnan(table.loc["random_forest", "r2"])
        assert not math.isnan(table.loc["xgboost", "r2"])
        assert trained.models["random_forest"].get_params()["max_depth"] in (2, 3)


@needs_xgboost
def test_a_real_training_failure_is_reported_and_the_run_continues(spark_session):
    # Random Forest rejects NaN features; XGBoost treats them as missing.
    rows = [(float(i % 5) if i % 4 else NAN, float(i % 7), float(i % 5 + i % 7)) for i in range(60)]
    frame = spark_session.createDataFrame(rows, ["x", "z", "y"])
    with pytest.warns(UserWarning, match="'random_forest' failed") as record:
        trained = AutomatedTraining(
            "simple_regression", algorithms=["random_forest", "xgboost"], params=SMALL, seed=3
        ).fit(frame, "y")
    failure = [w for w in record if "failed" in str(w.message)][0]
    assert failure.filename == __file__
    table = trained.comparison
    assert list(table["algorithm"]) == ["xgboost", "random_forest"]
    assert list(table["status"]) == ["ok", "failed"]
    assert "null or NaN" in table.loc[1, "error"]
    assert table.loc[0, "error"] is None
    assert math.isnan(table.loc[1, "rmse"])
    assert trained.best_algorithm == "xgboost"
    assert sorted(trained.models) == ["xgboost"]


# --- Ranking (fake fits) -------------------------------------------------------------


class TestRanking:
    def test_minimised_metric_picks_the_lowest(self, reg_df, fake_fits):
        fake_fits.outcomes.update(
            random_forest={"rmse": 3.0, "r2": 0.1},
            xgboost={"rmse": 1.0, "r2": 0.2},
            lightgbm={"rmse": 2.0, "r2": 0.9},
            catboost={"rmse": NAN, "r2": 0.95},
        )
        trained = AutomatedTraining("simple_regression").fit(reg_df, "y", ["x", "z"])
        table = trained.comparison
        assert list(table["algorithm"]) == ["xgboost", "lightgbm", "random_forest", "catboost"]
        assert list(table["status"]) == ["ok"] * 4
        assert trained.best_algorithm == "xgboost"
        assert trained.best_score == 1.0
        assert math.isnan(table.loc[3, "rmse"])  # undefined ranks last, still ok
        assert "catboost" in trained.models

    def test_maximised_metric_picks_the_highest(self, reg_df, fake_fits):
        fake_fits.outcomes.update(
            random_forest={"rmse": 3.0, "r2": 0.1},
            xgboost={"rmse": 1.0, "r2": 0.2},
            lightgbm={"rmse": 2.0, "r2": 0.9},
            catboost={"rmse": 0.5, "r2": None},
        )
        trained = AutomatedTraining("simple_regression", metric="r2").fit(reg_df, "y", ["x", "z"])
        assert list(trained.comparison["algorithm"]) == [
            "lightgbm",
            "xgboost",
            "random_forest",
            "catboost",
        ]
        assert trained.best_algorithm == "lightgbm"
        assert trained.best_score == 0.9

    def test_binary_auc_with_kfold_uses_the_fold_mean(self, bin_df, fake_fits):
        fake_fits.outcomes.update(
            random_forest={"auc_roc": 0.7, "accuracy": 0.6},
            xgboost={"auc_roc": 0.8, "accuracy": 0.5},
        )
        trained = AutomatedTraining("binary", algorithms=["random_forest", "xgboost"]).fit(
            bin_df, "label", ["x", "z"], validation="kfold", n_folds=3
        )
        assert trained.best_algorithm == "xgboost"
        assert trained.comparison.loc[0, "accuracy"] == 0.5
        assert {call["validation"] for call in fake_fits.calls} == {"kfold"}
        assert {call["n_folds"] for call in fake_fits.calls} == {3}

    def test_undefined_score_never_wins(self, bin_df, fake_fits):
        fake_fits.outcomes.update(
            random_forest={"auc_roc": None}, xgboost={"auc_roc": 0.4}, lightgbm={"auc_roc": NAN}
        )
        trained = AutomatedTraining("binary", algorithms=["random_forest", "xgboost", "lightgbm"])
        trained.fit(bin_df, "label", ["x", "z"])
        assert trained.best_algorithm == "xgboost"
        assert list(trained.comparison["algorithm"]) == ["xgboost", "random_forest", "lightgbm"]

    def test_undefined_for_every_algorithm_raises(self, bin_df, fake_fits):
        fake_fits.outcomes.update(random_forest={"auc_roc": None}, xgboost={"auc_roc": NAN})
        trained = AutomatedTraining("binary", algorithms=["random_forest", "xgboost"])
        with pytest.raises(ValidationError, match="undefined .* for every algorithm"):
            trained.fit(bin_df, "label", ["x", "z"])
        with pytest.raises(ModelNotFittedError):
            _ = trained.best_model

    def test_every_model_is_fitted_with_refit_and_no_own_cache(self, reg_df, fake_fits):
        fake_fits.outcomes.update(random_forest={"rmse": 1.0}, xgboost={"rmse": 2.0})
        AutomatedTraining("simple_regression", algorithms=["random_forest", "xgboost"]).fit(
            reg_df, "y", ["x"], test_size=0.3, fold_key="id"
        )
        for call in fake_fits.calls:
            assert call["refit"] is True
            assert call["cache_strategy"] == "none"
            assert call["test_size"] == 0.3
            assert call["fold_key"] == "id"
        assert len({call["split_seed_fallback"] for call in fake_fits.calls}) == 1


# --- Failures and skips --------------------------------------------------------------


class TestFailuresAndSkips:
    def test_failed_algorithms_are_reported_and_the_rest_complete(self, reg_df, fake_fits):
        with pytest.raises(_fs.jvm_error_types()):
            reg_df.select(F.raise_error(F.lit("JVM boom"))).collect()
        fake_fits.outcomes.update(
            random_forest={"rmse": 2.0},
            xgboost=ValidationError("bad xgboost"),
            # A real JVM-side error, as Spark raises it from a failing job.
            lightgbm=lambda frame: frame.select(F.raise_error(F.lit("JVM boom"))).collect(),
            catboost={"rmse": 1.0},
        )
        with pytest.warns(UserWarning, match="failed") as record:
            trained = AutomatedTraining("simple_regression").fit(reg_df, "y", ["x"])
        failures = [w for w in record if "failed" in str(w.message)]
        assert len(failures) == 2
        assert all(w.filename == __file__ for w in failures)
        table = trained.comparison.set_index("algorithm")
        assert list(trained.comparison["status"]) == ["ok", "ok", "failed", "failed"]
        assert table.loc["xgboost", "error"] == "ValidationError: bad xgboost"
        assert "JVM boom" in table.loc["lightgbm", "error"]
        assert table.loc["xgboost", "fit_seconds"] >= 0
        assert sorted(trained.models) == ["catboost", "random_forest"]
        assert trained.best_algorithm == "catboost"

    def test_every_algorithm_failing_raises_with_each_error(self, reg_df, fake_fits):
        fake_fits.outcomes.update(
            random_forest=ValidationError("rf broke"), xgboost=ValidationError("xgb broke")
        )
        trained = AutomatedTraining("simple_regression", algorithms=["random_forest", "xgboost"])
        with pytest.warns(UserWarning):
            with pytest.raises(ValidationError, match="rf broke") as info:
                trained.fit(reg_df, "y", ["x"])
        assert "xgb broke" in str(info.value)

    def test_failed_and_skipped_together_raise(self, reg_df, fake_fits, monkeypatch):
        monkeypatch.setattr(Algorithm, "available", _only_installed({"random_forest"}))
        fake_fits.outcomes.update(random_forest=ValidationError("rf broke"))
        trained = AutomatedTraining("simple_regression", algorithms=["random_forest", "xgboost"])
        with pytest.warns(UserWarning):
            with pytest.raises(ValidationError, match="rf broke.*xgboost: .*not installed"):
                trained.fit(reg_df, "y", ["x"])

    @pytest.mark.parametrize("error", [KeyboardInterrupt(), RuntimeError("not a user error")])
    def test_other_exceptions_propagate(self, reg_df, fake_fits, error):
        fake_fits.outcomes.update(random_forest={"rmse": 1.0}, xgboost=error)
        trained = AutomatedTraining("simple_regression", algorithms=["random_forest", "xgboost"])
        with pytest.raises(type(error)):
            trained.fit(reg_df, "y", ["x"])

    def test_unavailable_algorithm_is_skipped_with_its_install_hint(
        self, reg_df, fake_fits, monkeypatch
    ):
        monkeypatch.setattr(Algorithm, "available", _only_installed({"random_forest"}))
        fake_fits.outcomes.update(random_forest={"rmse": 1.0})
        with pytest.warns(UserWarning, match="skips 'xgboost'") as record:
            trained = AutomatedTraining(
                "simple_regression", algorithms=["xgboost", "random_forest"]
            ).fit(reg_df, "y", ["x"])
        skips = _skip_warnings(record)
        assert len(skips) == 1
        assert skips[0].filename == __file__
        assert algorithm.get("xgboost").dependency.install_hint in str(skips[0].message)
        row = trained.comparison.set_index("algorithm").loc["xgboost"]
        assert row["status"] == "skipped"
        assert "pip install smallaxe[xgboost]" in row["error"]
        assert [call["algorithm"] for call in fake_fits.calls] == ["random_forest"]

    def test_the_probe_is_the_records_availability_check(self, reg_df, fake_fits, monkeypatch):
        probed: List[str] = []
        original = Algorithm.available

        def available(record):
            probed.append(record.name)
            return original(record)

        monkeypatch.setattr(Algorithm, "available", available)
        fake_fits.outcomes.update(random_forest={"rmse": 1.0})
        AutomatedTraining("simple_regression", algorithms=["random_forest"]).fit(reg_df, "y", ["x"])
        assert probed == ["random_forest"]

    def test_nothing_installed_raises_dependency_error(self, reg_df, monkeypatch):
        monkeypatch.setattr(Algorithm, "available", _only_installed(set()))
        trained = AutomatedTraining("simple_regression", algorithms=["xgboost", "catboost"])
        with pytest.warns(UserWarning, match="skips"):
            with pytest.raises(DependencyError, match="xgboost.*catboost"):
                trained.fit(reg_df, "y", ["x"])


# --- Verbosity and caching -----------------------------------------------------------


class TestVerbosityAndCaching:
    def test_progress_lines_follow_the_global_verbosity(self, reg_df, fake_fits, capsys):
        fake_fits.outcomes.update(random_forest={"rmse": 1.0}, xgboost={"rmse": 2.0})
        trained = AutomatedTraining("simple_regression", algorithms=["random_forest", "xgboost"])
        trained.fit(reg_df, "y", ["x"])
        out = capsys.readouterr().out
        assert "training random_forest (1/2)..." in out
        assert "training xgboost (2/2)..." in out
        assert "xgboost rmse=2" in out

        smallaxe.set_verbosity("quiet")
        trained.fit(reg_df, "y", ["x"])
        assert capsys.readouterr().out == ""

    @pytest.mark.parametrize("strategy, persists", [(None, 1), ("none", 0), ("memory", 1)])
    def test_the_data_is_cached_once_per_run(
        self, reg_df, fake_fits, monkeypatch, strategy, persists
    ):
        fake_fits.outcomes.update(random_forest={"rmse": 1.0}, xgboost={"rmse": 2.0})
        calls: List[Any] = []
        original = BaseModel._apply_cache_strategy

        def recording(model, dataframe, *args):
            result = original(model, dataframe, *args)
            calls.append(result[1])
            return result

        monkeypatch.setattr(BaseModel, "_apply_cache_strategy", recording)
        frame = reg_df.select("*")
        AutomatedTraining("simple_regression", algorithms=["random_forest", "xgboost"]).fit(
            frame, "y", ["x"], cache_strategy=strategy
        )
        assert calls.count(True) == persists and len(calls) == 1
        assert not frame.storageLevel.useMemory  # unpersisted after the run


# --- Arguments -----------------------------------------------------------------------


def _bad_inits() -> Dict[str, Any]:
    return {
        "multiclass": dict(task="multiclass"),
        "unknown task": dict(task="ranking"),
        "task not a string": dict(task=["binary"]),
        "unknown algorithm": dict(task="binary", algorithms=["nope"]),
        "duplicate algorithm": dict(task="binary", algorithms=["xgboost", "xgboost"]),
        "empty algorithms": dict(task="binary", algorithms=[]),
        "algorithms a string": dict(task="binary", algorithms="xgboost"),
        "metric of another task": dict(task="simple_regression", metric="auc_roc"),
        "multiclass-only metric": dict(task="binary", metric="f1_score_macro"),
        "unknown metric": dict(task="binary", metric="nope"),
        "metric not a string": dict(task="binary", metric=3),
        "params not requested": dict(
            task="binary", algorithms=["xgboost"], params={"random_forest": {"max_depth": 3}}
        ),
        "params unknown name": dict(task="binary", params={"xgboost": {"nope": 1}}),
        "params not a dict": dict(task="binary", params=[1]),
        "params entry not a dict": dict(task="binary", params={"xgboost": 3}),
        "params seed": dict(task="binary", params={"xgboost": {"seed": 3}}),
        "space not requested": dict(
            task="binary", algorithms=["xgboost"], param_spaces={"lightgbm": {"max_depth": [3]}}
        ),
        "space empty": dict(task="binary", param_spaces={"xgboost": {}}),
        "space seed": dict(task="binary", param_spaces={"xgboost": {"seed": [1, 2]}}),
        "max_evals 0": dict(task="binary", max_evals=0),
        "max_evals bool": dict(task="binary", max_evals=True),
        "seed string": dict(task="binary", seed="1"),
        "seed bool": dict(task="binary", seed=True),
    }


@pytest.mark.parametrize("name, kwargs", list(_bad_inits().items()))
def test_bad_constructor_arguments_raise_validation_error(name, kwargs):
    with pytest.raises(ValidationError):
        AutomatedTraining(**kwargs)


def test_multiclass_message_names_the_supported_tasks():
    with pytest.raises(ValidationError, match="'simple_regression' and 'binary'"):
        AutomatedTraining("multiclass")


@pytest.fixture
def no_training(monkeypatch):
    """Fail the test if any model trains or any search starts."""

    def refuse(*args, **kwargs):
        raise AssertionError("a model was trained before the arguments were checked")

    monkeypatch.setattr(Model, "fit", refuse)
    monkeypatch.setattr(optimize, "run", refuse)


def _bad_fits(frame):
    """name -> (zero-argument call, expected exception)."""
    rf = AutomatedTraining("simple_regression", algorithms=["random_forest"])
    spaced = AutomatedTraining(
        "simple_regression",
        algorithms=["random_forest"],
        param_spaces={"random_forest": {"max_depth": ["a", "b"]}},
    )
    wrong_type = AutomatedTraining(
        "simple_regression",
        algorithms=["random_forest"],
        params={"random_forest": {"n_estimators": "5"}},
    )
    nulls = frame.withColumn("y", F.when(F.col("id") == 3, None).otherwise(F.col("y")))
    return {
        "validation none": (lambda: rf.fit(frame, "y", validation="none"), ValidationError),
        "validation list": (lambda: rf.fit(frame, "y", validation=[]), ValidationError),
        "missing label": (lambda: rf.fit(frame, "nope"), ColumnNotFoundError),
        "label None": (lambda: rf.fit(frame, None), ValidationError),
        "not a DataFrame": (lambda: rf.fit([1, 2], "y"), ValidationError),
        "pandas DataFrame": (lambda: rf.fit(pd.DataFrame({"y": [1]}), "y"), ValidationError),
        "string label": (
            lambda: rf.fit(frame.withColumn("s", F.lit("a")), "s", ["x"]),
            ValidationError,
        ),
        "missing feature": (lambda: rf.fit(frame, "y", ["nope"]), ColumnNotFoundError),
        "feature_cols string": (lambda: rf.fit(frame, "y", "x"), ValidationError),
        "exclude_cols int": (lambda: rf.fit(frame, "y", exclude_cols=3), ValidationError),
        "test_size 2": (lambda: rf.fit(frame, "y", test_size=2), ValidationError),
        "n_folds 1": (lambda: rf.fit(frame, "y", validation="kfold", n_folds=1), ValidationError),
        "fold_key int": (lambda: rf.fit(frame, "y", fold_key=3), ValidationError),
        "fold_key missing": (lambda: rf.fit(frame, "y", fold_key="nope"), ColumnNotFoundError),
        "cache_strategy bogus": (
            lambda: rf.fit(frame, "y", cache_strategy="bogus"),
            ValidationError,
        ),
        "null labels": (lambda: rf.fit(nulls, "y", ["x"]), ValidationError),
        "param value of the wrong type": (
            lambda: wrong_type.fit(frame, "y", ["x"]),
            ValidationError,
        ),
        "malformed space": (lambda: spaced.fit(frame, "y", ["x"]), ValidationError),
    }


_BAD_FIT_NAMES = [
    "validation none",
    "validation list",
    "missing label",
    "label None",
    "not a DataFrame",
    "pandas DataFrame",
    "string label",
    "missing feature",
    "feature_cols string",
    "exclude_cols int",
    "test_size 2",
    "n_folds 1",
    "fold_key int",
    "fold_key missing",
    "cache_strategy bogus",
    "null labels",
    "param value of the wrong type",
    "malformed space",
]


def test_every_bad_fit_case_is_parametrized(reg_df):
    assert sorted(_bad_fits(reg_df)) == sorted(_BAD_FIT_NAMES)


@pytest.mark.parametrize("name", _BAD_FIT_NAMES)
def test_bad_fit_arguments_raise_before_anything_trains(reg_df, no_training, name):
    call, expected = _bad_fits(reg_df)[name]
    with pytest.raises(expected):
        call()


def test_unfitted_access_raises_model_not_fitted(reg_df):
    trained = AutomatedTraining("binary", algorithms=["random_forest"])
    for name in ("best_model", "best_algorithm", "best_score", "models", "comparison"):
        with pytest.raises(ModelNotFittedError):
            getattr(trained, name)
    for call in (
        lambda: trained.predict(reg_df),
        lambda: trained.predict_proba(reg_df),
        lambda: trained.evaluate(reg_df),
    ):
        with pytest.raises(ModelNotFittedError):
            call()
    assert trained.task == "binary"
    assert trained.metric == "auc_roc"
    assert trained.algorithms == ["random_forest"]


def test_always_available_properties_resolve_defaults():
    trained = AutomatedTraining("simple_regression")
    assert trained.algorithms == ALL
    assert trained.metric == "rmse"
    listed = trained.algorithms
    listed.append("x")
    assert trained.algorithms == ALL


def test_predict_proba_is_binary_only(reg_df, fake_fits):
    fake_fits.outcomes.update(random_forest={"rmse": 1.0})
    trained = AutomatedTraining("simple_regression", algorithms=["random_forest"])
    with pytest.raises(ValidationError, match="binary"):
        trained.predict_proba(reg_df)
    trained.fit(reg_df, "y", ["x"])
    with pytest.raises(ValidationError, match="binary"):
        trained.predict_proba(reg_df)


def test_the_package_exports_automated_training_only():
    assert auto.__all__ == ["AutomatedTraining"]
    assert auto.AutomatedTraining is AutomatedTraining


def test_an_empty_dataframe_is_rejected_before_anything_trains(reg_df, no_training):
    trained = AutomatedTraining("simple_regression", algorithms=["random_forest", "xgboost"])
    with pytest.raises(ValidationError, match="no rows"):
        trained.fit(reg_df.limit(0), "y", ["x"])


def test_a_fit_that_raises_discards_the_previous_results(reg_df, fake_fits):
    fake_fits.outcomes.update(random_forest={"rmse": 1.0})
    trained = AutomatedTraining("simple_regression", algorithms=["random_forest"])
    trained.fit(reg_df, "y", ["x"])
    assert trained.best_algorithm == "random_forest"
    with pytest.raises(ValidationError):
        trained.fit(reg_df, "y", ["x"], validation="none")
    for read in (
        lambda: trained.best_model,
        lambda: trained.best_score,
        lambda: trained.comparison,
        lambda: trained.models,
    ):
        with pytest.raises(ModelNotFittedError):
            read()


@pytest.mark.parametrize("metric", ["", 0, [], {}])
def test_a_falsy_metric_is_not_read_as_the_default(metric):
    with pytest.raises(ValidationError, match="Invalid metric"):
        AutomatedTraining("simple_regression", metric=metric)
