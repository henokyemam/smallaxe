"""The model ``fit`` keeps, what it caches, which seed it uses, and what it infers.

See ``GOTCHAS.md`` items 4 (refit), 5 (caching), 6 (global settings), and 7
(feature inference).
"""

import warnings

import pytest
from pyspark import StorageLevel

import smallaxe
from smallaxe.training import classifiers, regressors


@pytest.fixture
def frame(spark_session):
    rows = [
        (i, f"id-{i}", i % 2 == 0, 20.0 + (i % 30), 1000.0 * i, 50.0 + i * 5 + (i % 7), i % 2)
        for i in range(120)
    ]
    return spark_session.createDataFrame(
        rows, ["id", "customer", "active", "age", "income", "target", "label"]
    )


@pytest.fixture(autouse=True)
def no_global_seed():
    smallaxe.set_seed(None)
    yield
    smallaxe.set_seed(None)


def _predictions(model, df):
    return [r["p"] for r in model.predict(df, output_col="p").orderBy("id").collect()]


# --- Item 4: refit -----------------------------------------------------------------


class TestRefit:
    def test_train_test_keeps_a_model_trained_on_all_rows(self, frame):
        model = regressors.random_forest(n_estimators=3, seed=1)
        model.fit(
            frame, label_col="target", feature_cols=["age", "income"], validation="train_test"
        )
        full = regressors.random_forest(n_estimators=3, seed=1)
        full.fit(frame, label_col="target", feature_cols=["age", "income"])

        assert model.metadata["kept_model"] == "full"
        assert model.metadata["refit"] is True
        assert model.metadata["n_training_rows"] == 120
        assert _predictions(model, frame) == _predictions(full, frame)

    def test_kfold_keeps_a_model_trained_on_all_rows(self, frame):
        model = regressors.random_forest(n_estimators=3, seed=1)
        model.fit(
            frame,
            label_col="target",
            feature_cols=["age", "income"],
            validation="kfold",
            n_folds=3,
        )
        assert model.metadata["kept_model"] == "full"
        assert model.metadata["n_training_rows"] == 120

    def test_refit_false_keeps_the_train_split_model(self, frame):
        model = regressors.random_forest(n_estimators=3, seed=1)
        model.fit(
            frame,
            label_col="target",
            feature_cols=["age", "income"],
            validation="train_test",
            test_size=0.25,
            refit=False,
        )
        assert model.metadata["kept_model"] == "train_split"
        assert model.metadata["refit"] is False
        assert 70 < model.metadata["n_training_rows"] < 110

    def test_refit_false_keeps_the_last_fold_model(self, frame):
        model = regressors.random_forest(n_estimators=3, seed=1)
        model.fit(
            frame,
            label_col="target",
            feature_cols=["age", "income"],
            validation="kfold",
            n_folds=4,
            refit=False,
        )
        assert model.metadata["kept_model"] == "last_fold"
        assert 70 < model.metadata["n_training_rows"] < 110

    def test_refit_false_trains_once_per_split(self, frame, monkeypatch):
        model = regressors.random_forest(n_estimators=3, seed=1)
        calls = []
        original = type(model)._fit_spark_model

        def counting(self, df, label_col, feature_cols):
            calls.append(1)
            return original(self, df, label_col, feature_cols)

        monkeypatch.setattr(type(model), "_fit_spark_model", counting)
        model.fit(
            frame,
            label_col="target",
            feature_cols=["age"],
            validation="kfold",
            n_folds=3,
            refit=False,
        )
        assert len(calls) == 3

    def test_validation_none_records_full(self, frame):
        model = regressors.random_forest(n_estimators=3, seed=1)
        model.fit(frame, label_col="target", feature_cols=["age"])
        assert model.metadata["kept_model"] == "full"
        assert model.metadata["n_training_rows"] == 120


# --- Item 5: caching ---------------------------------------------------------------


def _flags(level):
    """A StorageLevel as a tuple. ``StorageLevel.__eq__`` and ``.NONE`` need Spark 3.4."""
    return (level.useDisk, level.useMemory, level.useOffHeap, level.deserialized)


def _is_cached(df):
    level = df.storageLevel
    return level.useDisk or level.useMemory or level.useOffHeap


class TestCaching:
    def test_a_dataframe_the_user_cached_stays_cached(self, frame):
        cached = frame.persist(StorageLevel.MEMORY_AND_DISK)
        same_data = cached.select("*")
        try:
            model = regressors.random_forest(n_estimators=2, seed=1)
            for strategy in (None, "memory", "disk", "none"):
                model.fit(
                    same_data,
                    label_col="target",
                    feature_cols=["age"],
                    validation="kfold",
                    n_folds=2,
                    cache_strategy=strategy,
                )
                # ``same_data`` is a new plan over the cached one; Spark reports its own
                # level as NONE, so smallaxe may cache it too, but must leave the
                # caller's cache in place.
                assert _is_cached(cached)
                assert not _is_cached(same_data)
        finally:
            cached.unpersist()

    def test_the_same_plan_cached_by_the_user_is_not_persisted_again(self, frame, monkeypatch):
        cached = frame.persist(StorageLevel.MEMORY_ONLY)
        try:
            model = regressors.random_forest(n_estimators=2, seed=1)
            persisted = []
            original = type(frame).persist

            def spy(self, *args, **kwargs):
                persisted.append(1)
                return original(self, *args, **kwargs)

            monkeypatch.setattr(type(frame), "persist", spy)
            model.fit(cached, label_col="target", feature_cols=["age"], validation="kfold")
            assert persisted == []
            assert _flags(cached.storageLevel) == _flags(StorageLevel.MEMORY_ONLY)
        finally:
            cached.unpersist()

    def test_default_caches_for_kfold_and_unpersists_afterwards(self, frame, monkeypatch):
        model = regressors.random_forest(n_estimators=2, seed=1)
        seen = {}
        original = type(model)._fit_spark_model

        def spy(self, df, label_col, feature_cols):
            seen["level"] = df.storageLevel
            return original(self, df, label_col, feature_cols)

        monkeypatch.setattr(type(model), "_fit_spark_model", spy)
        model.fit(frame, label_col="target", feature_cols=["age"], validation="kfold", n_folds=2)

        assert _flags(seen["level"]) == _flags(StorageLevel.MEMORY_AND_DISK)
        assert not _is_cached(frame)

    def test_default_does_not_cache_a_single_pass(self, frame, monkeypatch):
        model = regressors.random_forest(n_estimators=2, seed=1)
        seen = {}
        original = type(model)._fit_spark_model

        def spy(self, df, label_col, feature_cols):
            seen["level"] = df.storageLevel
            return original(self, df, label_col, feature_cols)

        monkeypatch.setattr(type(model), "_fit_spark_model", spy)
        model.fit(frame, label_col="target", feature_cols=["age"])
        assert not (seen["level"].useMemory or seen["level"].useDisk)

    def test_persist_failure_warns_and_continues(self, frame, monkeypatch):
        model = regressors.random_forest(n_estimators=2, seed=1)

        def failing(self, level):
            raise RuntimeError("persist is disabled here")

        monkeypatch.setattr(type(frame), "persist", failing)
        with pytest.warns(UserWarning, match="Could not cache"):
            model.fit(frame, label_col="target", feature_cols=["age"], cache_strategy="memory")
        assert model.metadata["n_samples"] == 120


# --- Item 6: global settings -------------------------------------------------------


class TestSeeds:
    def test_global_seed_makes_two_fits_identical(self, frame):
        smallaxe.set_seed(42)
        first = regressors.random_forest(n_estimators=5)
        first.fit(frame, label_col="target", feature_cols=["age", "income"], validation="kfold")
        second = regressors.random_forest(n_estimators=5)
        second.fit(frame, label_col="target", feature_cols=["age", "income"], validation="kfold")

        assert _predictions(first, frame) == _predictions(second, frame)
        assert first.validation_scores == second.validation_scores
        assert first.metadata["seed"] == 42
        assert first.metadata["split_seed"] == 42

    def test_model_seed_beats_global_seed(self, frame):
        smallaxe.set_seed(42)
        model = regressors.random_forest(n_estimators=2, seed=7)
        model.fit(frame, label_col="target", feature_cols=["age"], validation="train_test")
        assert model.metadata["seed"] == 7
        assert model.metadata["split_seed"] == 7

    def test_unseeded_fit_records_the_seed_it_drew(self, frame):
        model = regressors.random_forest(n_estimators=2)
        model.fit(frame, label_col="target", feature_cols=["age"], validation="train_test")
        assert model.metadata["seed"] is None
        assert isinstance(model.metadata["split_seed"], int)

    def test_no_validation_records_no_split_seed(self, frame):
        model = regressors.random_forest(n_estimators=2, seed=1)
        model.fit(frame, label_col="target", feature_cols=["age"])
        assert model.metadata["split_seed"] is None


class TestSearchSettings:
    def test_global_seed_and_quiet_verbosity_reach_the_search(self, frame, monkeypatch):
        pytest.importorskip("hyperopt")
        from smallaxe.search import optimize

        captured = {}
        real_fmin = optimize.fmin

        def spy(**kwargs):
            captured.update(kwargs)
            return real_fmin(**kwargs)

        monkeypatch.setattr(optimize, "fmin", spy)
        smallaxe.set_seed(3)
        with smallaxe.verbosity("quiet"):
            optimize.run(
                regressors.random_forest(n_estimators=2),
                frame,
                label_col="target",
                param_space={"max_depth": [2, 3]},
                feature_cols=["age"],
                max_evals=1,
            )
        assert captured["show_progressbar"] is False
        assert captured["rstate"] is not None

    @pytest.mark.parametrize("validation", ["train_test", "kfold"])
    def test_unseeded_search_scores_every_trial_on_the_same_split(
        self, frame, monkeypatch, validation
    ):
        """With no seed anywhere, the search draws one split seed for all its trials."""
        pytest.importorskip("hyperopt")
        from smallaxe.search import optimize
        from smallaxe.training.base import BaseModel

        assert smallaxe.get_seed() is None
        trial_models = []
        original = BaseModel.fit

        def spy(self, df, *args, **kwargs):
            result = original(self, df, *args, **kwargs)
            if kwargs.get("validation") != "none":
                trial_models.append(self)
            return result

        monkeypatch.setattr(BaseModel, "fit", spy)
        optimize.run(
            regressors.random_forest(n_estimators=2),
            frame,
            label_col="target",
            param_space={"max_depth": [2, 3, 4]},
            feature_cols=["age"],
            validation=validation,
            n_folds=2,
            max_evals=4,
            verbose=False,
        )

        split_seeds = {model.metadata["split_seed"] for model in trial_models}
        assert len(trial_models) == 4
        assert len(split_seeds) == 1
        assert isinstance(split_seeds.pop(), int)
        # Estimator seeding is unchanged: an unseeded model stays unseeded.
        assert all(model.metadata["seed"] is None for model in trial_models)

    def test_best_model_does_not_keep_the_search_split_seed(self, frame, monkeypatch):
        """The search's split seed is for its trials; a later fit of best_model draws anew."""
        pytest.importorskip("hyperopt")
        from smallaxe.search import optimize
        from smallaxe.training.mixins import validation_mixin

        assert smallaxe.get_seed() is None
        result = optimize.run(
            regressors.random_forest(n_estimators=2),
            frame,
            label_col="target",
            param_space={"max_depth": [2, 3]},
            feature_cols=["age"],
            validation="kfold",
            n_folds=2,
            max_evals=2,
            verbose=False,
        )
        best = result.best_model
        assert best._split_seed_fallback is None

        monkeypatch.setattr(validation_mixin.random, "randrange", lambda _: 12345)
        best.fit(frame, "target", feature_cols=["age"], validation="kfold", n_folds=2)
        assert best.metadata["split_seed"] == 12345

    def test_train_test_search_persists_the_data_once(self, frame, monkeypatch):
        """Every trial reads the data, so the search caches it whatever the validation."""
        pytest.importorskip("hyperopt")
        from smallaxe.search import optimize

        persisted = []
        original = type(frame).persist

        def spy(self, *args, **kwargs):
            persisted.append(1)
            return original(self, *args, **kwargs)

        monkeypatch.setattr(type(frame), "persist", spy)
        optimize.run(
            regressors.random_forest(n_estimators=2, seed=1),
            frame,
            label_col="target",
            param_space={"max_depth": [2, 3]},
            feature_cols=["age"],
            validation="train_test",
            max_evals=2,
            refit=False,
            verbose=False,
        )
        assert persisted == [1]

    def test_trials_do_not_refit_and_best_model_sees_all_rows(self, frame, monkeypatch):
        pytest.importorskip("hyperopt")
        from smallaxe.search import optimize
        from smallaxe.training.base import BaseModel

        fits = []
        original = BaseModel.fit

        def spy(self, df, *args, **kwargs):
            fits.append((kwargs.get("validation"), kwargs.get("refit")))
            return original(self, df, *args, **kwargs)

        monkeypatch.setattr(BaseModel, "fit", spy)
        result = optimize.run(
            regressors.random_forest(n_estimators=2, seed=1),
            frame,
            label_col="target",
            param_space={"max_depth": [2, 3]},
            feature_cols=["age"],
            validation="kfold",
            n_folds=2,
            max_evals=2,
            verbose=False,
        )
        assert fits[:2] == [("kfold", False), ("kfold", False)]
        assert fits[2] == ("none", None)
        assert result.best_model.metadata["kept_model"] == "full"


# --- Item 7: feature inference -----------------------------------------------------


class TestFeatureInference:
    def test_boolean_is_a_feature_and_string_is_reported_once(self, frame):
        model = classifiers.random_forest(n_estimators=2, seed=1)
        with pytest.warns(UserWarning, match="customer \\(string\\)") as record:
            model.fit(frame, label_col="label", exclude_cols=["id", "target"])

        inference_warnings = [w for w in record if "Feature inference" in str(w.message)]
        assert len(inference_warnings) == 1
        assert model.metadata["feature_cols"] == ["active", "age", "income"]
        assert model.metadata["feature_cols_inferred"] is True
        assert model.metadata["ignored_cols"] == {"customer": "string"}

    def test_numeric_and_boolean_only_gives_no_warning(self, frame):
        model = regressors.random_forest(n_estimators=2, seed=1)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            model.fit(frame.drop("customer"), label_col="target", exclude_cols=["id", "label"])
        assert model.metadata["ignored_cols"] == {}

    def test_excluded_columns_are_not_reported(self, frame):
        model = regressors.random_forest(n_estimators=2, seed=1)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            model.fit(frame, label_col="target", exclude_cols=["id", "customer", "label"])
        assert model.metadata["feature_cols_inferred"] is True

    def test_explicit_feature_cols_are_not_inferred(self, frame):
        model = regressors.random_forest(n_estimators=2, seed=1)
        model.fit(frame, label_col="target", feature_cols=["age"])
        assert model.metadata["feature_cols_inferred"] is False

    def test_search_warns_at_most_once(self, frame):
        pytest.importorskip("hyperopt")
        from smallaxe.search import optimize

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            optimize.run(
                regressors.random_forest(n_estimators=2, seed=1),
                frame,
                label_col="target",
                param_space={"max_depth": [2, 3]},
                exclude_cols=["id", "label"],
                max_evals=3,
                verbose=False,
            )
        inference_warnings = [w for w in record if "Feature inference" in str(w.message)]
        assert len(inference_warnings) == 1
