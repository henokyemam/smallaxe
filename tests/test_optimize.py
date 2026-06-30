"""Tests for smallaxe.search.optimize (hyperparameter optimization)."""

import pytest

from smallaxe.exceptions import DependencyError, ValidationError
from smallaxe.search import optimize
from smallaxe.search.optimize import SearchResult
from smallaxe.training import Classifiers, Regressors
from smallaxe.training.xgboost import XGBOOST_AVAILABLE

hp = pytest.importorskip("hyperopt").hp


@pytest.fixture
def regression_df(spark_session):
    """Synthetic regression data with a learnable signal."""
    data = []
    for i in range(160):
        x1 = float(i % 20)
        x2 = float((i * 7) % 13)
        target = 3.0 * x1 - 2.0 * x2 + (i % 5)
        data.append((i, x1, x2, float(target)))
    return spark_session.createDataFrame(data, ["id", "x1", "x2", "target"])


@pytest.fixture
def classification_df(spark_session):
    """Synthetic binary classification data with separable classes."""
    data = []
    for i in range(160):
        x1 = float(i % 20)
        x2 = float((i * 3) % 11)
        label = 1 if (x1 - x2) > 3 else 0
        data.append((i, x1, x2, int(label)))
    return spark_session.createDataFrame(data, ["id", "x1", "x2", "label"])


@pytest.fixture
def seeded():
    """Set a global smallaxe seed for deterministic splits, then restore."""
    import smallaxe

    previous = smallaxe.get_seed()
    smallaxe.set_seed(42)
    yield
    smallaxe.set_seed(previous)


class TestDefaultMetric:
    """Tests for the default metric chosen per task type."""

    def test_regression_defaults_to_rmse(self):
        assert optimize._default_metric(Regressors.random_forest()) == "rmse"

    def test_binary_defaults_to_auc_roc(self):
        model = Classifiers.random_forest(task="binary")
        assert optimize._default_metric(model) == "auc_roc"

    def test_multiclass_defaults_to_accuracy(self):
        # f1/precision/recall are binary-only; multiclass must use accuracy.
        model = Classifiers.random_forest(task="multiclass")
        assert optimize._default_metric(model) == "accuracy"


class TestOptimizeDependencyGating:
    """Tests for hyperopt dependency handling."""

    def test_run_raises_dependency_error_without_hyperopt(self, monkeypatch, regression_df):
        """run() raises DependencyError with an install hint when hyperopt is absent."""
        monkeypatch.setattr(optimize, "HYPEROPT_AVAILABLE", False)
        model = Regressors.random_forest(seed=42)
        with pytest.raises(DependencyError, match="hyperopt"):
            optimize.run(
                model,
                regression_df,
                label_col="target",
                param_space={"max_depth": [3, 5]},
                max_evals=2,
            )


class TestOptimizeArgumentValidation:
    """Tests for argument validation in run()."""

    def test_invalid_metric_raises(self, regression_df):
        model = Regressors.random_forest(seed=42)
        with pytest.raises(ValidationError, match="Invalid metric"):
            optimize.run(
                model,
                regression_df,
                label_col="target",
                param_space={"max_depth": [3, 5]},
                metric="not_a_metric",
                max_evals=2,
            )

    def test_validation_none_rejected(self, regression_df):
        model = Regressors.random_forest(seed=42)
        with pytest.raises(ValidationError, match="train_test' or 'kfold'"):
            optimize.run(
                model,
                regression_df,
                label_col="target",
                param_space={"max_depth": [3, 5]},
                validation="none",
                max_evals=2,
            )

    def test_empty_param_space_rejected(self, regression_df):
        model = Regressors.random_forest(seed=42)
        with pytest.raises(ValidationError, match="non-empty dict"):
            optimize.run(
                model,
                regression_df,
                label_col="target",
                param_space={},
                max_evals=2,
            )

    def test_max_evals_below_one_rejected(self, regression_df):
        model = Regressors.random_forest(seed=42)
        with pytest.raises(ValidationError, match="max_evals"):
            optimize.run(
                model,
                regression_df,
                label_col="target",
                param_space={"max_depth": [3, 5]},
                max_evals=0,
            )

    def test_all_failed_trials_raises(self, regression_df):
        """A param_space referencing an invalid parameter fails every trial."""
        model = Regressors.random_forest(seed=42)
        with pytest.raises(ValidationError, match="All trials failed"):
            optimize.run(
                model,
                regression_df,
                label_col="target",
                param_space={"not_a_real_param": [1, 2]},
                max_evals=3,
                verbose=False,
            )


class TestOptimizeRegression:
    """Tests for optimizing a regressor."""

    def test_returns_search_result(self, regression_df, seeded):
        model = Regressors.random_forest(seed=42)
        result = optimize.run(
            model,
            regression_df,
            label_col="target",
            param_space={
                "n_estimators": hp.quniform("n_estimators", 10, 40, 10),
                "max_depth": hp.quniform("max_depth", 2, 6, 1),
            },
            metric="rmse",
            validation="train_test",
            max_evals=5,
            exclude_cols=["id"],
            seed=42,
            verbose=False,
        )
        assert isinstance(result, SearchResult)
        assert result.metric == "rmse"
        assert "n_estimators" in result.best_params
        assert "max_depth" in result.best_params

    def test_searched_int_params_are_ints(self, regression_df, seeded):
        """quniform yields floats; the result must be cast back to int."""
        model = Regressors.random_forest(seed=42)
        result = optimize.run(
            model,
            regression_df,
            label_col="target",
            param_space={
                "n_estimators": hp.quniform("n_estimators", 10, 40, 10),
                "max_depth": hp.quniform("max_depth", 2, 6, 1),
            },
            metric="rmse",
            max_evals=4,
            exclude_cols=["id"],
            seed=42,
            verbose=False,
        )
        assert isinstance(result.best_params["n_estimators"], int)
        assert isinstance(result.best_params["max_depth"], int)

    def test_best_model_is_fitted_and_predicts(self, regression_df, seeded):
        model = Regressors.random_forest(seed=42)
        result = optimize.run(
            model,
            regression_df,
            label_col="target",
            param_space={"max_depth": hp.quniform("max_depth", 2, 6, 1)},
            metric="rmse",
            max_evals=3,
            exclude_cols=["id"],
            seed=42,
            verbose=False,
        )
        assert result.best_model is not None
        preds = result.best_model.predict(regression_df)
        assert "predict_label" in preds.columns
        assert preds.count() == regression_df.count()

    def test_trials_history_complete(self, regression_df, seeded):
        model = Regressors.random_forest(seed=42)
        result = optimize.run(
            model,
            regression_df,
            label_col="target",
            param_space={"max_depth": hp.quniform("max_depth", 2, 6, 1)},
            metric="rmse",
            max_evals=4,
            exclude_cols=["id"],
            seed=42,
            verbose=False,
        )
        assert len(result.trials_history) == 4
        assert result.n_trials == 4
        assert result.n_successful_trials == 4
        for trial in result.trials_history:
            assert trial["status"] == "ok"
            assert trial["params"] is not None
            assert isinstance(trial["score"], float)

    def test_best_score_is_minimum_for_minimize_metric(self, regression_df, seeded):
        """For rmse (lower is better), best_score must be the smallest score."""
        model = Regressors.random_forest(seed=42)
        result = optimize.run(
            model,
            regression_df,
            label_col="target",
            param_space={"max_depth": hp.quniform("max_depth", 2, 8, 1)},
            metric="rmse",
            max_evals=5,
            exclude_cols=["id"],
            seed=42,
            verbose=False,
        )
        scores = [t["score"] for t in result.trials_history if t["score"] is not None]
        assert result.best_score == pytest.approx(min(scores))

    def test_refit_false_yields_no_model(self, regression_df, seeded):
        model = Regressors.random_forest(seed=42)
        result = optimize.run(
            model,
            regression_df,
            label_col="target",
            param_space={"max_depth": [3, 5]},
            metric="rmse",
            max_evals=2,
            exclude_cols=["id"],
            seed=42,
            refit=False,
            verbose=False,
        )
        assert result.best_model is None


class TestOptimizeClassification:
    """Tests for optimizing a classifier."""

    def test_kfold_auc_roc(self, classification_df, seeded):
        model = Classifiers.random_forest(task="binary", seed=42)
        result = optimize.run(
            model,
            classification_df,
            label_col="label",
            param_space={
                "n_estimators": hp.quniform("n_estimators", 10, 40, 10),
                "max_depth": [3, 5, 7],
            },
            metric="auc_roc",
            validation="kfold",
            n_folds=3,
            max_evals=4,
            exclude_cols=["id"],
            seed=42,
            verbose=False,
        )
        assert result.metric == "auc_roc"
        assert 0.0 <= result.best_score <= 1.0

    def test_best_score_is_maximum_for_maximize_metric(self, classification_df, seeded):
        """For auc_roc (higher is better), best_score must be the largest score."""
        model = Classifiers.random_forest(task="binary", seed=42)
        result = optimize.run(
            model,
            classification_df,
            label_col="label",
            param_space={"max_depth": [3, 5, 7]},
            metric="auc_roc",
            validation="train_test",
            max_evals=4,
            exclude_cols=["id"],
            seed=42,
            verbose=False,
        )
        scores = [t["score"] for t in result.trials_history if t["score"] is not None]
        assert result.best_score == pytest.approx(max(scores))

    def test_choice_list_value_is_from_list(self, classification_df, seeded):
        """A list param_space is treated as discrete choices."""
        model = Classifiers.random_forest(task="binary", seed=42)
        result = optimize.run(
            model,
            classification_df,
            label_col="label",
            param_space={"max_depth": [3, 5, 7]},
            metric="f1_score",
            max_evals=4,
            exclude_cols=["id"],
            seed=42,
            verbose=False,
        )
        assert result.best_params["max_depth"] in (3, 5, 7)


class TestOptimizeDeterminism:
    """Tests that searches are reproducible with a fixed seed."""

    def test_same_seed_same_best_params(self, regression_df, seeded):
        space = {
            "n_estimators": hp.quniform("n_estimators", 10, 40, 10),
            "max_depth": hp.quniform("max_depth", 2, 6, 1),
        }
        kwargs = dict(
            label_col="target",
            metric="rmse",
            validation="train_test",
            max_evals=5,
            exclude_cols=["id"],
            seed=7,
            verbose=False,
        )
        first = optimize.run(
            Regressors.random_forest(seed=42), regression_df, param_space=dict(space), **kwargs
        )
        second = optimize.run(
            Regressors.random_forest(seed=42), regression_df, param_space=dict(space), **kwargs
        )
        assert first.best_params == second.best_params
        assert first.best_score == pytest.approx(second.best_score)

    def test_template_model_not_mutated(self, regression_df, seeded):
        """The model passed to run() is used as a template and not modified."""
        model = Regressors.random_forest(n_estimators=20, max_depth=5, seed=42)
        before = model.get_params()
        optimize.run(
            model,
            regression_df,
            label_col="target",
            param_space={"max_depth": hp.quniform("max_depth", 2, 8, 1)},
            metric="rmse",
            max_evals=3,
            exclude_cols=["id"],
            seed=42,
            verbose=False,
        )
        assert model.get_params() == before


class TestOptimizeEarlyStopping:
    """Tests for the early stopping option."""

    def test_early_stopping_runs(self, regression_df, seeded):
        """Early stopping should not error and should cap the trial count."""
        model = Regressors.random_forest(seed=42)
        result = optimize.run(
            model,
            regression_df,
            label_col="target",
            param_space={"max_depth": hp.quniform("max_depth", 2, 6, 1)},
            metric="rmse",
            max_evals=20,
            exclude_cols=["id"],
            seed=42,
            early_stopping=True,
            early_stopping_rounds=3,
            verbose=False,
        )
        assert result.n_trials <= 20
        assert result.best_model is not None


@pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="xgboost not installed")
class TestOptimizeXGBoost:
    """Cross-model coverage: optimizing an XGBoost regressor via the factory."""

    def test_xgboost_regression_search(self, regression_df, seeded):
        model = Regressors.xgboost(seed=42)
        result = optimize.run(
            model,
            regression_df,
            label_col="target",
            param_space={
                "n_estimators": hp.quniform("n_estimators", 20, 60, 20),
                "max_depth": hp.quniform("max_depth", 3, 6, 1),
                "learning_rate": hp.uniform("learning_rate", 0.05, 0.3),
            },
            metric="rmse",
            validation="train_test",
            max_evals=4,
            exclude_cols=["id"],
            seed=42,
            verbose=False,
        )
        assert isinstance(result.best_params["n_estimators"], int)
        assert isinstance(result.best_params["learning_rate"], float)
        assert result.best_model is not None
        preds = result.best_model.predict(regression_df)
        assert "predict_label" in preds.columns
