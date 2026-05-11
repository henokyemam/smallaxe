"""Tests for CatBoostRegressor and CatBoostClassifier."""

import pytest

from smallaxe.exceptions import DependencyError, ValidationError

try:
    import catboost_spark  # noqa: F401

    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


@pytest.mark.skipif(not CATBOOST_AVAILABLE, reason="catboost_spark not installed")
class TestCatBoostRegressorInit:
    """Tests for CatBoostRegressor initialization."""

    def test_default_task(self):
        """Test that default task is 'simple_regression'."""
        from smallaxe.training.catboost import CatBoostRegressor

        model = CatBoostRegressor()
        assert model.task == "simple_regression"
        assert model.task_type == "regression"

    def test_invalid_task_raises_error(self):
        """Test that invalid task raises ValidationError."""
        from smallaxe.training.catboost import CatBoostRegressor

        with pytest.raises(ValidationError, match="Invalid regression task"):
            CatBoostRegressor(task="binary")


@pytest.mark.skipif(not CATBOOST_AVAILABLE, reason="catboost_spark not installed")
class TestCatBoostRegressorParams:
    """Tests for CatBoostRegressor params."""

    def test_params_dict(self):
        """Test that params returns parameter descriptions."""
        from smallaxe.training.catboost import CatBoostRegressor

        model = CatBoostRegressor()
        params = model.params

        assert "n_estimators" in params
        assert "max_depth" in params
        assert "learning_rate" in params
        assert "subsample" in params
        assert "l2_leaf_reg" in params
        assert "random_strength" in params
        assert "one_hot_max_size" in params
        assert "allow_writing_files" in params
        assert "train_dir" in params
        assert "seed" in params

    def test_default_params_dict(self):
        """Test that default_params returns default values."""
        from smallaxe.training.catboost import CatBoostRegressor

        model = CatBoostRegressor()
        defaults = model.default_params

        assert defaults["n_estimators"] == 100
        assert defaults["max_depth"] == 6
        assert defaults["learning_rate"] == 0.03
        assert defaults["subsample"] is None
        assert defaults["l2_leaf_reg"] == 3.0
        assert defaults["random_strength"] == 1.0
        assert defaults["one_hot_max_size"] is None
        assert defaults["allow_writing_files"] is False
        assert defaults["train_dir"] is None
        assert defaults["seed"] is None

    def test_set_param_multiple(self):
        """Test setting multiple parameters."""
        from smallaxe.training.catboost import CatBoostRegressor

        model = CatBoostRegressor()
        model.set_param({"n_estimators": 50, "max_depth": 4, "learning_rate": 0.1})

        assert model.get_param("n_estimators") == 50
        assert model.get_param("max_depth") == 4
        assert model.get_param("learning_rate") == 0.1

    def test_set_param_invalid_key(self):
        """Test that invalid parameter key raises ValidationError."""
        from smallaxe.training.catboost import CatBoostRegressor

        model = CatBoostRegressor()
        with pytest.raises(ValidationError, match="Invalid parameter"):
            model.set_param({"invalid_param": 10})


@pytest.mark.skipif(not CATBOOST_AVAILABLE, reason="catboost_spark not installed")
class TestCatBoostClassifierInit:
    """Tests for CatBoostClassifier initialization."""

    def test_default_task(self):
        """Test that default task is 'binary'."""
        from smallaxe.training.catboost import CatBoostClassifier

        model = CatBoostClassifier()
        assert model.task == "binary"
        assert model.task_type == "classification"

    def test_multiclass_task(self):
        """Test that multiclass task is accepted."""
        from smallaxe.training.catboost import CatBoostClassifier

        model = CatBoostClassifier(task="multiclass")
        assert model.task == "multiclass"

    def test_invalid_task_raises_error(self):
        """Test that invalid task raises ValidationError."""
        from smallaxe.training.catboost import CatBoostClassifier

        with pytest.raises(ValidationError, match="Invalid classification task"):
            CatBoostClassifier(task="simple_regression")


@pytest.mark.skipif(not CATBOOST_AVAILABLE, reason="catboost_spark not installed")
class TestCatBoostClassifierParams:
    """Tests for CatBoostClassifier params."""

    def test_params_dict(self):
        """Test that params returns parameter descriptions."""
        from smallaxe.training.catboost import CatBoostClassifier

        model = CatBoostClassifier()
        params = model.params

        assert "n_estimators" in params
        assert "max_depth" in params
        assert "learning_rate" in params
        assert "subsample" in params
        assert "l2_leaf_reg" in params
        assert "random_strength" in params
        assert "one_hot_max_size" in params
        assert "scale_pos_weight" in params
        assert "allow_writing_files" in params
        assert "train_dir" in params
        assert "seed" in params

    def test_default_params_dict(self):
        """Test that default_params returns default values."""
        from smallaxe.training.catboost import CatBoostClassifier

        model = CatBoostClassifier()
        defaults = model.default_params

        assert defaults["n_estimators"] == 100
        assert defaults["max_depth"] == 6
        assert defaults["learning_rate"] == 0.03
        assert defaults["subsample"] is None
        assert defaults["l2_leaf_reg"] == 3.0
        assert defaults["random_strength"] == 1.0
        assert defaults["one_hot_max_size"] is None
        assert defaults["scale_pos_weight"] is None
        assert defaults["allow_writing_files"] is False
        assert defaults["train_dir"] is None
        assert defaults["seed"] is None

    def test_set_param_multiple(self):
        """Test setting multiple parameters."""
        from smallaxe.training.catboost import CatBoostClassifier

        model = CatBoostClassifier()
        model.set_param({"n_estimators": 50, "max_depth": 4, "learning_rate": 0.1})

        assert model.get_param("n_estimators") == 50
        assert model.get_param("max_depth") == 4
        assert model.get_param("learning_rate") == 0.1


# =============================================================================
# DependencyError Tests (run always, even without CatBoost)
# =============================================================================


@pytest.mark.skipif(
    CATBOOST_AVAILABLE,
    reason="Test only runs when catboost_spark is NOT installed",
)
class TestCatBoostDependencyError:
    """Tests for DependencyError when catboost_spark is not installed."""

    def test_regressor_raises_dependency_error(self):
        """Test that CatBoostRegressor raises DependencyError when unavailable."""
        from smallaxe.training.catboost import CatBoostRegressor

        with pytest.raises(DependencyError, match="catboost_spark is not installed"):
            CatBoostRegressor()

    def test_classifier_raises_dependency_error(self):
        """Test that CatBoostClassifier raises DependencyError when unavailable."""
        from smallaxe.training.catboost import CatBoostClassifier

        with pytest.raises(DependencyError, match="catboost_spark is not installed"):
            CatBoostClassifier()
