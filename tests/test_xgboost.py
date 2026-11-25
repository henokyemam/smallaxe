"""XGBoost behaviour checked against the real ``xgboost.spark`` estimators.

Skipped where xgboost is not installed. The translation of every record against
fake estimators is in ``test_algorithm.py``.
"""

import pytest

from smallaxe.training import Classifiers, Regressors, algorithm

pytestmark = pytest.mark.skipif(
    not algorithm.get("xgboost").available(), reason="xgboost is not installed"
)

FEATURES = ["a", "b"]


@pytest.fixture(scope="module")
def frame(spark_session):
    rows = [
        (i, float(i % 17), float((i * 7) % 13), float(i % 5) + (i % 3) * 0.5, i % 2, i % 3)
        for i in range(200)
    ]
    return spark_session.createDataFrame(rows, ["id", "a", "b", "y", "binary", "multiclass"])


def _fit(frame, task, **params):
    if task == "simple_regression":
        model = Regressors.xgboost(**params)
        label = "y"
    else:
        model = Classifiers.xgboost(task=task, **params)
        label = task
    return model.fit(frame, label_col=label, feature_cols=FEATURES)


def _rounds(model) -> int:
    return model._spark_model.get_booster().num_boosted_rounds()


@pytest.mark.parametrize("task", ["simple_regression", "binary", "multiclass"])
def test_n_estimators_sets_the_number_of_boosting_rounds(frame, task):
    assert _rounds(_fit(frame, task, n_estimators=7)) == 7


@pytest.mark.parametrize("task", ["simple_regression", "binary"])
def test_default_trains_one_hundred_rounds(frame, task):
    assert _rounds(_fit(frame, task)) == 100


def _predictions(frame, seed):
    model = _fit(frame, "simple_regression", n_estimators=7, seed=seed, subsample=0.5)
    rows = model.predict(frame).select("id", "predict_label").orderBy("id").collect()
    return [row["predict_label"] for row in rows]


def test_seed_controls_training(frame):
    """With subsampling, the same seed reproduces the model and another seed does not.

    ``seed`` is not a declared xgboost.spark param: it reaches the booster through
    ``arbitrary_params_dict`` as XGBoost's native ``seed``.
    """
    first = _predictions(frame, seed=1)
    assert _predictions(frame, seed=1) == first
    assert _predictions(frame, seed=2) != first
