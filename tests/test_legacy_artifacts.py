"""Artifacts saved by smallaxe 0.8.1 must still load and predict identically.

``tests/fixtures/artifacts_0_8_1`` holds tiny Random Forest and XGBoost models saved by
the released 0.8.1 (their ``metadata.json`` records the old class name, not
``algorithm``) plus the predictions 0.8.1 produced for the frame below.
"""

import json
from pathlib import Path

import pytest

from smallaxe.training import Classifiers, Regressors, algorithm

FIXTURES = Path(__file__).parent / "fixtures" / "artifacts_0_8_1"
EXPECTED = json.loads((FIXTURES / "expected_predictions.json").read_text())


@pytest.fixture(scope="module")
def frame(spark_session):
    rows = [
        (i, 20.0 + (i % 30), 40000.0 + i * 1000.0, 50.0 + i * 5 + (i % 7), i % 2, i % 3)
        for i in range(1, 121)
    ]
    return spark_session.createDataFrame(
        rows, ["id", "age", "income", "target", "binary", "multiclass"]
    )


def _load(name):
    algorithm_name, kind = name.rsplit("_", 1)
    if not algorithm.get(algorithm_name).available():
        pytest.skip(f"{algorithm_name} is not installed")
    factory = Regressors if kind == "regressor" else Classifiers
    return factory.load(str(FIXTURES / name)), algorithm_name


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_legacy_artifact_loads_and_predicts_identically(frame, name):
    metadata = json.loads((FIXTURES / name / "metadata.json").read_text())
    assert "algorithm" not in metadata, "fixture must be a real 0.8.x artifact"

    model, algorithm_name = _load(name)

    assert model.algorithm.name == algorithm_name
    assert model._is_fitted
    assert model.get_param("n_estimators") == 5
    predictions = model.predict(frame).select("id", "predict_label").orderBy("id").collect()
    actual = [[row["id"], float(row["predict_label"])] for row in predictions]
    assert actual == EXPECTED[name]["predictions"]


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_resaved_legacy_artifact_uses_the_new_format(frame, name, tmp_path):
    model, algorithm_name = _load(name)
    path = str(tmp_path / name)

    model.save(path)

    metadata = json.loads((tmp_path / name / "metadata.json").read_text())
    assert metadata["algorithm"] == algorithm_name
    assert metadata["__class__"] == type(model).__name__
    reloaded = type(model).load(path)
    assert reloaded.predict(frame).count() == frame.count()
