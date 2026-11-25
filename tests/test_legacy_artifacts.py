"""Artifacts saved by released smallaxe versions must still load and predict identically.

``tests/fixtures/artifacts_<version>`` holds tiny Random Forest and XGBoost models saved
by that release plus the predictions it produced for the frame below:

- ``0_8_1``: ``metadata.json`` records the old class name, not ``algorithm``.
- ``0_9_0``, ``0_10_0``: record ``algorithm`` and ``task``, but no ``format_version``
  or ``kind`` (written by ``tests/fixtures/make_model_artifacts.py`` with the wheels
  from PyPI).
- ``format_1``: the current format (``format_version`` 1), so it stays readable.
"""

import json
import shutil
import warnings
from pathlib import Path

import pytest

import smallaxe
from smallaxe import _artifact
from smallaxe.training import algorithm, classifiers, regressors
from smallaxe.training.base import Classifier, Regressor

FIXTURES = Path(__file__).parent / "fixtures"
LEGACY_VERSIONS = ["0.8.1", "0.9.0", "0.10.0"]
NAMES = [
    "random_forest_regressor",
    "random_forest_classifier",
    "xgboost_regressor",
    "xgboost_classifier",
]
TASKS = {
    "random_forest_regressor": "simple_regression",
    "random_forest_classifier": "multiclass",
    "xgboost_regressor": "simple_regression",
    "xgboost_classifier": "binary",
}
# Before 0.11 XGBoost trained its default 100 rounds whatever n_estimators said.
XGBOOST_ROUNDS = {"0.8.1": 100, "0.9.0": 100, "0.10.0": 100, "format_1": 5}


def _directory(version):
    return FIXTURES / ("artifacts_" + version.replace(".", "_"))


def _expected(version):
    return json.loads((_directory(version) / "expected_predictions.json").read_text())


CASES = [(version, name) for version in LEGACY_VERSIONS + ["format_1"] for name in NAMES]
LEGACY_CASES = [case for case in CASES if case[0] != "format_1"]


def _ids(case):
    return f"{case[0]}-{case[1]}"


@pytest.fixture(scope="module")
def frame(spark_session):
    rows = [
        (i, 20.0 + (i % 30), 40000.0 + i * 1000.0, 50.0 + i * 5 + (i % 7), i % 2, i % 3)
        for i in range(1, 121)
    ]
    return spark_session.createDataFrame(
        rows, ["id", "age", "income", "target", "binary", "multiclass"]
    )


def _require(name):
    algorithm_name = name.rsplit("_", 1)[0]
    if not algorithm.get(algorithm_name).available():
        pytest.skip(f"{algorithm_name} is not installed")
    return algorithm_name


def _load(version, name):
    """Load a fixture through its factory."""
    _require(name)
    factory = regressors if name.endswith("regressor") else classifiers
    return factory.load(str(_directory(version) / name))


def _predictions(model, frame):
    rows = model.predict(frame).select("id", "predict_label").orderBy("id").collect()
    return [[row["id"], float(row["predict_label"])] for row in rows]


@pytest.mark.parametrize("version", LEGACY_VERSIONS)
def test_legacy_fixture_was_written_by_that_release(version):
    for name in NAMES:
        metadata = json.loads((_directory(version) / name / "metadata.json").read_text())
        assert "format_version" not in metadata and "kind" not in metadata
        assert metadata["metadata"]["smallaxe_version"] == version
        assert metadata["task"] == TASKS[name]
        if version == "0.8.1":
            assert "algorithm" not in metadata, "fixture must be a real 0.8.x artifact"
        else:
            assert metadata["algorithm"] == name.rsplit("_", 1)[0]


def test_current_format_fixture_has_the_header():
    for name in NAMES:
        metadata = json.loads((_directory("format_1") / name / "metadata.json").read_text())
        assert metadata["format_version"] == 1
        assert metadata["kind"] == "model"
        assert metadata["algorithm"] == name.rsplit("_", 1)[0]


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_fixture_loads_and_predicts_identically(frame, case):
    version, name = case
    algorithm_name = _require(name)

    for model in (_load(version, name), smallaxe.load_model(str(_directory(version) / name))):
        assert type(model) is (Regressor if name.endswith("regressor") else Classifier)
        assert model.algorithm.name == algorithm_name
        assert model.task == TASKS[name]
        assert model._is_fitted
        assert model.get_param("n_estimators") == 5
        assert _predictions(model, frame) == _expected(version)[name]["predictions"]


@pytest.mark.parametrize("case", LEGACY_CASES, ids=_ids)
def test_resaved_legacy_artifact_is_versioned_and_predicts_identically(frame, case, tmp_path):
    version, name = case
    algorithm_name = _require(name)
    model = smallaxe.load_model(str(_directory(version) / name))
    path = tmp_path / name

    model.save(str(path))

    metadata = json.loads((path / "metadata.json").read_text())
    assert metadata["format_version"] == _artifact.FORMAT_VERSION
    assert metadata["kind"] == "model"
    assert metadata["smallaxe_version"] == smallaxe.__version__
    assert metadata["algorithm"] == algorithm_name
    assert metadata["task"] == TASKS[name]
    assert metadata["__class__"] == type(model).__name__
    # The nested metadata still says which release trained the model.
    assert metadata["metadata"]["smallaxe_version"] == version
    for reloaded in (smallaxe.load_model(str(path)), type(model).load(str(path))):
        assert type(reloaded) is type(model)
        assert _predictions(reloaded, frame) == _expected(version)[name]["predictions"]


@pytest.mark.parametrize("version", sorted(XGBOOST_ROUNDS))
@pytest.mark.parametrize("name", ["xgboost_regressor", "xgboost_classifier"])
def test_xgboost_fixture_keeps_the_rounds_it_was_trained_with(version, name):
    """Up to 0.10 n_estimators reached xgboost.spark as num_round, which it ignored."""
    model = _load(version, name)

    assert model.get_param("n_estimators") == 5
    assert model._spark_model.get_booster().num_boosted_rounds() == XGBOOST_ROUNDS[version]


def _multiclass_artifact_with_scores(tmp_path, validation_scores):
    path = tmp_path / "random_forest_classifier"
    shutil.copytree(_directory("0.8.1") / "random_forest_classifier", path)
    metadata = json.loads((path / "metadata.json").read_text())
    assert metadata["task"] == "multiclass"
    metadata["validation_scores"] = validation_scores
    (path / "metadata.json").write_text(json.dumps(metadata))
    return str(path)


@pytest.mark.parametrize(
    "scores",
    [
        {"accuracy": 0.5, "precision": 0.4, "recall": 0.3, "f1_score": 0.35},
        {"mean_accuracy": 0.5, "mean_precision": 0.4, "fold_scores": []},
    ],
    ids=["train_test", "kfold"],
)
def test_class1_multiclass_scores_warn_on_load(tmp_path, scores):
    path = _multiclass_artifact_with_scores(tmp_path, scores)

    with pytest.warns(UserWarning, match="class 1 only") as record:
        model = classifiers.load(path)

    assert record[0].filename == __file__
    assert model.validation_scores == scores


def test_class1_multiclass_scores_warn_through_load_model(tmp_path):
    path = _multiclass_artifact_with_scores(tmp_path, {"precision": 0.4})

    with pytest.warns(UserWarning, match="class 1 only") as record:
        smallaxe.load_model(path)

    assert record[0].filename == __file__


@pytest.mark.parametrize(
    "scores",
    [
        None,
        {"precision": 0.4, "precision_macro": 0.4, "precision_weighted": 0.5},
        {"mean_precision": 0.4, "mean_precision_macro": 0.4, "fold_scores": []},
    ],
    ids=["no-validation", "train_test", "kfold"],
)
def test_macro_multiclass_scores_load_without_warning(tmp_path, scores):
    path = _multiclass_artifact_with_scores(tmp_path, scores)

    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*class 1 only")
        classifiers.load(path)
