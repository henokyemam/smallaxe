"""The versioned model artifact: what ``save`` writes and what loading rejects.

A model artifact is a directory with ``spark_model/`` and a ``metadata.json`` that
carries the header (``format_version``, ``kind`` = ``"model"``, ``smallaxe_version``)
next to the model's own keys. ``metadata.json`` is written last, so a directory that
has one holds a complete artifact. Loading validates the artifact instead of guessing.
"""

import dataclasses
import importlib.metadata
import json
import os
import shutil
import sys
import types
from importlib.metadata import version

import pytest

import smallaxe
from smallaxe import _artifact
from smallaxe.exceptions import ValidationError
from smallaxe.preprocessing import Imputer
from smallaxe.training import algorithm, classifiers, regressors
from smallaxe.training import catboost as catboost_module
from smallaxe.training.base import Classifier, Model, Regressor
from tests.fakes import install_fake_modules

PAYLOAD_KEYS = {
    "algorithm",
    "task",
    "params",
    "feature_cols",
    "features_col",
    "label_col",
    "exclude_cols",
    "is_fitted",
    "metadata",
    "validation_scores",
    "__class__",
    "__module__",
}

LOADERS = {
    "load_model": smallaxe.load_model,
    "Model.load": Model.load,
    "regressors.load": regressors.load,
    "classifiers.load": classifiers.load,
}


@pytest.fixture
def df(spark_session):
    rows = [(float(i), float(i % 5), float(i * 2 + i % 3), i % 2) for i in range(1, 41)]
    return spark_session.createDataFrame(rows, ["a", "b", "y", "label"])


@pytest.fixture
def saved_regressor(df, tmp_path):
    model = regressors.random_forest(n_estimators=3, max_depth=2, seed=1)
    model.fit(df, label_col="y", feature_cols=["a", "b"])
    path = tmp_path / "model"
    model.save(str(path))
    return path


def _metadata(path):
    return json.loads((path / "metadata.json").read_text())


def _rewrite(path, **changes):
    """Edit metadata.json by hand (and drop its Hadoop checksum, which would no longer match)."""
    metadata = _metadata(path)
    for key, value in changes.items():
        if value is _DROP:
            metadata.pop(key, None)
        else:
            metadata[key] = value
    (path / "metadata.json").write_text(json.dumps(metadata))
    crc = path / ".metadata.json.crc"
    if crc.exists():
        crc.unlink()
    return str(path)


_DROP = object()


def _no_distribution(name):
    raise importlib.metadata.PackageNotFoundError(name)


def _predictions(model, df):
    rows = model.predict(df).select("a", "predict_label").orderBy("a").collect()
    return [(row["a"], row["predict_label"]) for row in rows]


class TestHeader:
    def test_fresh_artifact_has_the_header_and_the_payload(self, saved_regressor):
        metadata = _metadata(saved_regressor)

        assert type(metadata["format_version"]) is int
        assert metadata["format_version"] == _artifact.FORMAT_VERSION == 1
        assert metadata["kind"] == "model"
        assert metadata["smallaxe_version"] == smallaxe.__version__
        assert set(metadata) == PAYLOAD_KEYS | set(_artifact.RESERVED_KEYS)
        assert metadata["algorithm"] == "random_forest"
        assert metadata["task"] == "simple_regression"
        assert metadata["__class__"] == "Regressor"
        assert metadata["params"] == {"n_estimators": 3, "max_depth": 2, "seed": 1}

    def test_xgboost_records_its_package_version(self, df, tmp_path):
        if not algorithm.get("xgboost").available():
            pytest.skip("xgboost is not installed")
        model = classifiers.xgboost(n_estimators=2, max_depth=2)
        model.fit(df, label_col="label", feature_cols=["a", "b"])
        model.save(str(tmp_path / "m"))

        nested = _metadata(tmp_path / "m")["metadata"]

        assert nested["algorithm_package"] == "xgboost"
        assert nested["algorithm_package_version"] == version("xgboost")
        assert "smallaxe_version" in nested and "spark_version" in nested
        assert smallaxe.load_model(str(tmp_path / "m")).metadata["algorithm_package"] == "xgboost"

    def test_random_forest_has_no_package_version(self, saved_regressor):
        nested = _metadata(saved_regressor)["metadata"]

        assert nested["algorithm_package"] is None
        assert nested["algorithm_package_version"] is None

    def test_package_version_of_a_package_without_distribution_metadata(self, monkeypatch):
        """catboost_spark ships in the Spark package jar, so pip may not know its version."""
        module = types.ModuleType("catboost_spark")
        module.__version__ = "1.2.3"
        monkeypatch.setitem(sys.modules, "catboost_spark", module)
        monkeypatch.setattr(importlib.metadata, "version", _no_distribution)

        assert algorithm._package_version(algorithm.get("catboost")) == "1.2.3"

    def test_package_version_imports_nothing(self, monkeypatch):
        monkeypatch.delitem(sys.modules, "catboost_spark", raising=False)
        monkeypatch.setattr(importlib.metadata, "version", _no_distribution)

        assert algorithm._package_version(algorithm.get("catboost")) is None
        assert "catboost_spark" not in sys.modules


class TestSaveOrder:
    def test_metadata_is_written_after_the_spark_model(self, monkeypatch, df, tmp_path):
        model = regressors.random_forest(n_estimators=2, max_depth=2)
        model.fit(df, label_col="y", feature_cols=["a", "b"])
        path = tmp_path / "model"
        seen = []
        real_write = _artifact.write_metadata

        def recording_write(target, kind, payload):
            seen.append((path / "spark_model" / "metadata").is_dir())
            real_write(target, kind, payload)

        monkeypatch.setattr(_artifact, "write_metadata", recording_write)

        model.save(str(path))

        assert seen == [True]

    def test_overwriting_removes_the_old_metadata_before_writing_the_model(
        self, monkeypatch, df, saved_regressor
    ):
        model = regressors.random_forest(n_estimators=2, max_depth=2)
        model.fit(df, label_col="y", feature_cols=["a", "b"])

        def failing_save(path):
            assert not os.path.exists(os.path.join(path, "metadata.json"))
            raise RuntimeError("disk full")

        monkeypatch.setattr(model, "_save_spark_model", failing_save)

        with pytest.raises(RuntimeError, match="disk full"):
            model.save(str(saved_regressor))

        # The half-written directory is not mistaken for the old, complete artifact.
        with pytest.raises(ValidationError, match="metadata.json"):
            smallaxe.load_model(str(saved_regressor))


class TestOverwrite:
    def test_saving_over_an_artifact_leaves_nothing_of_the_old_one(
        self, df, saved_regressor, tmp_path
    ):
        if not algorithm.get("xgboost").available():
            pytest.skip("xgboost is not installed")
        before = _metadata(saved_regressor)
        model = classifiers.xgboost(n_estimators=2, max_depth=2)
        model.fit(df, label_col="label", feature_cols=["a", "b"])

        model.save(str(saved_regressor))

        after = _metadata(saved_regressor)
        assert after["algorithm"] == "xgboost" and after["task"] == "binary"
        assert after["metadata"]["training_timestamp"] != before["metadata"]["training_timestamp"]
        names = sorted(p for p in os.listdir(saved_regressor) if not p.endswith(".crc"))
        assert names == ["metadata.json", "spark_model"]
        # Random Forest wrote data/ and treesMetadata/; none of it may survive.
        spark_files = sorted(p for p in os.listdir(saved_regressor / "spark_model") if p[0] != ".")
        assert spark_files == ["metadata", "model"]
        loaded = smallaxe.load_model(str(saved_regressor))
        assert isinstance(loaded, Classifier) and loaded.algorithm.name == "xgboost"
        assert _predictions(loaded, df) == _predictions(model, df)


class TestLoadRejects:
    @pytest.mark.parametrize("loader", sorted(LOADERS))
    def test_missing_path(self, tmp_path, loader):
        missing = str(tmp_path / "nope")
        with pytest.raises(ValidationError) as info:
            LOADERS[loader](missing)
        assert str(info.value).startswith(f"Cannot load model: path {missing!r} does not exist.")
        assert "file:/" in str(info.value)

    @pytest.mark.parametrize("loader", sorted(LOADERS))
    def test_directory_without_metadata(self, tmp_path, loader):
        (tmp_path / "empty").mkdir()
        path = str(tmp_path / "empty")
        with pytest.raises(ValidationError) as info:
            LOADERS[loader](path)
        assert str(info.value) == (
            f"Cannot load model: {path!r} is not a smallaxe model artifact "
            "(expected metadata.json in it)."
        )

    def test_newer_format_version(self, saved_regressor):
        path = _rewrite(saved_regressor, format_version=999, smallaxe_version="9.0.0")
        with pytest.raises(ValidationError, match="newer smallaxe") as info:
            smallaxe.load_model(path)
        assert "999" in str(info.value) and "9.0.0" in str(info.value)

    def test_imputer_artifact(self, spark_session, tmp_path):
        fit = spark_session.createDataFrame([(1.0,), (None,), (3.0,)], "x double")
        imputer = Imputer(numerical_strategy="mean")
        imputer.fit(fit, numerical_cols=["x"])
        imputer.save(str(tmp_path / "imputer"))

        with pytest.raises(ValidationError, match="'imputer' artifact, not a 'model'"):
            smallaxe.load_model(str(tmp_path / "imputer"))

    def test_unknown_parameter(self, saved_regressor):
        params = dict(_metadata(saved_regressor)["params"], num_leaves_from_the_future=7)
        path = _rewrite(saved_regressor, params=params)

        with pytest.raises(ValidationError) as info:
            smallaxe.load_model(path)
        message = str(info.value)
        assert "num_leaves_from_the_future" in message
        assert "random_forest" in message
        assert "newer smallaxe" in message

    def test_parameter_of_another_task(self, monkeypatch, df, tmp_path):
        """CatBoost's ``scale_pos_weight`` is a classification row: a regressor rejects it."""
        record = dataclasses.replace(algorithm.get("catboost"), prepare=None)
        install_fake_modules(monkeypatch, record)
        monkeypatch.setattr(catboost_module, "ALGORITHM", record)
        model = regressors.catboost(n_estimators=2)
        model.fit(df, label_col="y", feature_cols=["a", "b"])
        model.save(str(tmp_path / "m"))
        assert "scale_pos_weight" in {p.name for p in record.visible_params("binary")}
        path = _rewrite(tmp_path / "m", params={"scale_pos_weight": 2.0})

        with pytest.raises(ValidationError, match="scale_pos_weight") as info:
            smallaxe.load_model(path)
        assert "catboost" in str(info.value) and "simple_regression" in str(info.value)

    def test_params_that_are_not_a_mapping(self, saved_regressor):
        path = _rewrite(saved_regressor, params=["n_estimators"])
        with pytest.raises(ValidationError, match="params"):
            smallaxe.load_model(path)

    def test_missing_params_take_their_defaults(self, df, saved_regressor):
        path = _rewrite(saved_regressor, params={"n_estimators": 3})

        loaded = smallaxe.load_model(path)

        assert loaded.get_param("n_estimators") == 3
        assert loaded.get_param("max_depth") == loaded.default_params["max_depth"]
        assert loaded.predict(df).count() == df.count()

    @pytest.mark.parametrize("versioned", [True, False], ids=["versioned", "unversioned"])
    def test_missing_task(self, saved_regressor, versioned):
        changes = {"task": _DROP}
        if not versioned:
            changes.update(format_version=_DROP, kind=_DROP, smallaxe_version=_DROP)
        path = _rewrite(saved_regressor, **changes)

        with pytest.raises(ValidationError, match="'task'"):
            smallaxe.load_model(path)

    def test_task_the_algorithm_does_not_have(self, saved_regressor):
        path = _rewrite(saved_regressor, task="ranking")
        with pytest.raises(ValidationError, match="ranking"):
            smallaxe.load_model(path)

    def test_classifier_through_regressors_load(self, df, tmp_path):
        model = classifiers.random_forest(n_estimators=2, max_depth=2)
        model.fit(df, label_col="label", feature_cols=["a", "b"])
        model.save(str(tmp_path / "clf"))

        with pytest.raises(ValidationError, match="Cannot load a regressor"):
            regressors.load(str(tmp_path / "clf"))
        with pytest.raises(ValidationError, match="Regressor.load expects a regressor"):
            Regressor.load(str(tmp_path / "clf"))


class TestLoadModel:
    def test_returns_the_saved_kind(self, df, saved_regressor, tmp_path):
        clf = classifiers.random_forest(task="binary", n_estimators=2, max_depth=2)
        clf.fit(df, label_col="label", feature_cols=["a", "b"])
        clf.save(str(tmp_path / "clf"))

        regressor = smallaxe.load_model(str(saved_regressor))
        classifier = smallaxe.load_model(str(tmp_path / "clf"))

        assert type(regressor) is Regressor and regressor.task == "simple_regression"
        assert type(classifier) is Classifier and classifier.task == "binary"
        assert _predictions(classifier, df) == _predictions(clf, df)

    def test_loads_an_unversioned_artifact(self, df, saved_regressor):
        expected = _predictions(smallaxe.load_model(str(saved_regressor)), df)
        path = _rewrite(saved_regressor, format_version=_DROP, kind=_DROP, smallaxe_version=_DROP)

        assert _predictions(smallaxe.load_model(path), df) == expected

    def test_fake_algorithm_round_trip(self, monkeypatch, df, tmp_path):
        record = algorithm.get("xgboost")
        install_fake_modules(monkeypatch, record)
        model = classifiers.xgboost(task="multiclass", n_estimators=4)
        model.fit(df, label_col="label", feature_cols=["a", "b"])
        model.save(str(tmp_path / "m"))

        loaded = smallaxe.load_model(str(tmp_path / "m"))

        assert type(loaded) is Classifier and loaded.task == "multiclass"
        assert loaded.get_param("n_estimators") == 4

    def test_copied_artifact_loads(self, df, saved_regressor, tmp_path):
        copy = tmp_path / "copy"
        shutil.copytree(saved_regressor, copy)

        assert smallaxe.load_model(str(copy)).predict(df).count() == df.count()
