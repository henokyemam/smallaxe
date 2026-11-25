"""Pipeline.save / Pipeline.load, smallaxe.load_pipeline and smallaxe.load.

A pipeline artifact is a directory with ``metadata.json`` (kind ``"pipeline"``) and one
directory per saved step under ``steps/``, each that step's own artifact. Nothing is
pickled. Pipelines saved by smallaxe 0.10 (``tests/fixtures/pipeline_0_10``, written by
the released wheel) still load; their pickled Encoder is the one pickle read.
"""

import json
import math
import os
import pickle
import shutil
from pathlib import Path

import pytest
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F

import smallaxe
from smallaxe.exceptions import ValidationError
from smallaxe.pipeline import Pipeline
from smallaxe.preprocessing import Encoder, Imputer, Scaler
from smallaxe.training import Classifier, Classifiers, Regressor, Regressors, algorithm

FIXTURES = Path(__file__).parent / "fixtures"
LEGACY = FIXTURES / "pipeline_0_10"
FORMAT_1 = FIXTURES / "pipeline_format_1"

TASKS = ("simple_regression", "binary", "multiclass")
CASES = [(name, task) for name in ("random_forest", "xgboost") for task in TASKS]
NUMERICAL = ["age", "income"]
CATEGORICAL = ["city"]


@pytest.fixture(scope="module")
def frame(spark_session):
    """Nulls in every listed column, an unlisted string ID and an unlisted number."""
    cities = ["paris", "rome", "oslo"]
    rows = [
        (
            f"id-{i:03d}",
            float(20 + i % 30) if i % 7 else None,
            float(40000 + i * 850) if i % 9 else None,
            cities[i % 3] if i % 8 else None,
            float(i % 5),
            50.0 + i * 3 + (i % 4) * 10,
            i % 2,
            i % 3,
        )
        for i in range(1, 91)
    ]
    return spark_session.createDataFrame(
        rows,
        "customer_id string, age double, income double, city string, unlisted double, "
        "target double, binary int, multiclass int",
    )


@pytest.fixture(scope="module")
def shifted(frame):
    """The frame with an unseen city and a column the pipeline never saw."""
    return frame.withColumn(
        "city", F.when(F.col("unlisted") == 4.0, F.lit("lima")).otherwise(F.col("city"))
    ).withColumn("brand_new", F.lit("x"))


def _label(task):
    return "target" if task == "simple_regression" else task


def _model(name, task):
    if not algorithm.get(name).available():
        pytest.skip(f"{name} is not installed")
    params = {"n_estimators": 5, "seed": 7}
    if task == "simple_regression":
        return getattr(Regressors, name)(**params)
    return getattr(Classifiers, name)(task=task, **params)


def _pipeline(model, encoder_method="onehot"):
    return Pipeline(
        [
            ("imputer", Imputer(numerical_strategy="median", categorical_strategy="most_frequent")),
            ("scaler", Scaler()),
            ("encoder", Encoder(method=encoder_method)),
            ("model", model),
        ]
    )


def _fit_full(name, task, frame):
    pipeline = _pipeline(_model(name, task))
    return pipeline.fit(
        frame, label_col=_label(task), numerical_cols=NUMERICAL, categorical_cols=CATEGORICAL
    )


def _predictions(pipeline, df):
    rows = pipeline.predict(df).select("customer_id", "predict_label").orderBy("customer_id")
    return [tuple(r) for r in rows.collect()]


def _probabilities(pipeline, df):
    proba = pipeline.predict_proba(df).select(
        "customer_id", vector_to_array("probability").alias("p")
    )
    return [(r["customer_id"], list(r["p"])) for r in proba.orderBy("customer_id").collect()]


def _files(path):
    return sorted(str(p.relative_to(path)) for p in Path(path).rglob("*") if p.is_file())


def _assert_same_scores(got, want):
    """Equal metrics. Spark sums in partition order, so the last bits of a mean may vary."""
    assert set(got) == set(want)
    for key, value in want.items():
        if value is None:
            assert got[key] is None, key
        else:
            assert math.isclose(got[key], value, rel_tol=1e-12, abs_tol=1e-12), key


def _write_json(path, document):
    Path(path).write_text(json.dumps(document, indent=2))
    crc = Path(path).parent / f".{Path(path).name}.crc"
    if crc.exists():
        crc.unlink()


def _read_json(path):
    return json.loads(Path(path).read_text())


@pytest.mark.parametrize("name, task", CASES, ids=[f"{n}-{t}" for n, t in CASES])
@pytest.mark.parametrize("uri", [False, True], ids=["path", "file-uri"])
class TestFullPipelineContract:
    def test_round_trip_is_identical_through_every_loader(
        self, frame, shifted, tmp_path, name, task, uri
    ):
        pipeline = _fit_full(name, task, frame)
        target = tmp_path / "pipe"
        path = f"file://{target}" if uri else str(target)

        assert "customer_id" not in pipeline.feature_cols
        assert "unlisted" not in pipeline.feature_cols
        assert pipeline.feature_cols == NUMERICAL + pipeline["encoder"].output_cols

        before = _predictions(pipeline, frame)
        scores = pipeline.evaluate(frame)
        proba = _probabilities(pipeline, frame) if task != "simple_regression" else None

        pipeline.save(path)

        assert not [f for f in _files(target) if f.endswith(".pkl")]
        loaders = [Pipeline.load, smallaxe.load_pipeline, smallaxe.load]
        for loader in loaders:
            loaded = loader(path)
            assert isinstance(loaded, Pipeline)
            assert loaded.feature_cols == pipeline.feature_cols
            assert loaded["model"]._feature_cols == pipeline.feature_cols
            assert isinstance(loaded["model"], Classifier if proba else Regressor)
            assert _predictions(loaded, frame) == before
            _assert_same_scores(loaded.evaluate(frame), scores)
            if proba is not None:
                assert _probabilities(loaded, frame) == proba
            else:
                with pytest.raises(ValidationError, match="predict_proba"):
                    loaded.predict_proba(frame)

            result = loaded.predict(shifted)
            assert result.count() == shifted.count()
            assert result.filter(F.col("predict_label").isNull()).count() == 0
            assert "brand_new" in result.columns


class TestArtifactLayout:
    def test_layout_metadata_and_no_pickle(self, frame, tmp_path):
        pipeline = _fit_full("random_forest", "binary", frame)
        path = tmp_path / "pipe"

        pipeline.save(str(path))

        files = _files(path)
        assert not [f for f in files if f.endswith(".pkl")]
        assert "metadata.json" in files
        assert "steps/0_imputer/metadata.json" in files
        assert "steps/1_scaler/metadata.json" in files
        assert "steps/1_scaler/spark_model/metadata/part-00000" in files
        assert "steps/2_encoder/metadata.json" in files
        assert "steps/3_model/metadata.json" in files
        metadata = _read_json(path / "metadata.json")
        assert metadata["format_version"] == 1
        assert metadata["kind"] == "pipeline"
        assert metadata["smallaxe_version"] == smallaxe.__version__
        assert metadata["steps"] == [
            {"name": "imputer", "kind": "imputer", "path": "steps/0_imputer", "skipped": False},
            {"name": "scaler", "kind": "scaler", "path": "steps/1_scaler", "skipped": False},
            {"name": "encoder", "kind": "encoder", "path": "steps/2_encoder", "skipped": False},
            {"name": "model", "kind": "model", "path": "steps/3_model", "skipped": False},
        ]
        assert metadata["numerical_cols"] == NUMERICAL
        assert metadata["categorical_cols"] == CATEGORICAL
        assert metadata["label_col"] == "binary"
        assert metadata["feature_cols"] == pipeline.feature_cols
        assert metadata["has_model"] is True
        for step in ("0_imputer", "1_scaler", "2_encoder", "3_model"):
            kind = step.split("_", 1)[1]
            assert _read_json(path / "steps" / step / "metadata.json")["kind"] == kind

    def test_step_names_are_made_safe_for_directories(self, frame, tmp_path):
        pipeline = Pipeline([("my scaler/../x", Scaler())])
        pipeline.fit(frame, numerical_cols=["unlisted"])
        path = tmp_path / "pipe"

        pipeline.save(str(path))

        assert _read_json(path / "metadata.json")["steps"][0]["path"] == "steps/0_my_scaler_.._x"
        loaded = Pipeline.load(str(path))
        assert loaded.steps[0][0] == "my scaler/../x"

    def test_saving_over_a_pipeline_leaves_nothing_stale(self, frame, tmp_path):
        path = tmp_path / "pipe"
        _fit_full("random_forest", "binary", frame).save(str(path))
        small = Pipeline([("scale", Scaler())]).fit(frame, numerical_cols=["unlisted"])

        small.save(str(path))

        assert sorted(os.listdir(path / "steps")) == ["0_scale"]
        assert Pipeline.load(str(path)).steps[0][0] == "scale"

    def test_saving_over_a_0_10_pipeline_removes_its_files(self, frame, tmp_path):
        path = tmp_path / "pipe"
        shutil.copytree(LEGACY / "pipeline", path)
        small = Pipeline([("scale", Scaler())]).fit(frame, numerical_cols=["unlisted"])

        small.save(str(path))

        assert not [f for f in _files(path) if f.startswith("step_")]
        assert Pipeline.load(str(path)).steps[0][0] == "scale"

    def test_unknown_step_kind_cannot_be_saved_and_nothing_is_written(self, frame, tmp_path):
        class Doubler:
            def fit(self, df, numerical_cols=None, categorical_cols=None):
                return self

            def transform(self, df):
                return df.withColumn("unlisted", F.col("unlisted") * 2)

        pipeline = Pipeline([("scale", Scaler()), ("double", Doubler())])
        pipeline.fit(frame, numerical_cols=["unlisted"])
        path = tmp_path / "pipe"

        with pytest.raises(ValidationError, match="'double'"):
            pipeline.save(str(path))
        assert not path.exists()

    def test_save_before_fit_and_empty_path(self, frame):
        with pytest.raises(smallaxe.exceptions.ModelNotFittedError):
            Pipeline([("scaler", Scaler())]).save("/tmp/never")
        fitted = Pipeline([("scaler", Scaler())]).fit(frame, numerical_cols=["unlisted"])
        with pytest.raises(ValidationError, match="empty"):
            fitted.save("")


class TestSkippedStepsPersist:
    def test_skipped_steps_round_trip(self, frame, tmp_path):
        pipeline = Pipeline(
            [
                ("imputer", Imputer(numerical_strategy=0.0)),
                ("scaler", Scaler(method="minmax")),
                ("encoder", Encoder(method="label", max_categories=7, handle_rare="keep")),
                ("model", Regressors.random_forest(n_estimators=5, seed=3)),
            ]
        )
        clean = frame.fillna({"age": 1.0, "income": 1.0})
        with pytest.warns(UserWarning, match="Feature inference"):
            pipeline.fit(clean, label_col="target")
        assert pipeline.skipped_steps == ["imputer", "scaler", "encoder"]
        path = tmp_path / "pipe"

        pipeline.save(str(path))

        assert sorted(os.listdir(path / "steps")) == ["3_model"]
        entries = _read_json(path / "metadata.json")["steps"]
        assert [e["skipped"] for e in entries] == [True, True, True, False]
        loaded = Pipeline.load(str(path))
        assert loaded.skipped_steps == ["imputer", "scaler", "encoder"]
        assert isinstance(loaded["imputer"], Imputer) and not loaded["imputer"]._is_fitted
        assert loaded["imputer"]._numerical_strategy == 0.0
        assert loaded["scaler"].method == "minmax" and not loaded["scaler"]._is_fitted
        encoder = loaded["encoder"]
        assert (encoder.method, encoder.max_categories, encoder.handle_rare) == ("label", 7, "keep")
        assert loaded.feature_cols == pipeline.feature_cols
        assert _predictions(loaded, clean) == _predictions(pipeline, clean)
        assert loaded.transform(clean).collect() == clean.collect()

    def test_a_skipped_step_between_fitted_ones_round_trips(self, frame, tmp_path):
        pipeline = Pipeline(
            [
                ("imputer", Imputer(numerical_strategy="mean")),
                ("encoder", Encoder()),
                ("scaler", Scaler()),
                ("model", Regressors.random_forest(n_estimators=5, seed=3)),
            ]
        )
        pipeline.fit(frame, label_col="target", numerical_cols=NUMERICAL)
        assert pipeline.skipped_steps == ["encoder"]
        path = tmp_path / "pipe"

        pipeline.save(str(path))

        assert sorted(os.listdir(path / "steps")) == ["0_imputer", "2_scaler", "3_model"]
        loaded = Pipeline.load(str(path))
        assert loaded.skipped_steps == ["encoder"]
        assert _predictions(loaded, frame) == _predictions(pipeline, frame)


class TestLoadRejections:
    @pytest.fixture
    def saved(self, frame, tmp_path):
        path = tmp_path / "pipe"
        Pipeline(
            [("scaler", Scaler()), ("model", Regressors.random_forest(n_estimators=5, seed=1))]
        ).fit(frame.fillna(0.0), label_col="target", numerical_cols=NUMERICAL).save(str(path))
        return path

    def test_missing_path(self, tmp_path):
        with pytest.raises(ValidationError, match="does not exist"):
            Pipeline.load(str(tmp_path / "nope"))
        with pytest.raises(ValidationError, match="does not exist"):
            smallaxe.load(str(tmp_path / "nope"))

    def test_directory_without_metadata(self, tmp_path):
        with pytest.raises(ValidationError, match="not a smallaxe pipeline artifact"):
            Pipeline.load(str(tmp_path))

    def test_wrong_kind(self, frame, saved):
        with pytest.raises(ValidationError, match="'model' artifact, not a 'pipeline'"):
            Pipeline.load(str(saved / "steps" / "1_model"))
        with pytest.raises(ValidationError, match="'pipeline' artifact, not a 'model'"):
            smallaxe.load_model(str(saved))
        with pytest.raises(ValidationError, match="'scaler' artifact, not a 'pipeline'"):
            smallaxe.load_pipeline(str(saved / "steps" / "0_scaler"))

    def test_legacy_model_is_not_a_pipeline(self):
        with pytest.raises(ValidationError, match="load_model"):
            Pipeline.load(str(FIXTURES / "artifacts_0_10_0" / "random_forest_regressor"))

    def test_newer_format_version(self, saved):
        metadata = _read_json(saved / "metadata.json")
        metadata["format_version"] = 999
        _write_json(saved / "metadata.json", metadata)

        with pytest.raises(ValidationError, match="newer smallaxe"):
            Pipeline.load(str(saved))
        with pytest.raises(ValidationError, match="newer smallaxe"):
            smallaxe.load(str(saved))

    def test_missing_step_directory(self, saved):
        shutil.rmtree(saved / "steps" / "1_model")

        with pytest.raises(ValidationError, match="step 'model'.*missing"):
            Pipeline.load(str(saved))

    @pytest.mark.parametrize(
        "change, match",
        [
            (lambda m: m.pop("steps"), "missing"),
            (lambda m: m.update(steps="x"), "steps"),
            (lambda m: m["steps"][0].update(kind="mystery"), "mystery"),
            (lambda m: m["steps"][0].update(path="../../elsewhere"), "elsewhere"),
            (lambda m: m["steps"][0].update(path="/abs/steps/0_scaler"), "0_scaler"),
            (lambda m: m.update(feature_cols=["nope"]), "feature_cols"),
            (lambda m: m.update(numerical_cols="age"), "numerical_cols"),
        ],
        ids=[
            "no-steps",
            "steps-not-list",
            "unknown-kind",
            "path-escape",
            "abs-path",
            "features",
            "columns",
        ],
    )
    def test_corrupted_metadata(self, saved, change, match):
        metadata = _read_json(saved / "metadata.json")
        change(metadata)
        _write_json(saved / "metadata.json", metadata)

        with pytest.raises(ValidationError, match=match):
            Pipeline.load(str(saved))


def _legacy_copy(tmp_path):
    path = tmp_path / "legacy"
    shutil.copytree(LEGACY / "pipeline", path)
    return path


class TestLegacyPipeline:
    """Pipelines saved by the released smallaxe 0.10.0 (``pipeline_0_10/make_fixture.py``)."""

    def test_fixture_was_written_by_0_10_0_in_its_layout(self):
        expected = _read_json(LEGACY / "expected.json")
        assert expected["smallaxe_version"] == "0.10.0"
        metadata = _read_json(LEGACY / "pipeline" / "metadata.json")
        assert "format_version" not in metadata and "kind" not in metadata
        assert metadata["step_names"] == ["imputer", "scaler", "encoder"]
        assert metadata["step_types"] == ["Imputer", "Scaler", "Encoder"]
        assert (LEGACY / "pipeline" / "step_0_imputer" / "imputer_metadata.json").is_file()
        assert (LEGACY / "pipeline" / "step_1_scaler" / "scaler_metadata.json").is_file()
        assert (LEGACY / "pipeline" / "step_2_encoder.pkl").is_file()

    @pytest.mark.parametrize("loader", [Pipeline.load, smallaxe.load_pipeline, smallaxe.load])
    def test_loads_and_transforms_as_0_10_did(self, spark_session, loader):
        expected = _read_json(LEGACY / "expected.json")
        probe = spark_session.createDataFrame(
            [tuple(r) for r in expected["probe"]], expected["schema"]
        )

        loaded = loader(str(LEGACY / "pipeline"))

        assert isinstance(loaded, Pipeline)
        assert [type(s) for _, s in loaded.steps] == [Imputer, Scaler, Encoder]
        assert loaded.skipped_steps == []
        assert loaded.feature_cols is None
        result = loaded.transform(probe)
        assert result.columns == expected["columns"]
        got = [list(r) for r in result.orderBy("id").collect()]
        want = [list(r) for r in expected["output"]]
        # 0.10 label-encoded the unseen city 'lima' as null; it now gets __UNKNOWN__.
        assert want[7][3] is None
        want[7][3] = loaded["encoder"].category_mappings["city"]["__UNKNOWN__"]
        assert len(got) == len(want)
        for got_row, want_row in zip(got, want):
            for value, wanted in zip(got_row, want_row):
                if isinstance(wanted, float):
                    assert math.isclose(value, wanted, rel_tol=1e-12, abs_tol=1e-12)
                else:
                    assert value == wanted

    def test_resaves_in_the_current_format(self, spark_session, tmp_path):
        expected = _read_json(LEGACY / "expected.json")
        probe = spark_session.createDataFrame(
            [tuple(r) for r in expected["probe"]], expected["schema"]
        )
        legacy = Pipeline.load(str(LEGACY / "pipeline"))

        legacy.save(str(tmp_path / "new"))
        resaved = Pipeline.load(str(tmp_path / "new"))

        assert _read_json(tmp_path / "new" / "metadata.json")["kind"] == "pipeline"
        assert resaved.transform(probe).collect() == legacy.transform(probe).collect()

    def test_any_other_pickle_is_rejected(self, tmp_path):
        path = _legacy_copy(tmp_path)
        (path / "step_9_extra.pkl").write_bytes(pickle.dumps({"a": 1}))

        with pytest.raises(ValidationError, match="step_9_extra.pkl.*no longer loaded"):
            Pipeline.load(str(path))

    def test_pickle_inside_a_step_directory_is_rejected(self, tmp_path):
        path = _legacy_copy(tmp_path)
        (path / "step_0_imputer" / "imputer.pkl").write_bytes(pickle.dumps({"a": 1}))

        with pytest.raises(ValidationError, match="imputer.pkl.*no longer loaded"):
            Pipeline.load(str(path))

    def test_pickled_step_of_another_type_is_rejected(self, tmp_path):
        path = _legacy_copy(tmp_path)
        metadata = _read_json(path / "metadata.json")
        metadata["step_types"][2] = "MyTransformer"
        _write_json(path / "metadata.json", metadata)

        with pytest.raises(ValidationError, match="MyTransformer.*no longer loaded"):
            Pipeline.load(str(path))

    def test_pickled_encoder_that_is_not_an_encoder_is_rejected(self, tmp_path):
        path = _legacy_copy(tmp_path)
        (path / "step_2_encoder.pkl").write_bytes(pickle.dumps({"not": "an encoder"}))

        with pytest.raises(ValidationError, match="not an Encoder"):
            Pipeline.load(str(path))

    def test_pickled_encoder_referencing_other_code_is_not_run(self, tmp_path):
        path = _legacy_copy(tmp_path)
        marker = tmp_path / "ran"

        class Payload:
            def __reduce__(self):
                return (os.mkdir, (str(marker),))

        (path / "step_2_encoder.pkl").write_bytes(pickle.dumps(Payload()))

        with pytest.raises(ValidationError, match="posix.mkdir|os.mkdir|not allowed"):
            Pipeline.load(str(path))
        assert not marker.exists()

    def test_legacy_model_step_is_rejected_with_a_reason(self, tmp_path):
        path = _legacy_copy(tmp_path)
        metadata = _read_json(path / "metadata.json")
        metadata["step_names"].append("model")
        metadata["step_types"].append("RandomForestRegressor")
        _write_json(path / "metadata.json", metadata)

        with pytest.raises(ValidationError, match="RandomForestRegressor"):
            Pipeline.load(str(path))

    def test_missing_legacy_step(self, tmp_path):
        path = _legacy_copy(tmp_path)
        shutil.rmtree(path / "step_1_scaler")

        with pytest.raises(ValidationError, match="scaler"):
            Pipeline.load(str(path))


class TestLoadAnyArtifact:
    def test_load_returns_each_kind(self, frame, tmp_path):
        clean = frame.fillna({"age": 1.0, "income": 1.0, "city": "paris"})
        imputer = Imputer(numerical_strategy="mean").fit(frame, numerical_cols=NUMERICAL)
        scaler = Scaler().fit(clean, numerical_cols=NUMERICAL)
        encoder = Encoder().fit(clean, categorical_cols=CATEGORICAL)
        model = Regressors.random_forest(n_estimators=5).fit(
            clean, label_col="target", feature_cols=NUMERICAL
        )
        pipeline = Pipeline([("scaler", Scaler())]).fit(clean, numerical_cols=NUMERICAL)
        for name, obj, kind in [
            ("i", imputer, Imputer),
            ("s", scaler, Scaler),
            ("e", encoder, Encoder),
            ("m", model, Regressor),
            ("p", pipeline, Pipeline),
        ]:
            obj.save(str(tmp_path / name))
            assert type(smallaxe.load(str(tmp_path / name))) is kind

    def test_load_reads_legacy_layouts(self):
        assert isinstance(
            smallaxe.load(str(FIXTURES / "artifacts_0_10_0" / "random_forest_regressor")),
            Regressor,
        )
        assert isinstance(
            smallaxe.load(str(FIXTURES / "artifacts_0_8_1" / "random_forest_classifier")),
            Classifier,
        )
        assert isinstance(smallaxe.load(str(FIXTURES / "preprocessing_0_10" / "imputer")), Imputer)
        assert isinstance(smallaxe.load(str(FIXTURES / "preprocessing_0_10" / "scaler")), Scaler)
        assert isinstance(smallaxe.load(str(LEGACY / "pipeline")), Pipeline)

    def test_load_rejects_what_is_not_an_artifact(self, tmp_path):
        with pytest.raises(ValidationError, match="not a smallaxe artifact"):
            smallaxe.load(str(tmp_path))

    def test_load_rejects_an_unknown_kind(self, tmp_path):
        _write_json(tmp_path / "metadata.json", {"format_version": 1, "kind": "mystery"})

        with pytest.raises(ValidationError, match="mystery"):
            smallaxe.load(str(tmp_path))


class TestFrozenFormat1:
    """``pipeline_format_1`` was written by ``pipeline_format_1/make_fixture.py``."""

    @pytest.mark.parametrize("name", ["pipeline", "pipeline_skipped"])
    def test_loads_and_predicts_as_when_saved(self, spark_session, name):
        expected = _read_json(FORMAT_1 / "expected.json")
        spec = expected["pipelines"][name]
        probe = spark_session.createDataFrame(
            [tuple(r) for r in expected["probe"]], expected["schema"]
        ).drop("target")

        loaded = smallaxe.load_pipeline(str(FORMAT_1 / name))

        assert _read_json(FORMAT_1 / name / "metadata.json")["format_version"] == 1
        assert loaded.feature_cols == spec["feature_cols"]
        assert loaded.skipped_steps == spec["skipped_steps"]
        assert loaded._numerical_cols == spec["numerical_cols"]
        assert loaded._categorical_cols == spec["categorical_cols"]
        got = loaded.predict(probe).select("id", "predict_label").orderBy("id").collect()
        assert [r["id"] for r in got] == [r[0] for r in spec["predictions"]]
        for row, (_, want) in zip(got, spec["predictions"]):
            assert math.isclose(row["predict_label"], want, rel_tol=1e-9, abs_tol=1e-12)
