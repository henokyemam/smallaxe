"""Save / load for Imputer, Scaler and Encoder.

Every step saves a directory holding ``metadata.json`` (written through ``smallaxe._fs``,
the Hadoop FileSystem layer) and, for the Scaler, ``spark_model/``. Round trips must
reproduce the transform exactly, on data with nulls and unseen categories. Imputer and
Scaler artifacts written by smallaxe 0.10 (``imputer_metadata.json`` /
``scaler_metadata.json``) must still load.
"""

import json
import math
import os
import pickle
from pathlib import Path

import pytest

import smallaxe.preprocessing
from smallaxe import _artifact
from smallaxe.exceptions import ModelNotFittedError, ValidationError
from smallaxe.preprocessing import Encoder, Imputer, Scaler

FIXTURES = Path(__file__).parent / "fixtures" / "preprocessing_0_10"
EXPECTED = json.loads((FIXTURES / "expected.json").read_text())

KINDS = {Imputer: "imputer", Scaler: "scaler", Encoder: "encoder"}


def _rows(df):
    """Rows ordered by id, with NaN made comparable."""
    return [
        tuple("NaN" if isinstance(value, float) and math.isnan(value) else value for value in row)
        for row in df.orderBy("id").collect()
    ]


def _assert_same_output(actual, expected):
    assert actual.columns == expected.columns
    assert actual.dtypes == expected.dtypes
    assert _rows(actual) == _rows(expected)


def _write_raw_metadata(directory, document, name="metadata.json"):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(json.dumps(document))


@pytest.fixture
def mixed_fit(spark_session):
    rows = [
        (1, 25.0, 50000.0, "Paris", True, 3),
        (2, None, 60000.0, "Rome", False, 3),
        (3, 35.0, None, "Paris", True, 5),
        (4, 45.0, 80000.0, None, None, None),
        (5, 30.0, 70000.0, "Oslo", True, 5),
        (6, 52.0, 65000.0, "Paris", False, 3),
        (7, 41.0, 72000.0, "Rome", True, 8),
    ]
    return spark_session.createDataFrame(
        rows, "id int, age double, income double, city string, flag boolean, code int"
    )


@pytest.fixture
def mixed_probe(spark_session):
    """Nulls in every column, and categories never seen at fit."""
    rows = [
        (1, 25.0, 50000.0, "Paris", True, 3),
        (2, None, None, None, None, None),
        (3, 99.0, 1.0, "Lima", False, 42),
        (4, 30.0, None, "Rome", None, 5),
        (5, None, 10.0, "Oslo", True, None),
    ]
    return spark_session.createDataFrame(
        rows, "id int, age double, income double, city string, flag boolean, code int"
    )


class TestImputerRoundTrip:
    @pytest.mark.parametrize("numerical_strategy", ["mean", "median", "mode", -1.5])
    @pytest.mark.parametrize("categorical_strategy", ["most_frequent", "MISSING"])
    def test_round_trip(
        self, mixed_fit, mixed_probe, tmp_path, numerical_strategy, categorical_strategy
    ):
        imputer = Imputer(
            numerical_strategy=numerical_strategy, categorical_strategy=categorical_strategy
        )
        imputer.fit(mixed_fit, numerical_cols=["age", "income"], categorical_cols=["city"])
        path = str(tmp_path / "imputer")

        imputer.save(path)
        loaded = Imputer.load(path)

        assert loaded._numerical_strategy == numerical_strategy
        assert loaded._categorical_strategy == categorical_strategy
        assert loaded.numerical_fill_values == imputer.numerical_fill_values
        assert loaded.categorical_fill_values == imputer.categorical_fill_values
        _assert_same_output(loaded.transform(mixed_probe), imputer.transform(mixed_probe))

    def test_round_trip_with_non_string_categoricals(self, mixed_fit, mixed_probe, tmp_path):
        imputer = Imputer(categorical_strategy="most_frequent")
        imputer.fit(mixed_fit, categorical_cols=["flag", "code"])
        imputer.save(str(tmp_path / "imputer"))
        loaded = Imputer.load(str(tmp_path / "imputer"))

        result = loaded.transform(mixed_probe)
        _assert_same_output(result, imputer.transform(mixed_probe))
        assert result.filter("flag IS NULL OR code IS NULL").count() == 0

    def test_metadata_layout(self, mixed_fit, tmp_path):
        imputer = Imputer(numerical_strategy="mean", categorical_strategy="most_frequent")
        imputer.fit(mixed_fit, numerical_cols=["age"], categorical_cols=["city"])
        imputer.save(str(tmp_path / "imputer"))

        metadata = json.loads((tmp_path / "imputer" / "metadata.json").read_text())
        assert metadata["kind"] == "imputer"
        assert metadata["format_version"] == _artifact.FORMAT_VERSION
        assert "smallaxe_version" in metadata
        assert metadata["numerical_cols"] == ["age"]
        assert metadata["categorical_fill_values"] == {"city": "Paris"}
        assert not (tmp_path / "imputer" / "imputer_metadata.json").exists()

    def test_pickles(self, mixed_fit, mixed_probe):
        imputer = Imputer(numerical_strategy="median", categorical_strategy="most_frequent")
        imputer.fit(mixed_fit, numerical_cols=["age"], categorical_cols=["city"])
        restored = pickle.loads(pickle.dumps(imputer))
        _assert_same_output(restored.transform(mixed_probe), imputer.transform(mixed_probe))


class TestScalerRoundTrip:
    @pytest.mark.parametrize("method", ["standard", "minmax", "maxabs"])
    def test_round_trip(self, mixed_fit, mixed_probe, tmp_path, method):
        clean = mixed_fit.dropna(subset=["age", "income"])
        scaler = Scaler(method=method).fit(clean, numerical_cols=["age", "income"])
        path = str(tmp_path / "scaler")

        scaler.save(path)
        loaded = Scaler.load(path)

        assert loaded.method == method
        assert loaded.numerical_cols == ["age", "income"]
        assert (tmp_path / "scaler" / "spark_model").is_dir()
        _assert_same_output(loaded.transform(mixed_probe), scaler.transform(mixed_probe))

    def test_metadata_layout(self, mixed_fit, tmp_path):
        Scaler().fit(mixed_fit.dropna(), numerical_cols=["age"]).save(str(tmp_path / "s"))

        metadata = json.loads((tmp_path / "s" / "metadata.json").read_text())
        assert metadata["kind"] == "scaler"
        assert metadata["format_version"] == _artifact.FORMAT_VERSION
        assert metadata["method"] == "standard"
        assert metadata["numerical_cols"] == ["age"]
        assert not (tmp_path / "s" / "scaler_metadata.json").exists()

    def test_save_overwrites(self, mixed_fit, mixed_probe, tmp_path):
        clean = mixed_fit.dropna(subset=["age", "income"])
        path = str(tmp_path / "scaler")
        Scaler(method="standard").fit(clean, numerical_cols=["age"]).save(path)
        second = Scaler(method="minmax").fit(clean, numerical_cols=["income"])
        second.save(path)

        loaded = Scaler.load(path)
        assert loaded.method == "minmax"
        _assert_same_output(loaded.transform(mixed_probe), second.transform(mixed_probe))


ENCODER_SETTINGS = [
    ("other", 100),
    ("other", 1),
    ("keep", 1),
    ("error", 10),
]


class TestEncoderRoundTrip:
    @pytest.mark.parametrize("method", ["onehot", "label"])
    @pytest.mark.parametrize("handle_rare,max_categories", ENCODER_SETTINGS)
    def test_round_trip(
        self, mixed_fit, mixed_probe, tmp_path, method, handle_rare, max_categories
    ):
        encoder = Encoder(method=method, handle_rare=handle_rare, max_categories=max_categories)
        encoder.fit(mixed_fit, categorical_cols=["city", "flag", "code"])
        path = str(tmp_path / "encoder")

        encoder.save(path)
        loaded = Encoder.load(path)

        assert loaded.method == method
        assert loaded.handle_rare == handle_rare
        assert loaded.max_categories == max_categories
        assert loaded.categorical_cols == ["city", "flag", "code"]
        assert loaded.category_mappings == encoder.category_mappings
        result = loaded.transform(mixed_probe)
        _assert_same_output(result, encoder.transform(mixed_probe))
        assert result.count() == mixed_probe.count()

    @pytest.mark.parametrize("method", ["onehot", "label"])
    def test_pickle_and_save_agree(self, mixed_fit, mixed_probe, tmp_path, method):
        encoder = Encoder(method=method).fit(mixed_fit, categorical_cols=["city"])
        encoder.save(str(tmp_path / "e"))
        from_disk = Encoder.load(str(tmp_path / "e"))
        from_pickle = pickle.loads(pickle.dumps(encoder))

        _assert_same_output(from_disk.transform(mixed_probe), from_pickle.transform(mixed_probe))

    def test_metadata_layout(self, mixed_fit, tmp_path):
        encoder = Encoder(method="label").fit(mixed_fit, categorical_cols=["city"])
        encoder.save(str(tmp_path / "e"))

        metadata = json.loads((tmp_path / "e" / "metadata.json").read_text())
        assert metadata["kind"] == "encoder"
        assert metadata["format_version"] == _artifact.FORMAT_VERSION
        assert metadata["method"] == "label"
        assert metadata["max_categories"] == 100
        assert metadata["handle_rare"] == "other"
        assert metadata["categorical_cols"] == ["city"]
        assert metadata["category_mappings"] == {
            "city": {"Paris": 0, "Rome": 1, "Oslo": 2, "__UNKNOWN__": 3}
        }

    def test_hadoop_uri_path(self, mixed_fit, mixed_probe, tmp_path):
        """Paths go through the Hadoop FileSystem layer, so a file: URI works."""
        uri = "file://" + str(tmp_path / "e")
        encoder = Encoder().fit(mixed_fit, categorical_cols=["city"])
        encoder.save(uri)

        assert (tmp_path / "e" / "metadata.json").exists()
        loaded = Encoder.load(uri)
        _assert_same_output(loaded.transform(mixed_probe), encoder.transform(mixed_probe))

    def test_corrupted_mapping_raises(self, tmp_path):
        _write_raw_metadata(
            tmp_path / "e",
            {
                "format_version": 1,
                "kind": "encoder",
                "method": "label",
                "max_categories": 100,
                "handle_rare": "other",
                "categorical_cols": ["c"],
                "category_mappings": {"c": {"a": 0, "b": 5}},
            },
        )
        with pytest.raises(ValidationError, match="corrupted"):
            Encoder.load(str(tmp_path / "e"))


def _fitted(cls, df):
    if cls is Imputer:
        return Imputer(numerical_strategy="mean").fit(df, numerical_cols=["age"])
    if cls is Scaler:
        return Scaler().fit(df.dropna(subset=["age"]), numerical_cols=["age"])
    return Encoder().fit(df, categorical_cols=["city"])


@pytest.mark.parametrize("cls", [Imputer, Scaler, Encoder])
class TestPersistenceErrors:
    def test_save_unfitted_raises(self, cls, tmp_path):
        with pytest.raises(ModelNotFittedError, match="not been fitted"):
            cls().save(str(tmp_path / "x"))
        assert not (tmp_path / "x").exists()

    def test_load_missing_path_raises(self, cls, tmp_path):
        missing = str(tmp_path / "missing")
        with pytest.raises(ValidationError, match="does not exist") as info:
            cls.load(missing)
        assert missing in str(info.value)

    def test_load_directory_without_metadata_raises(self, cls, tmp_path):
        (tmp_path / "empty").mkdir()
        with pytest.raises(ValidationError, match="metadata.json"):
            cls.load(str(tmp_path / "empty"))

    def test_load_wrong_kind_raises(self, cls, mixed_fit, tmp_path):
        other = Encoder if cls is not Encoder else Imputer
        _fitted(other, mixed_fit).save(str(tmp_path / "other"))

        with pytest.raises(ValidationError, match=f"not a '{KINDS[cls]}'") as info:
            cls.load(str(tmp_path / "other"))
        assert f"'{KINDS[other]}' artifact" in str(info.value)

    def test_load_newer_format_version_raises(self, cls, tmp_path):
        _write_raw_metadata(
            tmp_path / "future",
            {"format_version": 999, "kind": KINDS[cls], "smallaxe_version": "99.0.0"},
        )
        with pytest.raises(ValidationError, match="newer smallaxe") as info:
            cls.load(str(tmp_path / "future"))
        assert "999" in str(info.value)

    def test_load_incomplete_metadata_raises(self, cls, tmp_path):
        _write_raw_metadata(tmp_path / "partial", {"format_version": 1, "kind": KINDS[cls]})
        with pytest.raises(ValidationError, match="missing"):
            cls.load(str(tmp_path / "partial"))

    def test_save_empty_path_raises(self, cls, mixed_fit):
        with pytest.raises(ValidationError, match="empty"):
            _fitted(cls, mixed_fit).save("")


def test_scaler_without_spark_model_raises(tmp_path):
    _write_raw_metadata(
        tmp_path / "s",
        {"format_version": 1, "kind": "scaler", "method": "standard", "numerical_cols": ["a"]},
    )
    with pytest.raises(ValidationError, match="spark_model"):
        Scaler.load(str(tmp_path / "s"))


def test_preprocessing_does_no_local_file_io():
    """Artifact IO goes through smallaxe._fs (Hadoop FileSystem), never open()/os."""
    sources = sorted(Path(smallaxe.preprocessing.__file__).parent.glob("*.py"))
    assert sources
    for source in sources:
        text = source.read_text()
        for forbidden in ("import os", "os.path", "os.makedirs", "open(", "import json"):
            assert forbidden not in text, f"{source.name} uses {forbidden!r}"


class TestLegacyArtifacts:
    """Artifacts in the smallaxe <= 0.10 layout still load and transform identically."""

    def test_imputer_fixture_is_in_the_old_layout(self):
        directory = FIXTURES / "imputer"
        assert sorted(p.name for p in directory.iterdir()) == ["imputer_metadata.json"]
        metadata = json.loads((directory / "imputer_metadata.json").read_text())
        assert "kind" not in metadata
        assert "format_version" not in metadata
        assert metadata["is_fitted"] is True

    def test_imputer_fixture_loads_and_transforms_identically(self, spark_session):
        spec = EXPECTED["imputer"]
        schema = "id int, age double, income double, city string"
        probe = spark_session.createDataFrame([tuple(r) for r in spec["probe"]], schema)

        imputer = Imputer.load(str(FIXTURES / "imputer"))

        assert imputer.numerical_fill_values == {"age": 30.0, "income": 60000.0}
        assert imputer.categorical_fill_values == {"city": "Paris"}
        result = imputer.transform(probe)
        assert result.columns == spec["columns"]
        assert [list(row) for row in _rows(result)] == spec["output"]

    def test_resaved_imputer_fixture_uses_the_new_layout(self, spark_session, tmp_path):
        spec = EXPECTED["imputer"]
        schema = "id int, age double, income double, city string"
        probe = spark_session.createDataFrame([tuple(r) for r in spec["probe"]], schema)
        legacy = Imputer.load(str(FIXTURES / "imputer"))

        legacy.save(str(tmp_path / "resaved"))
        resaved = Imputer.load(str(tmp_path / "resaved"))

        assert (tmp_path / "resaved" / "metadata.json").exists()
        _assert_same_output(resaved.transform(probe), legacy.transform(probe))

    def test_scaler_fixture_is_in_the_old_layout(self):
        directory = FIXTURES / "scaler"
        assert sorted(p.name for p in directory.iterdir()) == [
            "scaler_metadata.json",
            "spark_model",
        ]
        metadata = json.loads((directory / "scaler_metadata.json").read_text())
        assert metadata == {"method": "standard", "numerical_cols": ["x", "y"], "is_fitted": True}
        assert (directory / "spark_model" / "metadata").is_dir()
        assert list((directory / "spark_model" / "data").glob("*.parquet"))

    def test_scaler_fixture_loads_and_transforms_identically(self, spark_session):
        directory = FIXTURES / "scaler"
        spec = EXPECTED["scaler"]
        probe = spark_session.createDataFrame(
            [tuple(r) for r in spec["probe"]], "id int, x double, y double, note string"
        )

        result = Scaler.load(str(directory)).transform(probe)

        assert result.columns == spec["columns"]
        actual = [list(row) for row in _rows(result)]
        assert len(actual) == len(spec["output"])
        for got_row, want_row in zip(actual, spec["output"]):
            for got, want in zip(got_row, want_row):
                if isinstance(want, float):
                    assert math.isclose(got, want, rel_tol=1e-12, abs_tol=1e-12)
                else:
                    assert got == want

    @pytest.mark.parametrize("method", ["standard", "minmax", "maxabs"])
    def test_scaler_in_the_0_10_layout_loads_and_transforms_identically(
        self, mixed_fit, mixed_probe, tmp_path, method
    ):
        """An artifact written exactly as smallaxe 0.10's Scaler.save wrote it."""
        clean = mixed_fit.dropna(subset=["age", "income"])
        original = Scaler(method=method).fit(clean, numerical_cols=["age", "income"])
        path = str(tmp_path / "legacy")

        # Scaler.save as of smallaxe 0.10:
        os.makedirs(path, exist_ok=True)
        metadata = {"method": method, "numerical_cols": ["age", "income"], "is_fitted": True}
        with open(os.path.join(path, "scaler_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        original._scaler_model.write().overwrite().save(os.path.join(path, "spark_model"))

        assert original._scaler_model.getInputCol() == "_assembled_features"
        loaded = Scaler.load(path)

        assert loaded.method == method
        assert loaded.numerical_cols == ["age", "income"]
        _assert_same_output(loaded.transform(mixed_probe), original.transform(mixed_probe))
