"""Tests for the versioned artifact metadata helper (smallaxe._artifact)."""

import json

import pytest

import smallaxe
from smallaxe import _artifact, _fs
from smallaxe.exceptions import ValidationError


@pytest.fixture(autouse=True)
def _spark(spark_session):
    """_fs resolves the Hadoop FileSystem through the active SparkSession."""
    return spark_session


def _write_raw(directory, document):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "metadata.json").write_text(json.dumps(document))


class TestJoin:
    def test_local_path(self):
        assert _fs.join("/tmp/a", "b", "c.json") == "/tmp/a/b/c.json"

    def test_uri_paths_keep_their_scheme(self):
        assert _fs.join("dbfs:/models/m", "metadata.json") == "dbfs:/models/m/metadata.json"
        assert _fs.join("s3://bucket/", "/x") == "s3://bucket/x"

    def test_trailing_slash(self):
        assert _fs.join("/tmp/a/", "b") == "/tmp/a/b"


class TestWriteMetadata:
    def test_round_trip_adds_header_keys(self, tmp_path):
        path = str(tmp_path / "artifact")
        _artifact.write_metadata(path, "imputer", {"a": 1, "b": {"c": [1, 2]}})

        metadata = _artifact.read_metadata(path, "imputer")

        assert metadata == {
            "format_version": _artifact.FORMAT_VERSION,
            "kind": "imputer",
            "smallaxe_version": smallaxe.__version__,
            "a": 1,
            "b": {"c": [1, 2]},
        }

    def test_format_version_is_one(self):
        assert _artifact.FORMAT_VERSION == 1

    def test_writes_plain_json_file(self, tmp_path):
        path = tmp_path / "artifact"
        _artifact.write_metadata(str(path), "encoder", {"x": "y"})

        on_disk = json.loads((path / "metadata.json").read_text())
        assert on_disk["kind"] == "encoder"
        assert on_disk["format_version"] == 1
        assert on_disk["x"] == "y"

    def test_accepts_a_hadoop_uri(self, tmp_path):
        """The path goes through the Hadoop FileSystem layer, so a file: URI works."""
        uri = "file://" + str(tmp_path / "artifact")
        _artifact.write_metadata(uri, "scaler", {"x": 1})

        assert (tmp_path / "artifact" / "metadata.json").exists()
        assert _artifact.read_metadata(uri, "scaler")["x"] == 1

    def test_overwrites_existing_metadata(self, tmp_path):
        path = str(tmp_path / "artifact")
        _artifact.write_metadata(path, "imputer", {"x": 1})
        _artifact.write_metadata(path, "imputer", {"x": 2})

        assert _artifact.read_metadata(path, "imputer")["x"] == 2

    @pytest.mark.parametrize("key", ["format_version", "kind", "smallaxe_version"])
    def test_reserved_payload_key_raises(self, tmp_path, key):
        with pytest.raises(ValidationError, match="reserved"):
            _artifact.write_metadata(str(tmp_path / "a"), "imputer", {key: "x"})

    def test_empty_path_raises(self):
        with pytest.raises(ValidationError, match="empty"):
            _artifact.write_metadata("", "imputer", {})


class TestReadMetadata:
    def test_missing_path_raises(self, tmp_path):
        missing = str(tmp_path / "nope")
        with pytest.raises(ValidationError, match="does not exist") as info:
            _artifact.read_metadata(missing, "imputer")
        assert missing in str(info.value)

    def test_empty_path_raises(self):
        with pytest.raises(ValidationError, match="does not exist"):
            _artifact.read_metadata("", "imputer")

    def test_directory_without_metadata_raises(self, tmp_path):
        (tmp_path / "empty").mkdir()
        with pytest.raises(ValidationError, match="metadata.json") as info:
            _artifact.read_metadata(str(tmp_path / "empty"), "scaler")
        assert "scaler" in str(info.value)

    def test_wrong_kind_raises(self, tmp_path):
        path = str(tmp_path / "artifact")
        _artifact.write_metadata(path, "scaler", {})

        with pytest.raises(ValidationError, match="'scaler' artifact, not a 'encoder'"):
            _artifact.read_metadata(path, "encoder")

    def test_missing_kind_raises(self, tmp_path):
        _write_raw(tmp_path / "a", {"format_version": 1})
        with pytest.raises(ValidationError, match="None"):
            _artifact.read_metadata(str(tmp_path / "a"), "encoder")

    def test_newer_format_version_raises(self, tmp_path):
        _write_raw(
            tmp_path / "a", {"format_version": 999, "kind": "encoder", "smallaxe_version": "9.0"}
        )
        with pytest.raises(ValidationError, match="newer smallaxe") as info:
            _artifact.read_metadata(str(tmp_path / "a"), "encoder")
        message = str(info.value)
        assert "999" in message
        assert "format_version 1" in message
        assert "9.0" in message

    def test_older_format_version_is_read(self, tmp_path):
        _write_raw(tmp_path / "a", {"format_version": 0, "kind": "encoder", "x": 3})
        assert _artifact.read_metadata(str(tmp_path / "a"), "encoder")["x"] == 3

    @pytest.mark.parametrize("version", [None, "1", True, 1.0])
    def test_non_integer_format_version_raises(self, tmp_path, version):
        document = {"kind": "encoder"}
        if version is not None:
            document["format_version"] = version
        _write_raw(tmp_path / "a", document)
        with pytest.raises(ValidationError, match="format_version"):
            _artifact.read_metadata(str(tmp_path / "a"), "encoder")

    def test_invalid_json_raises(self, tmp_path):
        (tmp_path / "a").mkdir()
        (tmp_path / "a" / "metadata.json").write_text("{not json")
        with pytest.raises(ValidationError, match="not valid JSON"):
            _artifact.read_metadata(str(tmp_path / "a"), "encoder")

    def test_non_object_json_raises(self, tmp_path):
        (tmp_path / "a").mkdir()
        (tmp_path / "a" / "metadata.json").write_text("[1, 2]")
        with pytest.raises(ValidationError, match="JSON object"):
            _artifact.read_metadata(str(tmp_path / "a"), "encoder")
