"""smallaxe needs a classic (JVM, py4j) Spark session; Spark Connect is refused clearly.

``smallaxe._fs`` is the one place smallaxe reaches the JVM (Hadoop FileSystem IO for
every save and load). A Spark Connect session has no JVM on the client, so there it
raises ``ConfigurationError`` instead of an ``AttributeError`` from ``spark._jvm``.
The Connect sessions here are fakes: only their class's module says "Spark Connect",
which is what the check reads, so the test runs on Spark 3.3 (which has no
``pyspark.sql.connect``) as well.
"""

import sys

import pytest
from pyspark.sql import SparkSession

import smallaxe
from smallaxe import _fs
from smallaxe.exceptions import ConfigurationError, SmallaxeError

MESSAGE = "smallaxe needs a classic Spark session; Spark Connect is not supported"


def _connect_session(module: str = "pyspark.sql.connect.session") -> object:
    """An object whose class looks like PySpark's Spark Connect ``SparkSession``.

    It has none of a session's attributes: if the check let it through, the JVM access
    after it would fail with an ``AttributeError`` instead.
    """
    fake_class = type("SparkSession", (), {"__module__": module})
    return fake_class()


@pytest.fixture
def connect_active(monkeypatch):
    """Make ``SparkSession.getActiveSession()`` return a fake Connect session."""
    session = _connect_session()
    monkeypatch.setattr(SparkSession, "getActiveSession", classmethod(lambda cls: session))
    return session


class TestConnectSessionIsRefused:
    @pytest.mark.parametrize(
        "call",
        [
            lambda path: _fs.exists(path),
            lambda path: _fs.makedirs(path),
            lambda path: _fs.read_json(path),
            lambda path: _fs.write_json(path, {"a": 1}),
            lambda path: _fs.list_names(path),
            lambda path: _fs.read_bytes(path),
            lambda path: _fs.delete(path),
        ],
        ids=["exists", "makedirs", "read_json", "write_json", "list_names", "read_bytes", "delete"],
    )
    def test_every_filesystem_call_raises_configuration_error(self, connect_active, call):
        with pytest.raises(ConfigurationError) as info:
            call("/tmp/smallaxe-connect-test")
        assert str(info.value) == MESSAGE

    def test_the_error_is_a_smallaxe_error_not_an_attribute_error(self, connect_active):
        with pytest.raises(SmallaxeError):
            _fs.exists("/tmp/smallaxe-connect-test")

    def test_loading_an_artifact_raises_configuration_error(self, connect_active):
        with pytest.raises(ConfigurationError, match="Spark Connect is not supported"):
            smallaxe.load("/tmp/smallaxe-connect-test")

    def test_a_connect_session_from_the_builder_is_refused(self, monkeypatch):
        # No active session: _fs falls back to SparkSession.builder.getOrCreate(), which
        # on Spark 4 with spark.api.mode=connect (or pyspark-connect installed) returns a
        # Connect session.
        session = _connect_session()
        monkeypatch.setattr(SparkSession, "getActiveSession", classmethod(lambda cls: None))
        monkeypatch.setattr(type(SparkSession.builder), "getOrCreate", lambda self: session)
        with pytest.raises(ConfigurationError) as info:
            _fs.exists("/tmp/smallaxe-connect-test")
        assert str(info.value) == MESSAGE

    @pytest.mark.parametrize(
        "module",
        ["pyspark.sql.connect.session", "pyspark.sql.connect", "pyspark.sql.connect.client.x"],
    )
    def test_detection_reads_the_session_class_module(self, module):
        assert _fs._is_spark_connect(_connect_session(module))

    @pytest.mark.parametrize(
        "module", ["pyspark.sql.session", "pyspark.sql.classic.session", "pyspark.sql.connector"]
    )
    def test_other_modules_are_classic(self, module):
        assert not _fs._is_spark_connect(_connect_session(module))


class TestClassicSessionIsUnaffected:
    def test_the_test_session_is_classic(self, spark_session):
        assert not _fs._is_spark_connect(spark_session)

    def test_filesystem_round_trip(self, spark_session, tmp_path):
        path = str(tmp_path / "meta.json")
        _fs.write_json(path, {"a": 1})
        assert _fs.exists(path)
        assert _fs.read_json(path) == {"a": 1}

    def test_the_check_never_imports_pyspark_sql_connect(
        self, spark_session, tmp_path, monkeypatch
    ):
        # None in sys.modules makes any import of the module raise ImportError, so the
        # check, and the IO behind it, must not import it (Spark 3.3 has no such module).
        for name in [m for m in sys.modules if m.startswith("pyspark.sql.connect")]:
            monkeypatch.delitem(sys.modules, name)
        monkeypatch.setitem(sys.modules, "pyspark.sql.connect", None)
        path = str(tmp_path / "meta.json")
        _fs.write_json(path, {"b": 2})
        assert _fs.read_json(path) == {"b": 2}
        assert sys.modules["pyspark.sql.connect"] is None
