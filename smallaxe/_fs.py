"""Filesystem-agnostic JSON IO for model artifacts.

smallaxe models are always trained through Spark, and Spark persists models via
the Hadoop ``FileSystem`` layer. Model *metadata* must live on the **same
filesystem** as those model files, or save/load breaks on any deployment where
the driver-local filesystem differs from Spark's default filesystem (Databricks
DBFS, S3, HDFS, ...). Routing metadata IO through the Hadoop ``FileSystem`` API —
the same layer Spark uses — keeps metadata and model together and makes
``save``/``load`` round-trip on local disk, DBFS, S3, and HDFS alike.

Only core ``java.io`` classes are used (always on the Spark classpath); no
optional dependencies.

This module is the one place smallaxe reaches the JVM (``spark._jvm``,
``spark._jsc``), so it needs a classic Spark session: the default of ``pip install
pyspark`` on Spark 3.3 to 4.x. A Spark Connect session has no JVM on the client; with
one active, every function here raises ``ConfigurationError`` (see :func:`_spark`).
"""

import json
from typing import Any, Dict, List, Tuple, Type

from smallaxe.exceptions import ConfigurationError

__all__ = [
    "join",
    "exists",
    "makedirs",
    "delete",
    "list_names",
    "read_bytes",
    "jvm_error_types",
    "write_json",
    "read_json",
]


_CONNECT_UNSUPPORTED = "smallaxe needs a classic Spark session; Spark Connect is not supported"


def _is_spark_connect(spark: Any) -> bool:
    """Whether ``spark`` is a Spark Connect session rather than a classic (JVM) one.

    Reads the module of the session's class (``pyspark.sql.connect.session`` for
    Connect) instead of an ``isinstance`` check, so it never imports
    ``pyspark.sql.connect``, which PySpark 3.3 does not have and which needs the
    Connect client's dependencies (grpcio, pyarrow) where it does exist.
    """
    module = type(spark).__module__ or ""
    return module == "pyspark.sql.connect" or module.startswith("pyspark.sql.connect.")


def _spark() -> Any:
    """Return the active classic SparkSession (models are always fit through Spark).

    Raises:
        ConfigurationError: If the session is a Spark Connect session, which has no JVM
            to reach the Hadoop FileSystem through.
    """
    from pyspark.sql import SparkSession

    spark = SparkSession.getActiveSession()
    if spark is None:
        spark = SparkSession.builder.getOrCreate()
    if _is_spark_connect(spark):
        raise ConfigurationError(_CONNECT_UNSUPPORTED)
    return spark


def _fs_and_path(path: str) -> Tuple[Any, Any, Any]:
    """Resolve the Hadoop FileSystem and Path for ``path``.

    Returns:
        The Py4J JVM view, the Hadoop ``FileSystem`` and the Hadoop ``Path`` (Py4J proxies,
        which have no static types).
    """
    spark = _spark()
    jvm = spark._jvm
    hpath = jvm.org.apache.hadoop.fs.Path(path)
    fs = hpath.getFileSystem(spark._jsc.hadoopConfiguration())
    return jvm, fs, hpath


def join(path: str, *parts: str) -> str:
    """Join ``parts`` onto ``path`` with ``/``.

    Unlike ``os.path.join`` this never uses the driver's path separator, so it is safe
    for URIs such as ``dbfs:/...``, ``s3://...`` and ``hdfs://...`` as well as local paths.
    """
    result = path
    for part in parts:
        result = result.rstrip("/") + "/" + part.lstrip("/")
    return result


def exists(path: str) -> bool:
    """Return whether ``path`` exists on the resolved filesystem."""
    _jvm, fs, hpath = _fs_and_path(path)
    return bool(fs.exists(hpath))


def makedirs(path: str) -> None:
    """Create ``path`` (and parents) on the resolved filesystem."""
    _jvm, fs, hpath = _fs_and_path(path)
    fs.mkdirs(hpath)


def delete(path: str, recursive: bool = False) -> None:
    """Delete ``path`` (with its Hadoop checksum file) on the resolved filesystem, if present."""
    _jvm, fs, hpath = _fs_and_path(path)
    fs.delete(hpath, recursive)


def list_names(path: str) -> List[str]:
    """Names of the entries directly inside the directory ``path`` (Hadoop ``.crc`` files too)."""
    _jvm, fs, hpath = _fs_and_path(path)
    return [str(status.getPath().getName()) for status in fs.listStatus(hpath)]


def read_bytes(path: str) -> bytes:
    """Read the whole file at ``path`` on the resolved filesystem."""
    jvm, fs, hpath = _fs_and_path(path)
    stream = fs.open(hpath)
    try:
        buffer = jvm.java.io.ByteArrayOutputStream()
        jvm.org.apache.hadoop.io.IOUtils.copyBytes(stream, buffer, 65536, False)
        return bytes(buffer.toByteArray())
    finally:
        stream.close()


def jvm_error_types() -> Tuple[Type[BaseException], ...]:
    """Exception types for a failure raised on the JVM side of a Spark or Hadoop call.

    ``Py4JJavaError`` is what a raw Hadoop ``FileSystem`` call raises (for example a
    ``ChecksumException`` when a file was edited after Hadoop wrote its ``.crc``);
    ``CapturedException`` is what PySpark raises when it converts a JVM error (for example
    from ``MLReader.load``). Py4J connection errors are deliberately not included.
    """
    from py4j.protocol import Py4JJavaError
    from pyspark.sql.utils import CapturedException

    return (Py4JJavaError, CapturedException)


def write_json(path: str, obj: Dict[str, Any]) -> None:
    """Write ``obj`` as JSON to ``path`` on the resolved filesystem (overwrite)."""
    jvm, fs, hpath = _fs_and_path(path)
    parent = hpath.getParent()
    if parent is not None:
        fs.mkdirs(parent)
    text = json.dumps(obj, indent=2, default=str)
    stream = fs.create(hpath, True)  # overwrite=True
    writer = jvm.java.io.OutputStreamWriter(stream, "UTF-8")
    try:
        writer.write(text)
    finally:
        writer.close()  # flushes and closes the underlying stream


def read_json(path: str) -> object:
    """Read and parse JSON from ``path`` on the resolved filesystem.

    Args:
        path: The file to read, local or a Hadoop URI.

    Returns:
        The parsed value, of whatever JSON type the file holds (typed ``object``, so a
        caller narrows it, for example with ``isinstance(value, dict)``, before use).
    """
    jvm, fs, hpath = _fs_and_path(path)
    stream = fs.open(hpath)
    reader = jvm.java.io.BufferedReader(jvm.java.io.InputStreamReader(stream, "UTF-8"))
    try:
        chunks = []
        line = reader.readLine()
        while line is not None:
            chunks.append(line)
            line = reader.readLine()
    finally:
        reader.close()
    return json.loads("\n".join(chunks))
