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
"""

import json
from typing import Any, Dict


def _spark() -> Any:
    """Return the active SparkSession (models are always fit through Spark)."""
    from pyspark.sql import SparkSession

    spark = SparkSession.getActiveSession()
    if spark is None:
        spark = SparkSession.builder.getOrCreate()
    return spark


def _fs_and_path(path: str):
    """Resolve the Hadoop FileSystem and Path for ``path``."""
    spark = _spark()
    jvm = spark._jvm
    hpath = jvm.org.apache.hadoop.fs.Path(path)
    fs = hpath.getFileSystem(spark._jsc.hadoopConfiguration())
    return jvm, fs, hpath


def exists(path: str) -> bool:
    """Return whether ``path`` exists on the resolved filesystem."""
    _jvm, fs, hpath = _fs_and_path(path)
    return bool(fs.exists(hpath))


def makedirs(path: str) -> None:
    """Create ``path`` (and parents) on the resolved filesystem."""
    _jvm, fs, hpath = _fs_and_path(path)
    fs.mkdirs(hpath)


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


def read_json(path: str) -> Dict[str, Any]:
    """Read and parse JSON from ``path`` on the resolved filesystem."""
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
