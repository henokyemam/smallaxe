"""Tests for the shared internal helpers in smallaxe._checks."""

from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StructField, StructType

from smallaxe import _checks


class TestOneRow:
    """``one_row`` returns the row a global aggregate always has."""

    def test_returns_the_aggregate_row(self, spark_session):
        df = spark_session.createDataFrame([(1.0,), (3.0,)], ["x"])

        row = _checks.one_row(df.select(F.mean("x").alias("mean")))

        assert row["mean"] == 2.0

    def test_empty_input_still_has_one_row(self, spark_session):
        schema = StructType([StructField("x", DoubleType())])
        empty = spark_session.createDataFrame([], schema)

        row = _checks.one_row(empty.select(F.mean("x").alias("mean")))

        assert row["mean"] is None
