"""smallaxe._sql: Spark 3's default string-to-integer cast, in and out of ANSI mode."""

import pyspark
import pytest
from pyspark.sql import functions as F
from pyspark.sql.types import ByteType, IntegerType, LongType, ShortType

from smallaxe._sql import integral_from_string, quoted

ANSI = "spark.sql.ansi.enabled"

# (string, value as tinyint, smallint, int, bigint): Spark 3.5's default cast.
SPELLINGS = [
    ("7", 7, 7, 7, 7),
    (" 7", 7, 7, 7, 7),
    ("\t7\n", 7, 7, 7, 7),
    ("\x007\x7f", 7, 7, 7, 7),
    ("7.", 7, 7, 7, 7),
    ("7.9", 7, 7, 7, 7),
    ("-7.9", -7, -7, -7, -7),
    ("7.9999999999999999999", 7, 7, 7, 7),
    (".5", 0, 0, 0, 0),
    (".", 0, 0, 0, 0),
    ("+.", 0, 0, 0, 0),
    ("-0", 0, 0, 0, 0),
    ("00012", 12, 12, 12, 12),
    ("0000000000000000000000000007", 7, 7, 7, 7),
    ("128", None, 128, 128, 128),
    ("-129", None, -129, -129, -129),
    ("32768", None, None, 32768, 32768),
    ("2147483648", None, None, None, 2147483648),
    ("-2147483648", None, None, -2147483648, -2147483648),
    ("9223372036854775807", None, None, None, 9223372036854775807),
    ("-9223372036854775808", None, None, None, -9223372036854775808),
    ("9223372036854775808", None, None, None, None),
    ("99999999999999999999", None, None, None, None),
    ("+", None, None, None, None),
    ("-", None, None, None, None),
    ("", None, None, None, None),
    ("  ", None, None, None, None),
    ("1e3", None, None, None, None),
    ("0x10", None, None, None, None),
    ("1,000", None, None, None, None),
    ("+-7", None, None, None, None),
    ("7.9.9", None, None, None, None),
    ("7.x", None, None, None, None),
    ("٧", None, None, None, None),
    (" 7", None, None, None, None),
    ("7 ", None, None, None, None),
    (None, None, None, None, None),
]
TYPES = [ByteType(), ShortType(), IntegerType(), LongType()]


def _cast(spark, ansi, column):
    previous = spark.conf.get(ANSI)
    spark.conf.set(ANSI, ansi)
    try:
        frame = spark.createDataFrame([(s,) for s, *_ in SPELLINGS], "s string")
        return [list(row) for row in frame.select(*column(F.col("s"))).collect()]
    finally:
        spark.conf.set(ANSI, previous)


@pytest.mark.parametrize("ansi", ["false", "true"])
def test_integral_from_string_is_spark_3_default_cast(spark_session, ansi):
    got = _cast(spark_session, ansi, lambda s: [integral_from_string(s, t) for t in TYPES])
    assert got == [list(values) for _, *values in SPELLINGS]


@pytest.mark.skipif(
    int(pyspark.__version__.split(".")[0]) >= 4,
    reason="the reference is Spark 3's own default cast; the pinned values hold on every Spark",
)
def test_pinned_values_are_this_spark_3_default_cast(spark_session):
    assert _cast(spark_session, "false", lambda s: [s.cast(t) for t in TYPES]) == [
        list(values) for _, *values in SPELLINGS
    ]


def test_quoted_names_one_column():
    assert quoted("a.b") == "`a.b`"
    assert quoted("a`b") == "`a``b`"
