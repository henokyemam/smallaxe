"""Column expressions that give Spark 3's default result in and out of ANSI mode.

Spark 4 turns ANSI SQL mode on by default, and ANSI mode
changes casts: a string that is not a number raises instead of becoming null, and the
string-to-integer cast no longer reads a fraction (``"7.9"`` is 7 without ANSI mode, an
error with it). smallaxe's results are defined as Spark 3's default (non-ANSI) results;
the helpers here spell those casts out so they do not depend on the option.
"""

from typing import Dict, Tuple, Type

from pyspark.sql import Column
from pyspark.sql import functions as F
from pyspark.sql.types import (
    ByteType,
    DecimalType,
    IntegerType,
    IntegralType,
    LongType,
    ShortType,
)

__all__ = ["quoted", "integral_from_string"]


def quoted(name: str) -> str:
    """``name`` as a SQL identifier, so a name holding a dot or a backtick is one column."""
    return "`" + name.replace("`", "``") + "`"


# The bytes Spark trims from both ends of a string before reading it as an integer:
# whitespace and ISO control characters among ASCII (0x00 to 0x20, and 0x7F).
_PADDING = "[\\x00-\\x20\\x7f]"
# What remains must be an optional sign, digits and an optional fraction ("7", "-7.",
# ".5", "+."); the fraction is dropped. ``\z`` is the absolute end (``$`` would also
# match before a final line terminator such as U+2028).
_INTEGRAL_SPELLING = "^[+-]?[0-9]*(\\.[0-9]*)?\\z"
# Every integral type fits in 19 digits; a longer integer part overflows each of them.
_MAX_DIGITS = 19
_WIDE = DecimalType(20, 0)
_BOUNDS: Dict[Type[IntegralType], Tuple[int, int]] = {
    ByteType: (-(2**7), 2**7 - 1),
    ShortType: (-(2**15), 2**15 - 1),
    IntegerType: (-(2**31), 2**31 - 1),
    LongType: (-(2**63), 2**63 - 1),
}


def integral_from_string(column: Column, data_type: IntegralType) -> Column:
    """``column`` (a string) cast to ``data_type`` as Spark 3's default cast does, in either mode.

    Without ANSI mode, Spark reads a string as an integer after trimming the ASCII
    whitespace and control characters at both ends: an optional sign, digits, and an
    optional ``.`` followed by digits, which is dropped (``"7.9"`` is 7, ``"."`` and
    ``".5"`` are 0). Anything else (``""``, ``"+"``, ``"1e3"``, ``"0x10"``, non-ASCII
    digits) and a value outside the type's range are null. ANSI mode rejects the fraction
    and raises on the rest, so the cast is built from steps that cannot fail: a pattern
    check, the integer part as at most 19 digits, a ``decimal(20,0)`` that holds every
    such number, and a range check before the last cast.

    Args:
        column: A string column expression.
        data_type: The integral type to cast to (``tinyint`` to ``bigint``).

    Returns:
        The integral column expression; null where Spark 3's default cast gives null.
    """
    low, high = _BOUNDS[type(data_type)]
    trimmed = F.regexp_replace(column, f"^{_PADDING}+|{_PADDING}+\\z", "")
    spelled = trimmed.rlike(_INTEGRAL_SPELLING) & ~trimmed.isin("", "+", "-")
    sign = F.regexp_extract(trimmed, "^([+-]?)", 1)
    digits = F.regexp_extract(trimmed, "^[+-]?0*([0-9]*)", 1)
    text = F.when(
        spelled & (F.length(digits) <= _MAX_DIGITS),
        F.concat(sign, F.when(digits == "", F.lit("0")).otherwise(digits)),
    )
    wide = text.cast(_WIDE)
    in_range = F.when((wide >= F.lit(low)) & (wide <= F.lit(high)), wide)
    return in_range.cast(data_type)
