"""Where smallaxe's warnings point: the first caller outside smallaxe."""

import sys
from types import FrameType

__all__ = ["caller_stacklevel"]


def _is_smallaxe_frame(frame: FrameType) -> bool:
    """Whether ``frame`` runs code of the smallaxe package itself."""
    name = frame.f_globals.get("__name__", "")
    return isinstance(name, str) and (name == "smallaxe" or name.startswith("smallaxe."))


def caller_stacklevel() -> int:
    """The ``stacklevel`` that attributes a warning to the first frame outside smallaxe.

    Meant for the function that calls ``warnings.warn``: level 1 is that function, so
    the count starts at its caller and skips every smallaxe frame (nested library calls,
    a pipeline, a search). smallaxe's warnings use it, so a warning points at the user's
    line however deep inside smallaxe it was raised.

    Returns:
        The stacklevel to pass to ``warnings.warn``.
    """
    frame = sys._getframe(2)  # 0: this function, 1: the warning function, 2: its caller
    level = 2
    while frame.f_back is not None and _is_smallaxe_frame(frame):
        frame = frame.f_back
        level += 1
    return level
