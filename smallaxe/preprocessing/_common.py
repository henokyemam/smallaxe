"""Helpers shared by the preprocessing steps: column naming and artifact IO."""

from typing import Any, Dict, Iterable, List, Optional

from pyspark.sql import Column
from pyspark.sql import functions as F

from smallaxe import _artifact, _fs
from smallaxe.exceptions import ValidationError

__all__ = [
    "column_ref",
    "unique_column_name",
    "read_step_metadata",
    "column_list",
    "require_keys",
]


def column_ref(name: str) -> Column:
    """Reference column ``name`` literally, even when it contains dots or backticks."""
    return F.col("`" + name.replace("`", "``") + "`")


def unique_column_name(base: str, taken: Iterable[str]) -> str:
    """Return ``base``, or ``base_<n>``, whichever is not in ``taken``.

    Spark resolves column names case-insensitively by default, so the comparison is too.
    """
    lowered = {name.lower() for name in taken}
    name = base
    n = 0
    while name.lower() in lowered:
        name = f"{base}_{n}"
        n += 1
    return name


def read_step_metadata(path: str, kind: str, legacy_file: Optional[str] = None) -> Dict[str, Any]:
    """Read a preprocessing step's metadata, accepting the <= 0.10 layout too.

    A directory holding ``metadata.json`` is read through :func:`_artifact.read_metadata`.
    A directory that holds only ``legacy_file`` (``imputer_metadata.json`` /
    ``scaler_metadata.json``, written by smallaxe 0.10 and earlier) is read as-is.

    Raises:
        ValidationError: If the path is missing, holds neither file, is the wrong kind,
            or was written by a newer smallaxe, or ``path`` is not a string.
    """
    _artifact.check_path(path, allow_empty=True)
    if (
        path
        and legacy_file
        and not _fs.exists(_fs.join(path, _artifact.METADATA_FILE))
        and _fs.exists(_fs.join(path, legacy_file))
    ):
        legacy_path = _fs.join(path, legacy_file)
        metadata = _artifact.read_json(legacy_path, kind)
        if not isinstance(metadata, dict):
            raise ValidationError(f"Cannot load {kind}: {legacy_path!r} is not a JSON object.")
        return metadata
    return _artifact.read_metadata(path, kind)


def column_list(
    metadata: Dict[str, Any], key: str, kind: str, path: str, allow_empty: bool = False
) -> List[str]:
    """``metadata[key]`` as a list of column names, or ValidationError if it is not one."""
    value = metadata[key]
    if (
        not isinstance(value, list)
        or not all(isinstance(col, str) for col in value)
        or (not value and not allow_empty)
    ):
        expected = "a list of column names" if allow_empty else "a non-empty list of column names"
        raise ValidationError(
            f"Cannot load {kind}: {key!r} at {path!r} must be {expected}, got {value!r}; "
            "the artifact is corrupted."
        )
    return list(value)


def require_keys(metadata: Dict[str, Any], keys: List[str], kind: str, path: str) -> None:
    """Raise ValidationError naming the keys ``metadata`` lacks."""
    missing = [key for key in keys if key not in metadata]
    if missing:
        raise ValidationError(
            f"Cannot load {kind}: the metadata at {path!r} is missing {missing}; "
            "the artifact is incomplete or corrupted."
        )
