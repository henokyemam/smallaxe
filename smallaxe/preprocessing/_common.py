"""Helpers shared by the preprocessing steps: column naming and artifact IO."""

from typing import Any, Dict, Iterable, List, Optional

from pyspark.sql import Column
from pyspark.sql import functions as F

from smallaxe import _artifact, _fs
from smallaxe.exceptions import ValidationError


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
            or was written by a newer smallaxe.
    """
    if (
        path
        and legacy_file
        and not _fs.exists(_fs.join(path, _artifact.METADATA_FILE))
        and _fs.exists(_fs.join(path, legacy_file))
    ):
        legacy_path = _fs.join(path, legacy_file)
        try:
            metadata = _fs.read_json(legacy_path)
        except ValueError as exc:
            raise ValidationError(
                f"Cannot load {kind}: {legacy_path!r} is not valid JSON ({exc})."
            ) from exc
        if not isinstance(metadata, dict):
            raise ValidationError(f"Cannot load {kind}: {legacy_path!r} is not a JSON object.")
        return metadata
    return _artifact.read_metadata(path, kind)


def require_keys(metadata: Dict[str, Any], keys: List[str], kind: str, path: str) -> None:
    """Raise ValidationError naming the keys ``metadata`` lacks."""
    missing = [key for key in keys if key not in metadata]
    if missing:
        raise ValidationError(
            f"Cannot load {kind}: the metadata at {path!r} is missing {missing}; "
            "the artifact is incomplete or corrupted."
        )
