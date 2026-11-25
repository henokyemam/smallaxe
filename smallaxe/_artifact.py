"""Versioned ``metadata.json`` for saved smallaxe artifacts.

Every artifact smallaxe writes is a directory whose ``metadata.json`` records:

- ``format_version``: the layout version of the artifact (:data:`FORMAT_VERSION`).
- ``kind``: what was saved (for example ``"imputer"``), so loading the wrong kind of
  artifact fails with a clear message instead of a ``KeyError``.
- ``smallaxe_version``: the smallaxe release that wrote it.

plus the caller's payload. All IO goes through :mod:`smallaxe._fs` (the Hadoop
FileSystem layer), so metadata lands on the same filesystem as any Spark model saved next
to it.
"""

from typing import Any, Dict

from smallaxe import _fs
from smallaxe.exceptions import ValidationError

FORMAT_VERSION = 1
METADATA_FILE = "metadata.json"
RESERVED_KEYS = ("format_version", "kind", "smallaxe_version")


def _smallaxe_version() -> str:
    from smallaxe import __version__

    return __version__


def write_metadata(path: str, kind: str, payload: Dict[str, Any]) -> None:
    """Write ``<path>/metadata.json`` with the header keys and ``payload``.

    Raises:
        ValidationError: If ``path`` is empty or ``payload`` uses a reserved key.
    """
    if not path:
        raise ValidationError("Path cannot be empty.")
    clashes = [key for key in RESERVED_KEYS if key in payload]
    if clashes:
        raise ValidationError(f"Artifact payload may not use the reserved keys {clashes}.")
    document: Dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "kind": kind,
        "smallaxe_version": _smallaxe_version(),
    }
    document.update(payload)
    _fs.write_json(_fs.join(path, METADATA_FILE), document)


def read_metadata(path: str, kind: str) -> Dict[str, Any]:
    """Read ``<path>/metadata.json`` and check it holds a readable ``kind`` artifact.

    Returns:
        The whole metadata dict, header keys included.

    Raises:
        ValidationError: If the path or its ``metadata.json`` does not exist, the file is
            not a JSON object, ``kind`` differs, or ``format_version`` is missing or newer
            than :data:`FORMAT_VERSION`.
    """
    if not path or not _fs.exists(path):
        raise ValidationError(f"Cannot load {kind}: path {path!r} does not exist.")
    metadata_path = _fs.join(path, METADATA_FILE)
    if not _fs.exists(metadata_path):
        raise ValidationError(
            f"Cannot load {kind}: {path!r} is not a smallaxe {kind} artifact "
            f"(expected {METADATA_FILE} in it)."
        )
    try:
        metadata = _fs.read_json(metadata_path)
    except ValueError as exc:
        raise ValidationError(
            f"Cannot load {kind}: {metadata_path!r} is not valid JSON ({exc})."
        ) from exc
    if not isinstance(metadata, dict):
        raise ValidationError(f"Cannot load {kind}: {metadata_path!r} is not a JSON object.")

    saved_kind = metadata.get("kind")
    if saved_kind != kind:
        raise ValidationError(
            f"Cannot load {kind}: {path!r} holds a {saved_kind!r} artifact, not a {kind!r} one."
        )

    version = metadata.get("format_version")
    if isinstance(version, bool) or not isinstance(version, int):
        raise ValidationError(
            f"Cannot load {kind}: {metadata_path!r} has no integer 'format_version'."
        )
    if version > FORMAT_VERSION:
        raise ValidationError(
            f"Cannot load {kind}: {path!r} was written by a newer smallaxe "
            f"(smallaxe_version {metadata.get('smallaxe_version')!r}, artifact "
            f"format_version {version}); this smallaxe ({_smallaxe_version()}) reads "
            f"format_version {FORMAT_VERSION} or older. Upgrade smallaxe to load it."
        )
    return metadata
