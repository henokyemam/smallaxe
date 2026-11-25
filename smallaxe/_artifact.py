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


def read_json(path: str, kind: str) -> Any:
    """Parse the JSON file at ``path``, raising ValidationError when it cannot be read.

    ``kind`` names what was being loaded, for the message.

    Raises:
        ValidationError: If the file is not valid JSON, or the filesystem refuses to read
            it (for example a Hadoop checksum mismatch after the file was edited by hand).
    """
    try:
        return _fs.read_json(path)
    except ValueError as exc:
        raise ValidationError(f"Cannot load {kind}: {path!r} is not valid JSON ({exc}).") from exc
    except _fs.jvm_error_types() as exc:
        raise ValidationError(
            f"Cannot load {kind}: {path!r} could not be read ({jvm_error_summary(exc)}). If it was "
            "edited after saving, delete the '.crc' file next to it or save the artifact again."
        ) from exc


def jvm_error_summary(exc: BaseException) -> str:
    """A one-line description of a JVM-side error (the Java exception, when Py4J has it)."""
    java_exception = getattr(exc, "java_exception", None)
    text = str(java_exception.toString() if java_exception is not None else exc).strip()
    return text.splitlines()[0] if text else type(exc).__name__


def is_unversioned(metadata: Dict[str, Any]) -> bool:
    """Whether ``metadata`` predates the header: it has neither ``format_version`` nor ``kind``."""
    return "format_version" not in metadata and "kind" not in metadata


def read_metadata(path: str, kind: str, allow_unversioned: bool = False) -> Dict[str, Any]:
    """Read ``<path>/metadata.json`` and check it holds a readable ``kind`` artifact.

    Args:
        path: The artifact directory.
        kind: The kind of artifact expected (``"model"``, ``"imputer"``, ...).
        allow_unversioned: Accept a ``metadata.json`` written before the header existed
            (neither ``format_version`` nor ``kind``, as every model artifact of smallaxe
            0.10 and earlier) and return it as is; the caller validates its keys. Only for
            kinds that have such legacy artifacts. A file with either header key is checked
            in full whatever this says.

    Returns:
        The whole metadata dict, header keys included.

    Raises:
        ValidationError: If the path or its ``metadata.json`` does not exist, the file
            cannot be read or is not a JSON object, ``kind`` differs, or ``format_version``
            is missing or newer than :data:`FORMAT_VERSION`.
    """
    if not path or not _fs.exists(path):
        raise ValidationError(f"Cannot load {kind}: path {path!r} does not exist.")
    metadata_path = _fs.join(path, METADATA_FILE)
    if not _fs.exists(metadata_path):
        raise ValidationError(
            f"Cannot load {kind}: {path!r} is not a smallaxe {kind} artifact "
            f"(expected {METADATA_FILE} in it)."
        )
    metadata = read_json(metadata_path, kind)
    if not isinstance(metadata, dict):
        raise ValidationError(f"Cannot load {kind}: {metadata_path!r} is not a JSON object.")
    if allow_unversioned and is_unversioned(metadata):
        return metadata

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
