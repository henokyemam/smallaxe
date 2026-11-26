"""Print every runtime dependency of pyproject.toml pinned to its floor.

The test-lowest CI job installs the output with ``pip install -r``. Each
``"name>=floor[,...][; marker]"`` in ``[project] dependencies`` becomes
``name==floor[; marker]``: the marker is kept, so pip installs a pin only on the Pythons
it names. A dependency without a ``>=`` floor is an error, which keeps every dependency
in the form this job can follow.

Runs on the oldest supported Python (3.8), so it uses the standard library only.

Usage: ``python .github/scripts/floor_pins.py [pyproject.toml] > floors.txt``
"""

import re
import sys
from typing import List, Sequence

_DEPENDENCIES = re.compile(r"^dependencies = \[(.*?)^\]", re.S | re.M)
_FLOOR = re.compile(r"([A-Za-z0-9_.-]+)\s*>=\s*([^,\s]+)")


def floor_pins(pyproject_text: str) -> List[str]:
    """Pin each ``[project]`` dependency of a pyproject.toml to its ``>=`` floor.

    Args:
        pyproject_text: The contents of pyproject.toml.

    Returns:
        One ``name==floor`` requirement per dependency, in order, each followed by
        ``; <marker>`` when the dependency has an environment marker.

    Raises:
        ValueError: If there is no ``dependencies = [...]`` list, or a dependency has no
            ``>=`` floor.
    """
    block = _DEPENDENCIES.search(pyproject_text)
    if block is None:
        raise ValueError("no [project] dependencies list in pyproject.toml")
    code = "\n".join(line.split("#", 1)[0] for line in block.group(1).splitlines())
    pins = []
    for spec in re.findall(r'"([^"]+)"', code):
        requirement, _, marker = spec.partition(";")
        match = _FLOOR.match(requirement.strip())
        if match is None:
            raise ValueError("no >= floor in dependency: " + spec)
        pin = match.group(1) + "==" + match.group(2)
        if marker.strip():
            pin += "; " + marker.strip()
        pins.append(pin)
    return pins


def main(argv: Sequence[str]) -> int:
    """Print the floor pins of the pyproject.toml named in ``argv`` (default: ./pyproject.toml).

    Args:
        argv: The command line, ``sys.argv``.

    Returns:
        The exit status: 0, or 1 with a message on stderr when a dependency has no floor.
    """
    path = argv[1] if len(argv) > 1 else "pyproject.toml"
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    try:
        pins = floor_pins(text)
    except ValueError as exc:
        print(f"{path}: {exc}", file=sys.stderr)
        return 1
    print("\n".join(pins))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
