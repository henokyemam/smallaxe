"""Every Python code block in README.md runs, as written, against this checkout.

Each fenced ``python`` block is one test. A block runs unless the line before its fence
is a marker comment:

- ``<!-- test: skip (<reason>) -->``: the block never runs here; the reason is required
  (for example a block that needs LightGBM's JVM package or a Databricks cluster).
- ``<!-- test: needs <name> -->``: the block runs only where ``<name>`` is available. An
  algorithm name (``xgboost``) is checked with ``algorithm.get(name).available()``;
  any other name must be an importable module (``plotly``).

So a new block without a marker is tested by default, and a marker that is misspelled
or not followed by a block fails the suite instead of skipping silently.

Each block runs in a fresh namespace, with the test session's SparkSession active (so
``SparkSession.builder.getOrCreate()`` and the dataset loaders use it), with
``tempfile.mkdtemp()`` pointing into the test's temporary directory, and with plotly's
``Figure.show`` replaced by a no-op so no browser opens. The global seed and verbosity
are restored afterwards.
"""

import importlib.util
import re
import tempfile
from pathlib import Path
from typing import List, NamedTuple, Optional

import pytest

import smallaxe
from smallaxe.training import algorithm

README = Path(__file__).resolve().parent.parent / "README.md"
MARKER = re.compile(r"^<!--\s*test:\s*(?P<body>.*?)\s*-->$")
SKIP = re.compile(r"^skip\s*\((?P<reason>.*\S.*)\)$")
NEEDS = re.compile(r"^needs\s+(?P<name>[A-Za-z_][A-Za-z0-9_.]*)$")
QUICK_START_LIMIT = 20  # Goals.md, first v1 acceptance criterion: "under 20 lines"


class Block(NamedTuple):
    line: int  # 1-based line of the first source line
    source: str
    marker: Optional[str]  # the marker's body, e.g. "skip (needs LightGBM)"
    heading: str  # the nearest "## " heading above the block


def _blocks(text: str) -> List[Block]:
    """The fenced ``python`` blocks of ``text``, with their marker and section."""
    lines = text.splitlines()
    blocks: List[Block] = []
    heading = ""
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("## "):
            heading = line[3:].strip()
        if line.strip() == "```python":
            start = i + 1
            end = start
            while end < len(lines) and lines[end].strip() != "```":
                end += 1
            if end == len(lines):
                raise AssertionError(f"README.md:{i + 1}: unterminated ```python block")
            previous = i - 1
            while previous >= 0 and not lines[previous].strip():
                previous -= 1
            match = MARKER.match(lines[previous].strip()) if previous >= 0 else None
            marker = match.group("body") if match else None
            blocks.append(Block(start + 1, "\n".join(lines[start:end]) + "\n", marker, heading))
            i = end
        i += 1
    return blocks


def _markers(text: str) -> List[int]:
    """1-based line numbers of every ``<!-- test: ... -->`` marker in ``text``."""
    return [n for n, line in enumerate(text.splitlines(), 1) if MARKER.match(line.strip())]


BLOCKS = _blocks(README.read_text(encoding="utf-8"))


def _available(name: str) -> bool:
    if name in algorithm.names():
        return algorithm.get(name).available()
    return importlib.util.find_spec(name) is not None


def _skip_reason(marker: Optional[str]) -> Optional[str]:
    """Why a block does not run here, or None when it runs."""
    if marker is None:
        return None
    skip = SKIP.match(marker)
    if skip:
        return f"README marker: {skip.group('reason')}"
    needs = NEEDS.match(marker)
    if needs:
        name = needs.group("name")
        return None if _available(name) else f"README marker: needs {name}, not installed"
    raise AssertionError(f"unknown README test marker {marker!r}")


@pytest.fixture
def readme_environment(spark_session, tmp_path, monkeypatch):
    """The active session, a private temp directory, no browser, and clean globals."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    if importlib.util.find_spec("plotly") is not None:
        import plotly.graph_objects as go

        monkeypatch.setattr(go.Figure, "show", lambda self, *args, **kwargs: None)
    seed, level = smallaxe.get_seed(), smallaxe.get_verbosity()
    smallaxe.set_seed(None)
    smallaxe.set_verbosity("normal")
    yield spark_session
    smallaxe.set_seed(seed)
    smallaxe.set_verbosity(level)


def _run(block: Block) -> None:
    # Pad with newlines so a traceback names the block's own line in README.md.
    padded = "\n" * (block.line - 1) + block.source
    exec(compile(padded, str(README), "exec"), {"__name__": "__readme__"})


@pytest.mark.parametrize("block", BLOCKS, ids=[f"README.md:{b.line}" for b in BLOCKS])
def test_readme_block_runs(readme_environment, block):
    reason = _skip_reason(block.marker)
    if reason is not None:
        pytest.skip(reason)
    _run(block)


def test_readme_has_python_blocks():
    assert len(BLOCKS) >= 10
    assert any(block.marker is None for block in BLOCKS)


def test_every_marker_belongs_to_a_block_and_is_well_formed():
    text = README.read_text(encoding="utf-8")
    lines = text.splitlines()
    for number in _markers(text):
        following = number  # index of the line after the marker
        while following < len(lines) and not lines[following].strip():
            following += 1
        assert (
            following < len(lines) and lines[following].strip() == "```python"
        ), f"README.md:{number}: a test marker must come right before a ```python block"
    for block in BLOCKS:
        if block.marker is not None:
            assert SKIP.match(block.marker) or NEEDS.match(block.marker), (
                f"README.md:{block.line}: marker {block.marker!r} is neither "
                "'skip (<reason>)' nor 'needs <name>'"
            )


def test_quick_start_is_one_runnable_block_under_20_lines():
    quick_start = [block for block in BLOCKS if block.heading == "Quick Start"]
    assert len(quick_start) == 1
    block = quick_start[0]
    assert block.marker is None, "the Quick Start must run in the test suite"
    n_lines = len(block.source.splitlines())
    assert n_lines < QUICK_START_LIMIT, f"Quick Start is {n_lines} lines"
    for step in ("random_forest(", ".fit(", "validation_scores", ".save(", "load(", ".predict("):
        assert step in block.source, step


def test_a_failing_block_fails(readme_environment):
    """The runner propagates a block's error, pointing at the README line."""
    broken = Block(line=7, source="raise RuntimeError('broken example')\n", marker=None, heading="")
    with pytest.raises(RuntimeError, match="broken example") as excinfo:
        _run(broken)
    assert excinfo.traceback[-1].lineno + 1 == 7
