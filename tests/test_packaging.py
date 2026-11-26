"""The declared dependencies: what the test-lowest CI job installs, and what each Python needs.

These read pyproject.toml and requirements-dev.txt; they need no Spark session.
"""

import importlib.util
import re
from pathlib import Path
from types import ModuleType
from typing import Dict, List

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

ROOT = Path(__file__).resolve().parent.parent
SUPPORTED_PYTHONS = ["3.8", "3.9", "3.10", "3.11", "3.12"]


def _load_floor_pins() -> ModuleType:
    path = ROOT / ".github" / "scripts" / "floor_pins.py"
    spec = importlib.util.spec_from_file_location("floor_pins", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


floor_pins = _load_floor_pins().floor_pins


def _pyproject() -> str:
    return (ROOT / "pyproject.toml").read_text(encoding="utf-8")


def _dependencies() -> List[Requirement]:
    """The ``[project] dependencies`` of pyproject.toml, parsed."""
    block = re.search(r"^dependencies = \[(.*?)^\]", _pyproject(), re.S | re.M)
    assert block is not None
    code = "\n".join(line.split("#", 1)[0] for line in block.group(1).splitlines())
    return [Requirement(spec) for spec in re.findall(r'"([^"]+)"', code)]


def _extra(name: str) -> List[Requirement]:
    """The requirements of one ``[project.optional-dependencies]`` extra, parsed."""
    block = re.search(rf"^{name} = \[(.*?)^\]", _pyproject(), re.S | re.M)
    assert block is not None, name
    code = "\n".join(line.split("#", 1)[0] for line in block.group(1).splitlines())
    return [Requirement(spec) for spec in re.findall(r'"([^"]+)"', code)]


def _on(python: str, requirements: List[Requirement]) -> Dict[str, Requirement]:
    """The requirements whose marker holds on ``python``, by name (one each)."""
    applying: Dict[str, Requirement] = {}
    for requirement in requirements:
        marker = requirement.marker
        if marker is None or marker.evaluate({"python_version": python}):
            assert requirement.name not in applying, (python, requirement.name)
            applying[requirement.name] = requirement
    return applying


class TestFloorPins:
    """``.github/scripts/floor_pins.py``, which the test-lowest CI job runs."""

    def test_pins_each_dependency_to_its_floor_keeping_its_marker(self):
        text = (
            "[project]\n"
            "dependencies = [\n"
            '    "pyspark>=3.3,<4.0",\n'
            '    # a comment with a "quoted" word\n'
            "    \"setuptools>=56.0.0,<82; python_version < '3.10'\",\n"
            "]\n"
        )

        assert floor_pins(text) == [
            "pyspark==3.3",
            "setuptools==56.0.0; python_version < '3.10'",
        ]

    def test_a_dependency_without_a_floor_is_an_error(self):
        text = 'dependencies = [\n    "pyspark<4.0",\n]\n'

        with pytest.raises(ValueError, match="no >= floor in dependency: pyspark<4.0"):
            floor_pins(text)

    def test_every_declared_dependency_has_a_floor(self):
        pins = floor_pins(_pyproject())

        assert len(pins) == len(_dependencies())
        for pin, requirement in zip(pins, _dependencies()):
            pinned = Requirement(pin)
            assert pinned.name == requirement.name
            assert str(pinned.marker) == str(requirement.marker)
            (version,) = [spec.version for spec in pinned.specifier]
            assert requirement.specifier.contains(version), (pin, str(requirement))

    def test_the_lowest_python_gets_one_pin_per_dependency(self):
        pins = [Requirement(pin) for pin in floor_pins(_pyproject())]

        assert sorted(_on("3.8", pins)) == [
            "hyperopt",
            "numpy",
            "pandas",
            "py4j",
            "pyspark",
            "setuptools",
        ]


class TestDeclaredDependencies:
    """Each supported Python gets a dependency set that imports."""

    @pytest.mark.parametrize("python", SUPPORTED_PYTHONS)
    def test_hyperopt_that_imports_pkg_resources_comes_with_a_setuptools_that_has_it(self, python):
        # hyperopt 0.2.x imports pkg_resources (hyperopt/atpe.py), which setuptools 82
        # removed; hyperopt 0.3.0 does not, but needs Python 3.10.
        applying = _on(python, _dependencies())
        hyperopt = applying["hyperopt"]
        if not hyperopt.specifier.contains("0.2.7"):
            assert all(Version(spec.version) >= Version("0.3.0") for spec in hyperopt.specifier)
            return
        setuptools = applying.get("setuptools")
        assert setuptools is not None, f"Python {python}: hyperopt 0.2.7 needs setuptools"
        assert not setuptools.specifier.contains("82.0.0")
        assert setuptools.specifier.contains("81.0.0")

    @pytest.mark.parametrize("python", ["3.8", "3.9"])
    def test_hyperopt_floor_is_installable_before_python_3_10(self, python):
        # hyperopt 0.3.0 requires Python >= 3.10; 0.2.7 is the newest release before it.
        assert _on(python, _dependencies())["hyperopt"].specifier.contains("0.2.7")

    def test_python_3_12_gets_setuptools_for_distutils(self):
        assert "setuptools" in _on("3.12", _dependencies())

    def test_requirements_dev_lists_the_same_core_dependencies(self):
        text = (ROOT / "requirements-dev.txt").read_text(encoding="utf-8")
        core = text.split("# Core dependencies", 1)[1].split("\n\n", 1)[0]
        listed = [
            Requirement(line)
            for line in core.splitlines()[1:]
            if line.strip() and not line.startswith("#")
        ]

        assert [str(r) for r in listed] == [str(r) for r in _dependencies()]

    def test_requirements_dev_lists_the_dev_extra(self):
        # Everything after the core section: testing and code-quality tools, uncommented.
        text = (ROOT / "requirements-dev.txt").read_text(encoding="utf-8")
        rest = text.split("# Core dependencies", 1)[1].split("\n\n", 1)[1]
        listed = [
            Requirement(line)
            for line in rest.splitlines()
            if line.strip() and not line.startswith("#")
        ]

        assert [str(r) for r in listed] == [str(r) for r in _extra("dev")]


class TestVizExtra:
    """plotly is optional for users and installed for the tests."""

    def test_plotly_is_not_a_hard_dependency(self):
        assert "plotly" not in {requirement.name for requirement in _dependencies()}

    def test_the_dev_extra_installs_plotly_at_the_viz_floor(self):
        (viz,) = _extra("viz")
        dev = {requirement.name: requirement for requirement in _extra("dev")}

        assert viz.name == "plotly"
        assert str(dev["plotly"].specifier) == str(viz.specifier) == ">=5.0.0"
        # No marker: plotly 5.0.0 supports Python 3.6+, so every supported Python gets it.
        assert dev["plotly"].marker is None
