# Contributing to smallaxe

## Getting Started

1. Fork and clone the repository
2. Install in development mode:
   ```bash
   pip install -e ".[dev,all]"
   ```
3. Run the test suite to verify your setup:
   ```bash
   pytest
   ```

## Development Workflow

1. Create a feature branch from `main`
2. Make your changes
3. Add or update tests in `tests/`
4. Ensure all tests pass: `pytest`. This includes `tests/test_readme_examples.py`, which
   runs every Python example in `README.md`
5. Format code: `black smallaxe tests` (CI runs `black --check smallaxe tests`)
6. Lint: `ruff check smallaxe tests examples` (CI lints `smallaxe` and `tests`)
7. Type check: `mypy smallaxe` (strict; CI fails on any error)
8. Add a `CHANGELOG.md` entry for a user-visible change
9. Open a pull request

## Code Style

- Line length: 100 characters
- Formatter: black
- Linter: ruff
- Full type annotations on every function (mypy runs with `disallow_untyped_defs`); a
  `# type: ignore` carries an error code and the reason
- In the test suite a `DeprecationWarning` attributed to a smallaxe module is an error:
  one smallaxe raises about itself, or a deprecated numpy or standard-library call on a
  line of smallaxe. PySpark's deprecations are `FutureWarning`s attributed to PySpark,
  so the filter does not catch them
- Dependencies: every entry in `[project] dependencies` has a `>=` floor (the
  test-lowest CI job installs each floor, through `.github/scripts/floor_pins.py`);
  keep `requirements-dev.txt` the same (`tests/test_packaging.py` checks both)
- Google-style docstrings (Args / Returns / Raises) on every public class, method and function
- User errors raise smallaxe exceptions (`smallaxe.exceptions`), never a raw `ValueError` / `KeyError` / `TypeError`

## The Public API

smallaxe follows semantic versioning from 1.0 (see "Stability and versioning" in the
README). Every module defines `__all__`, and every module-level name outside it starts
with `_`. `tests/test_public_api.py` pins the public surface: if your change makes it
fail, the change alters the public API. Update the snapshot only on purpose, together
with a CHANGELOG entry. To remove or rename something public, deprecate it first: keep
the old spelling working with a `DeprecationWarning` naming the replacement for at least
one minor release, and remove it in the next major release.

## Adding New Features

- New algorithms are one `ALGORITHM` record in `smallaxe/training/<algorithm>.py`, registered in `smallaxe/training/algorithm.py`, with factory functions in `regressors.py`/`classifiers.py` (see `HOWTOAI.md` and `GLOSSARY.md`)
- New preprocessing steps go in `smallaxe/preprocessing/`
- New metrics go in `smallaxe/metrics/`
- All new code needs corresponding tests in `tests/`

## Reporting Issues

Open an issue at https://github.com/henokyemam/smallaxe/issues with:
- Steps to reproduce
- Expected vs actual behavior
- Python and PySpark version
