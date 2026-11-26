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
4. Ensure all tests pass: `pytest`
5. Format code: `black .`
6. Lint: `ruff check .`
7. Open a pull request

## Code Style

- Line length: 100 characters
- Formatter: black
- Linter: ruff
- Type annotations required for public APIs
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
