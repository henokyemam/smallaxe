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

## Adding New Features

- New algorithms go in `smallaxe/training/` with factory methods in `regressors.py`/`classifiers.py`
- New preprocessing steps go in `smallaxe/preprocessing/`
- New metrics go in `smallaxe/metrics/`
- All new code needs corresponding tests in `tests/`

## Reporting Issues

Open an issue at https://github.com/henokyemam/smallaxe/issues with:
- Steps to reproduce
- Expected vs actual behavior
- Python and PySpark version
