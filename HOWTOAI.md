# How to Work with smallaxe Using AI

## Quick Context

smallaxe is a PySpark MLOps library. All model training flows through `Model.fit()` (implemented once in `smallaxe/training/base.py`) with a consistent API. The public API is frozen at 1.0: every module's `__all__` is pinned by `tests/test_public_api.py`. The codebase is a single Python package with no microservices or infrastructure concerns.

## Common Tasks

### Adding a New Algorithm

1. Create `smallaxe/training/<algorithm>.py` exporting one `ALGORITHM = Algorithm(...)` record (see `GLOSSARY.md`): the `Param` table, estimator and fitted-model class paths per task, column kwargs, task-fixed params, an importances reader, and its `Dependency`. Add `prepare` / `fit_context` hooks only for behaviour data cannot express.
2. Register the module in `_MODULES` in `smallaxe/training/algorithm.py`
3. Add a one-line constructor function to `smallaxe/training/regressors.py` and `smallaxe/training/classifiers.py` (`return _create("<algorithm>", params)`; the classifier one takes `task`), list it in each module's `__all__`, and list every row of its `Param` table in its docstring (`tests/test_factories.py` checks the docstring against the table)
4. Add optional dependency in `pyproject.toml` under `[project.optional-dependencies]`
5. `tests/test_algorithm.py` and `tests/test_model_contract.py` pick the record up from the registry; add `tests/test_<algorithm>.py` only for behaviour unique to it

### Adding a Preprocessing Step

1. Create `smallaxe/preprocessing/<step>.py` with a class that has `fit(dataframe, ...)` and `transform(dataframe)` methods, and `save` / `load` through `smallaxe/preprocessing/_common.py` and `smallaxe._artifact` with its own artifact kind
2. List it in the module's `__all__`, export it from `smallaxe/preprocessing/__init__.py`, and pin it in `tests/test_public_api.py` with a CHANGELOG entry
3. A `Pipeline` accepts any object with `fit` and `transform` as a duck-typed step, but saves only the four kinds `_step_kind` in `smallaxe/pipeline/pipeline.py` recognises. For the new step to save inside a pipeline, add its kind there and in `smallaxe.load` (`smallaxe/__init__.py`)

### Changing AutomatedTraining

`smallaxe/auto/_automated.py` must stay built on the public training API and on the private helpers it shares with `optimize.run` (metric check and orientation, score lookup, split-seed fallback); change those helpers in `smallaxe/search/optimize.py`, not a copy. A new algorithm needs nothing here: it is picked up from `regressors.algorithms()` / `classifiers.algorithms()`. Training failures are caught as `SmallaxeError` or `smallaxe._fs.jvm_error_types()` only. `tests/test_auto.py` uses a fake `Model.fit` (`fake_fits`) for ranking, failure and skip rules, and real Random Forest / XGBoost runs for the rest.

### Adding or Changing a Chart

Charts live in `smallaxe/viz/` and are exported from `smallaxe/viz/__init__.py`. A DataFrame chart checks its arguments first (`smallaxe/metrics/_checks.py`, `smallaxe/viz/_style.py`), then builds one aggregated Spark DataFrame in a private helper (`_confusion_table`, `_score_bins`, `_residual_table`, `_density_table`) and collects it through `smallaxe/viz/_frames.py`; nothing calls `collect()` or `toPandas()` on row-level data. Refer to the user's columns with `_frames.column(name)`, never `F.col(name)`, so a name holding a dot works. Get plotly only from `_style.plotly()` (never import it at module level: `import smallaxe.viz` must not load it), build the figure with `_style.figure(...)`, and use only `plotly.graph_objects` features that plotly 5.0 has. Test the figure's data and the helper's row count in `tests/test_viz.py`, and pin a new function in `tests/test_public_api.py`.

### Changing a README Example

`tests/test_readme_examples.py` runs every fenced `python` block of `README.md`. A block must be self-contained (its own imports and data, from `smallaxe.datasets`) and quick. A block that cannot run in the test suite gets a marker on the line before its fence: `<!-- test: skip (<reason>) -->`, or `<!-- test: needs <name> -->` for one that needs an algorithm (`xgboost`) or a module (`plotly`). The Quick Start must stay one runnable block under 20 lines.

### Adding a Metric

1. Add the function to `smallaxe/metrics/regression.py` or `smallaxe/metrics/classification.py`
2. Metrics take the DataFrame (parameter `dataframe`), label column, and prediction column as arguments; options after those are keyword-only. Start with `validate_columns` from `smallaxe/metrics/_checks.py`, so bad input raises `ValidationError` / `ColumnNotFoundError`
3. Update `BaseModel._compute_regression_metrics()` or `_compute_classification_metrics_single_pass()` if the metric should be included in validation scores
4. List it in the module's `__all__` and in `smallaxe/metrics/__init__.py`, and add it to the `tests/test_public_api.py` snapshot with a CHANGELOG entry

## Running and Testing

```bash
# Install in dev mode with all algorithms
pip install -e ".[dev,all]"

# Run all tests
pytest

# Run tests for a specific module
pytest tests/test_xgboost.py -v

# Format and lint (CI checks smallaxe and tests)
black --check smallaxe tests && ruff check smallaxe tests examples

# Type check (strict settings in pyproject.toml; CI blocks on it)
mypy smallaxe
```

## Architecture Decisions to Respect

- **Mixin composition over deep inheritance**: model behaviors are split across 5 mixins in `smallaxe/training/mixins/`
- **Factory functions for model creation**: users call `regressors.xgboost()` / `classifiers.xgboost(task=...)` (from `smallaxe.training import regressors, classifiers`); they are the only factories
- **A frozen public API**: semantic versioning from 1.0. Changing a public name or signature changes the `tests/test_public_api.py` snapshot and needs a CHANGELOG entry; a removal is deprecated for at least one minor release first (one `DeprecationWarning` naming the replacement, attributed with `smallaxe._warn.caller_stacklevel()`) and happens only in a major release
- **smallaxe exceptions for user errors**: public methods check arguments up front (`smallaxe/_checks.py`) and raise `ValidationError`, `ColumnNotFoundError`, `ModelNotFittedError` or `ConfigurationError`, never a raw `ValueError` / `KeyError` / `TypeError`
- **PySpark MLlib under the hood**: models wrap Spark's native ML pipeline internally
- **Optional algorithm dependencies**: XGBoost, LightGBM, CatBoost (and plotly, `viz`) are extras, not core deps; the hard dependencies are exactly what smallaxe imports at runtime. plotly is imported lazily by `smallaxe.viz`, and the dev extra installs it so the tests draw every chart
- **Charts aggregate in Spark**: `smallaxe.viz` collects only grouped or globally aggregated rows, bounded by bins or classes
- **Typed package**: smallaxe ships `py.typed`; `mypy smallaxe` passes under the strict settings in `pyproject.toml`. Fix a type error by correcting the type; a `# type: ignore` needs an error code and a reason
- **Session-scoped SparkSession in tests**: expensive to create, shared across all tests in a run

## Pitfalls

- Tests require Java installed for Spark to run: 11 for Spark 3 (CI also uses 8 and 17), 17 or later for Spark 4 (CI uses 17 and 25)
- smallaxe needs a classic Spark session; on Spark Connect, save and load raise `ConfigurationError` (the check is in `smallaxe/_fs.py`, the one place that reaches the JVM)
- Spark 4 turns ANSI SQL mode on by default; smallaxe's results must be the same with it on or off and it never sets the option (`tests/test_ansi_mode.py`)
- The `spark_session` fixture uses `local[2]`; don't assume distributed behavior in tests
- Feature columns are inferred from the numeric and boolean columns if not explicitly passed; be aware of this when debugging unexpected behavior (see `GOTCHAS.md`)
- CatBoost and LightGBM need JVM packages and native libraries beyond the Python packages; CI does not run them, and their translation is tested against fake estimators (`tests/fakes.py`, `tests/test_algorithm.py`)
- plotly is optional: never import it at module level outside `smallaxe/viz/_style.py`, or `import smallaxe.viz` breaks without it
