# How to Work with smallaxe Using AI

## Quick Context

smallaxe is a PySpark MLOps library. All model training flows through `Model.fit()` (defined on `BaseModel` in `smallaxe/training/base.py`) with a consistent API. The codebase is a single Python package with no microservices or infrastructure concerns.

## Common Tasks

### Adding a New Algorithm

1. Create `smallaxe/training/<algorithm>.py` exporting one `ALGORITHM = Algorithm(...)` record (see `GLOSSARY.md`): the `Param` table, estimator and fitted-model class paths per task, column kwargs, task-fixed params, an importances reader, and its `Dependency`. Add `prepare` / `fit_context` hooks only for behaviour data cannot express.
2. Register the module in `_MODULES` in `smallaxe/training/algorithm.py`
3. Add a one-line constructor function to `smallaxe/training/regressors.py` and `smallaxe/training/classifiers.py` (`return _create("<algorithm>", params)`; the classifier one takes `task`) and list it in each module's `__all__`
4. Add optional dependency in `pyproject.toml` under `[project.optional-dependencies]`
5. `tests/test_algorithm.py` and `tests/test_model_contract.py` pick the record up from the registry; add `tests/test_<algorithm>.py` only for behaviour unique to it

### Adding a Preprocessing Step

1. Create `smallaxe/preprocessing/<step>.py` with a class that has `fit()` and `transform()` methods
2. Export it from `smallaxe/preprocessing/__init__.py`
3. The `Pipeline` class in `smallaxe/pipeline/pipeline.py` will accept it as a step

### Adding a Metric

1. Add the function to `smallaxe/metrics/regression.py` or `smallaxe/metrics/classification.py`
2. Metrics take the DataFrame (parameter `dataframe`), label column, and prediction column as arguments; options after those are keyword-only. Decorate a public function with `@_deprecation.df_keyword` only if it once called the parameter `df`
3. Update `BaseModel._compute_regression_metrics()` or `_compute_classification_metrics()` if the metric should be included in validation scores

## Running and Testing

```bash
# Install in dev mode with all algorithms
pip install -e ".[dev,all]"

# Run all tests
pytest

# Run tests for a specific module
pytest tests/test_xgboost.py -v

# Format and lint
black . && ruff check .

# Type check
mypy smallaxe/
```

## Architecture Decisions to Respect

- **Mixin composition over deep inheritance** — model behaviors are split across 5 mixins in `smallaxe/training/mixins/`
- **Factory functions for model creation** — users call `regressors.xgboost()` (from `smallaxe.training import regressors`), not `XGBoostRegressor()` or the deprecated `Regressors.xgboost()`
- **Deprecations go through `smallaxe/_deprecation.py`** — the old spelling keeps working with one `DeprecationWarning` that names the replacement and says "removed in smallaxe 1.0"; smallaxe's own code uses only the new spelling
- **PySpark MLlib under the hood** — models wrap Spark's native ML pipeline internally
- **Optional algorithm dependencies** — XGBoost, LightGBM, CatBoost are extras, not core deps
- **Session-scoped SparkSession in tests** — expensive to create, shared across all tests in a run

## Pitfalls

- Tests require Java 8 or 11 installed for Spark to run
- The `spark_session` fixture uses `local[2]` — don't assume distributed behavior in tests
- Feature columns are auto-inferred from numeric columns if not explicitly passed — be aware of this when debugging unexpected behavior
- CatBoost and LightGBM have additional native library dependencies beyond the Python packages
