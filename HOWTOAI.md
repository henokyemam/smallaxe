# How to Work with smallaxe Using AI

## Quick Context

smallaxe is a PySpark MLOps library. All model training flows through `BaseModel.fit()` with a consistent API. The codebase is a single Python package with no microservices or infrastructure concerns.

## Common Tasks

### Adding a New Algorithm

1. Create `smallaxe/training/<algorithm>.py` with a class extending `BaseRegressor` or `BaseClassifier`
2. Implement `_fit_spark_model()`, `_predict_spark_model()`, and `_predict_proba_spark_model()` (classifiers only)
3. Add factory methods to `smallaxe/training/regressors.py` and/or `smallaxe/training/classifiers.py`
4. Add optional dependency in `pyproject.toml` under `[project.optional-dependencies]`
5. Create `tests/test_<algorithm>.py` following the pattern of existing algorithm tests

### Adding a Preprocessing Step

1. Create `smallaxe/preprocessing/<step>.py` with a class that has `fit()` and `transform()` methods
2. Export it from `smallaxe/preprocessing/__init__.py`
3. The `Pipeline` class in `smallaxe/pipeline/pipeline.py` will accept it as a step

### Adding a Metric

1. Add the function to `smallaxe/metrics/regression.py` or `smallaxe/metrics/classification.py`
2. Metrics take a DataFrame, label column, and prediction column as arguments
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
- **Factory pattern for model creation** — users call `Regressors.xgboost()` not `XGBoostRegressor()`
- **PySpark MLlib under the hood** — models wrap Spark's native ML pipeline internally
- **Optional algorithm dependencies** — XGBoost, LightGBM, CatBoost are extras, not core deps
- **Session-scoped SparkSession in tests** — expensive to create, shared across all tests in a run

## Pitfalls

- Tests require Java 8 or 11 installed for Spark to run
- The `spark_session` fixture uses `local[2]` — don't assume distributed behavior in tests
- Feature columns are auto-inferred from numeric columns if not explicitly passed — be aware of this when debugging unexpected behavior
- CatBoost and LightGBM have additional native library dependencies beyond the Python packages
