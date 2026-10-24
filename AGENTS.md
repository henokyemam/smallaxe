# smallaxe - AI Agent Instructions

## Overview

smallaxe is a PySpark MLOps library that provides a scikit-learn-like interface for model training, evaluation, and optimization on PySpark DataFrames. It supports XGBoost, LightGBM, CatBoost, and Random Forest algorithms with built-in preprocessing pipelines, hyperparameter optimization (hyperopt), and cross-validation.

## Repository Structure

```
smallaxe/
├── smallaxe/                  # Main package
│   ├── __init__.py            # Global config (verbosity, spark session, seed, cache)
│   ├── _config.py             # Internal config state
│   ├── auto/                  # AutomatedTraining - train all algorithms and compare
│   ├── datasets/              # Sample data loading utilities
│   │   └── _data.py
│   ├── exceptions/            # Custom exception hierarchy
│   ├── metrics/               # Evaluation metrics
│   │   ├── classification.py  # accuracy, precision, recall, f1, auc_roc, auc_pr, log_loss
│   │   └── regression.py      # mse, rmse, mae, r2, mape
│   ├── pipeline/              # Chainable Pipeline (Imputer → Scaler → Encoder → Model)
│   │   └── pipeline.py
│   ├── preprocessing/         # Data transformers
│   │   ├── encoder.py         # Categorical encoding (onehot, etc.)
│   │   ├── imputer.py         # Missing value imputation
│   │   └── scaler.py          # Feature scaling
│   ├── search/                # Hyperparameter optimization via hyperopt
│   ├── training/              # Model training classes
│   │   ├── algorithm.py       # Algorithm/Param records, the translation rule, registry, 0.8.x name map
│   │   ├── base.py            # BaseModel, Model, Regressor, Classifier
│   │   ├── catboost.py        # CatBoost ALGORITHM record (+ deprecated class aliases)
│   │   ├── classifiers.py     # Classifiers factory
│   │   ├── lightgbm.py        # LightGBM ALGORITHM record (+ deprecated class aliases)
│   │   ├── mixins/            # Composable model behaviors
│   │   │   ├── metadata_mixin.py
│   │   │   ├── param_mixin.py
│   │   │   ├── persistence_mixin.py
│   │   │   ├── spark_model_mixin.py
│   │   │   └── validation_mixin.py
│   │   ├── random_forest.py   # Random Forest ALGORITHM record (+ deprecated class aliases)
│   │   ├── regressors.py      # Regressors factory
│   │   └── xgboost.py         # XGBoost ALGORITHM record (+ deprecated class aliases)
│   └── viz/                   # Plotly-based visualization
├── tests/                     # pytest test suite (session-scoped SparkSession)
├── .github/workflows/
│   ├── ci.yml                 # CI pipeline
│   └── publish.yml            # Package publishing
├── pyproject.toml             # Build config, dependencies, tool settings
├── requirements-dev.txt       # Dev dependencies
├── Goals.md                   # Project roadmap
└── TODO.md                    # Current task list
```

## Architecture

### Model Hierarchy

All models inherit from `BaseModel` which composes five mixins:
- `ParamMixin` - hyperparameter management
- `PersistenceMixin` - model save/load
- `ValidationMixin` - train/test split, k-fold CV (stratified for classification)
- `MetadataMixin` - training metadata capture
- `SparkModelMixin` - PySpark MLlib model wrapping

`BaseRegressor` and `BaseClassifier` extend `BaseModel` with task-type validation.

### The Algorithm Seam

`Model` (and its two kinds, `Regressor` and `Classifier`, in `base.py`) is bound to one
`Algorithm` record from `smallaxe/training/algorithm.py`. The record is declarative: the
strict `Param` table, Spark estimator and fitted-model class paths per task (resolved
lazily, so optional packages are never imported at load), column kwargs, task-fixed
params (LightGBM `objective`, CatBoost `lossFunction`), a feature-importances reader,
the optional `Dependency`, and at most two code hooks (`prepare`, `fit_context`).
`estimator_kwargs()` is the single translation rule for all algorithms. Each algorithm
file exports one `ALGORITHM` record; `algorithm.get(name)` is the registry. Vocabulary
is in `GLOSSARY.md`.

The eight 0.8.x classes (`XGBoostRegressor`, ...) remain as deprecated aliases until 1.0.
Artifacts they saved load through `LEGACY_CLASS_NAMES`; new artifacts record `algorithm`.

### Factory Pattern

`Regressors` and `Classifiers` classes provide static factory methods (`.xgboost()`, `.random_forest()`, `.lightgbm()`, `.catboost()`) that return a `Regressor` / `Classifier` bound to that algorithm.

### Pipeline

The `Pipeline` class chains preprocessing steps and a model into a single `fit()`/`predict()` flow, accepting `numerical_cols` and `categorical_cols` parameters.

## Development

### Prerequisites

- Python 3.8 - 3.12
- PySpark 3.3+
- Java 8 or 11 (for Spark)

### Setup

```bash
pip install -e ".[dev,all]"
```

### Running Tests

```bash
pytest
```

Tests use a session-scoped `SparkSession` fixture (`local[2]` mode) defined in `tests/conftest.py`. `tests/test_algorithm.py` checks every algorithm's translation against fake estimators (no JVM packages needed); `tests/test_model_contract.py` runs the fit/predict/validate/save/load contract per (algorithm, task) wherever the package is installed; `tests/test_legacy_artifacts.py` loads models saved by 0.8.1. Per-algorithm files hold only behaviour unique to that algorithm.

### Code Style

- Formatter: `black` (line-length 100)
- Linter: `ruff` (line-length 100, target py38)
- Type checker: `mypy` (ignore_missing_imports=true)

### Key Conventions

- All model classes follow the `fit(df, label_col, ...)` / `predict(df)` API pattern
- Feature columns are auto-inferred from numeric columns if not specified
- Validation strategies: `'none'`, `'train_test'`, `'kfold'`
- Cache strategies: `'none'`, `'memory'`, `'disk'`
- Task types: `'simple_regression'`, `'binary'`, `'multiclass'`
- Optional algorithm deps are extras: `smallaxe[xgboost]`, `smallaxe[lightgbm]`, `smallaxe[catboost]`, `smallaxe[all]`

## Contributing Guidelines

- Add tests for new features in `tests/test_<module>.py`
- Maintain the mixin-based architecture for model behaviors
- Keep the scikit-learn-like API consistent across all model types
- Use `ValidationError` / `ColumnNotFoundError` / `ModelNotFittedError` from `smallaxe.exceptions`
