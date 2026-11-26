# smallaxe - AI Agent Instructions

## Overview

smallaxe is a PySpark MLOps library that provides a scikit-learn-like interface for model training, evaluation, and optimization on PySpark DataFrames. It supports XGBoost, LightGBM, CatBoost, and Random Forest algorithms with built-in preprocessing pipelines, hyperparameter optimization (hyperopt), and cross-validation.

## Repository Structure

```
smallaxe/
├── smallaxe/                  # Main package
│   ├── __init__.py            # Global config (verbosity, spark session, seed, cache)
│   ├── _config.py             # Internal config state
│   ├── _deprecation.py        # DeprecationWarning helpers (df= keyword, private-name shims)
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
│   │   ├── classifiers.py     # classifiers.<algorithm>() / load() (+ deprecated Classifiers)
│   │   ├── lightgbm.py        # LightGBM ALGORITHM record (+ deprecated class aliases)
│   │   ├── mixins/            # Composable model behaviors
│   │   │   ├── metadata_mixin.py
│   │   │   ├── param_mixin.py
│   │   │   ├── persistence_mixin.py
│   │   │   ├── spark_model_mixin.py
│   │   │   └── validation_mixin.py
│   │   ├── random_forest.py   # Random Forest ALGORITHM record (+ deprecated class aliases)
│   │   ├── regressors.py      # regressors.<algorithm>() / load() (+ deprecated Regressors)
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
- `ValidationMixin` - train/test split and k-fold CV by a content hash (each class splits in the same proportion)
- `MetadataMixin` - training metadata capture
- `SparkModelMixin` - PySpark MLlib model wrapping

`BaseRegressor` and `BaseClassifier` extend `BaseModel` with task-type validation. All three
stay in `smallaxe.training.base`; the public types are `Model`, `Regressor`, `Classifier`.

### The Algorithm Seam

`Model` (and its two kinds, `Regressor` and `Classifier`, in `base.py`) is bound to one
`Algorithm` record from `smallaxe/training/algorithm.py`. The record is declarative: the
strict `Param` table, Spark estimator and fitted-model class paths per task (resolved
lazily, so optional packages are never imported at load), column kwargs, task-fixed
params (LightGBM `objective`, CatBoost `lossFunction`), a feature-importances reader,
the optional `Dependency`, and at most two code hooks (`prepare`, `fit_context`).
`_estimator_kwargs()` is the single translation rule for all algorithms. Each algorithm
file exports one `ALGORITHM` record; `algorithm.get(name)` is the registry. Vocabulary
is in `GLOSSARY.md`.

The eight 0.8.x classes (`XGBoostRegressor`, ...) remain as deprecated aliases until 1.0.
Artifacts they saved load through `_LEGACY_CLASS_NAMES`; new artifacts record `algorithm`.

### Factory Pattern

The modules `smallaxe.training.regressors` and `smallaxe.training.classifiers` define one function per algorithm (`random_forest()`, `xgboost()`, `lightgbm()`, `catboost()`; the classifier ones take `task=`) that returns a `Regressor` / `Classifier` bound to that algorithm, plus `load(path)`, `algorithms()` and `available_models()`. The `Regressors` / `Classifiers` classes are their deprecated 0.x spelling and are removed in 1.0.

### Deprecations

Every deprecation goes through `smallaxe/_deprecation.py`: one `DeprecationWarning` naming the replacement and saying "removed in smallaxe 1.0", attributed to the first caller outside smallaxe. `df_keyword` keeps `df=` working where the parameter is now `dataframe`; `keyword_only` keeps positional values for keyword-only parameters; `module_getattr` keeps names that became private importable. smallaxe's own code never uses a deprecated spelling (`tests/test_deprecations.py` checks it).

### Pipeline

The `Pipeline` class chains preprocessing steps and a model into a single `fit()`/`predict()` flow, accepting `numerical_cols` and `categorical_cols` parameters. The model is fitted on exactly `numerical_cols` plus the Encoder's `output_cols` (the Pipeline owns its feature columns); whether a model needs an Encoder comes from its algorithm record's `accepts_raw_categoricals`. `Pipeline.save` writes each step's own artifact under `steps/` and a versioned `metadata.json` (kind `pipeline`) through `smallaxe._fs`; no pickle. `smallaxe.load(path)` loads any artifact by its kind.

## Development

### Prerequisites

- Python 3.8 - 3.12 (3.11 needs PySpark 3.4 or 3.5; 3.12 needs PySpark 3.5; PySpark 3.3 supports 3.8 - 3.10)
- PySpark 3.3 - 3.5
- Java 11 (CI also tests Java 8 with PySpark 3.3 and Java 17 with PySpark 3.5)
- Linux (CI runs on Ubuntu only)

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

- All model classes follow the `fit(dataframe, label_col, ...)` / `predict(dataframe)` API pattern; every public parameter holding a DataFrame is named `dataframe`
- Set parameters with `set_params(**params)`
- Feature columns are auto-inferred from numeric and boolean columns if not specified; other columns are reported in one `UserWarning` and in `metadata["ignored_cols"]`
- Validation strategies: `'none'`, `'train_test'`, `'kfold'`
- Cache strategies (per `fit` call): `None` (default: cache only when fit reads the data more than once), `'none'`, `'memory'`, `'disk'`. The global `set_cache_strategy` is deprecated and has no effect.
- Task types: `'simple_regression'`, `'binary'`, `'multiclass'`
- Optional algorithm deps are extras: `smallaxe[xgboost]`, `smallaxe[lightgbm]`, `smallaxe[catboost]`, `smallaxe[all]`

## Contributing Guidelines

- Add tests for new features in `tests/test_<module>.py`
- Maintain the mixin-based architecture for model behaviors
- Keep the scikit-learn-like API consistent across all model types
- Use `ValidationError` / `ColumnNotFoundError` / `ModelNotFittedError` from `smallaxe.exceptions`
