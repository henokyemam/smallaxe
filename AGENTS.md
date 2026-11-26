# smallaxe - AI Agent Instructions

## Overview

smallaxe is a PySpark MLOps library that provides a scikit-learn-like interface for model training, evaluation, and optimization on PySpark DataFrames. It supports XGBoost, LightGBM, CatBoost, and Random Forest algorithms with built-in preprocessing pipelines, hyperparameter optimization (hyperopt), and cross-validation.

## Repository Structure

```
smallaxe/
├── smallaxe/                  # Main package
│   ├── __init__.py            # Global settings (verbosity, seed) and load / load_model / load_pipeline
│   ├── _artifact.py           # Versioned metadata.json header of every saved artifact
│   ├── _checks.py             # Shared argument checks (a DataFrame, a list of column names)
│   ├── _config.py             # Internal config state
│   ├── _fs.py                 # Hadoop FileSystem IO (local, dbfs:/, s3://, hdfs://); the one JVM access point (refuses Spark Connect)
│   ├── _warn.py               # caller_stacklevel(): warnings point at the user's line
│   ├── auto/                  # AutomatedTraining: every algorithm on one split, best kept
│   │   └── _automated.py
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
│   │   ├── algorithm.py       # Algorithm/Param records, task names, translation rule, registry
│   │   ├── base.py            # Model, Regressor, Classifier (on BaseModel and the mixins)
│   │   ├── catboost.py        # CatBoost ALGORITHM record
│   │   ├── classifiers.py     # classifiers.<algorithm>() / load() / algorithms() / available_models()
│   │   ├── lightgbm.py        # LightGBM ALGORITHM record
│   │   ├── mixins/            # Composable model behaviors
│   │   │   ├── metadata_mixin.py
│   │   │   ├── param_mixin.py
│   │   │   ├── persistence_mixin.py
│   │   │   ├── spark_model_mixin.py
│   │   │   └── validation_mixin.py
│   │   ├── random_forest.py   # Random Forest ALGORITHM record
│   │   ├── regressors.py      # regressors.<algorithm>() / load() / algorithms() / available_models()
│   │   └── xgboost.py         # XGBoost ALGORITHM record
│   └── viz/                   # Plotly charts that aggregate in Spark (extra `viz`)
│       ├── _classification.py # confusion_matrix, roc_curve, precision_recall_curve
│       ├── _regression.py     # residuals, actual_vs_predicted
│       ├── _models.py         # feature_importances, model_comparison
│       ├── _frames.py         # The rows a chart counts; the one collect of an aggregate
│       └── _style.py          # Lazy plotly import, shared layout, argument checks
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
are implementation classes in `smallaxe.training.base`, not exported by
`smallaxe.training`; the public types are `Model`, `Regressor`, `Classifier`. `BaseModel`
is abstract: `SparkModelMixin._fit_spark_model` and `_get_feature_importance` are
implemented by `Model` from its algorithm record.

### The Algorithm Seam

`Model` (and its two kinds, `Regressor` and `Classifier`, in `base.py`) is bound to one
`Algorithm` record from `smallaxe/training/algorithm.py`. The record is declarative: the
strict `Param` table, Spark estimator and fitted-model class paths per task (resolved
lazily, so optional packages are never imported at load), column kwargs, task-fixed
params (LightGBM `objective`, CatBoost `lossFunction`), a feature-importances reader,
the optional `Dependency`, and at most two code hooks (`prepare`, `fit_context`).
`_estimator_kwargs()` is the single translation rule for all algorithms. Each algorithm
file exports one `ALGORITHM` record; `algorithm.get(name)` is the registry. The task
names (`REGRESSION_TASKS`, `CLASSIFICATION_TASKS`, `ALL_TASKS`) are defined once, in
`algorithm.py`. Vocabulary is in `GLOSSARY.md`.

New artifacts record `algorithm`. Artifacts saved by 0.8.x record their class name, which
`Model.load` maps to the algorithm through the private `_LEGACY_CLASS_NAMES`, the only
place those names appear.

### Factory Pattern

The modules `smallaxe.training.regressors` and `smallaxe.training.classifiers` define one function per algorithm (`random_forest()`, `xgboost()`, `lightgbm()`, `catboost()`; the classifier ones take `task=`) that returns a `Regressor` / `Classifier` bound to that algorithm, plus `load(path)`, `algorithms()` and `available_models()`. They are the only factories. Each factory's docstring lists its algorithm's whole `Param` table; `tests/test_factories.py` keeps the two equal.

### Public API

The public API is frozen at 1.0 and follows semantic versioning (README, "Stability and versioning"). Every module defines `__all__`; every module-level name outside it starts with `_`. `tests/test_public_api.py` pins each public module's `__all__` and the parameters of every public function, method and class: changing the snapshot is a deliberate API change and needs a CHANGELOG entry. A removal or incompatible change first deprecates the old spelling for at least one minor release (a `DeprecationWarning` naming the replacement, attributed to the caller with `smallaxe._warn.caller_stacklevel()`) and lands only in the next major release. The implementation classes (`BaseModel`, `BaseRegressor`, `BaseClassifier`, the mixins) are in their modules' `__all__` but outside the public API; their members are public through `Model`. Public methods raise smallaxe exceptions for user errors; check arguments with `smallaxe._checks` before any Spark job runs, and wrap a Spark call that checks user arguments itself in `_checks.spark_argument_errors(...)`.

### Pipeline

The `Pipeline` class chains preprocessing steps and a model into a single `fit()`/`predict()` flow, accepting `numerical_cols` and `categorical_cols` parameters. The model is fitted on exactly `numerical_cols` plus the Encoder's `output_cols` (the Pipeline owns its feature columns); whether a model needs an Encoder comes from its algorithm record's `accepts_raw_categoricals`. `Pipeline.fit` passes the model's validation options (`validation`, `test_size`, `n_folds`, `fold_key`, `refit`, `cache_strategy`, keyword-only) to the model step, and `Pipeline.validation_scores` reports them; preprocessing is fitted on every row first, so its statistics see the validation rows. `Pipeline.save` writes each step's own artifact under `steps/` and a versioned `metadata.json` (kind `pipeline`) through `smallaxe._fs`; no pickle. `smallaxe.load(path)` loads any artifact by its kind.

### AutomatedTraining

`smallaxe.auto.AutomatedTraining` (in `smallaxe/auto/_automated.py`) is built only on the
public API: the factories, `Model.fit` / `validation_scores` / `predict` / `evaluate`, and
`optimize.run` for algorithms given a `param_spaces` entry. It adds no training logic and
no artifact kind. It reuses, rather than copies: `optimize.py`'s `_resolve_metric`,
`_higher_is_better`, `_lookup_metric` and `_score_key` (metric check, orientation, score
extraction), `_resolve_split_seed` with the models' private `_split_seed_fallback` (one
split seed per run, which `optimize.run` also honours), `Model._validate_fit_params` /
`_validate_columns` / `_apply_cache_strategy` (the checks and caching rule of `fit`, once
per run), and the record's `available()` probe. It supports `simple_regression` and
`binary`; failures are caught as `SmallaxeError` or `_fs.jvm_error_types()`, never a bare
`except`.

### Visualization

`smallaxe.viz` exports seven functions, each returning a `plotly.graph_objects.Figure`
and never calling `show()`. A DataFrame chart aggregates in Spark and collects only the
aggregate: one `groupBy(label, prediction).count()` for `confusion_matrix` (at most
`50 * 50 + 1` rows collected; more than 50 classes is a `ValidationError`), one
`groupBy` over score bins for `roc_curve` / `precision_recall_curve` (at most `n_bins + 1`
rows), a global aggregate and then one `groupBy` over value bins for `residuals` (at most
`n_bins` rows) and `actual_vs_predicted` (at most `n_bins * n_bins`). Every collect goes
through `smallaxe/viz/_frames.py`; never call `collect()` / `toPandas()` on row-level
data, and name the user's columns with `_frames.column(name)` (quoted, so `a.y` is one
column, not struct access). The curves' area is the exact `smallaxe.metrics.auc_roc` /
`auc_pr`, computed on the frame `_scored` projects to `label` / `score` (a numeric score
cast to double, which Spark's evaluator requires). The regression charts bin only finite
values (`_frames.finite`) and note the rows left out.
`feature_importances` reads `Model.feature_importances` (a Pipeline's model step), and
`model_comparison` reads `AutomatedTraining.comparison` or a mapping of scores, ordered
with `optimize.py`'s `_higher_is_better` (the one orientation rule). plotly is imported
only inside `_style.plotly()`; annotations use a `TYPE_CHECKING` import. The charts use
only `plotly.graph_objects` features present in plotly 5.0 (the extra's floor): cell
labels are layout annotations, not the heatmap `texttemplate` of 5.5. `tests/test_viz.py`
asserts on figure data and on the row counts of the private aggregation helpers.

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

Tests use a session-scoped `SparkSession` fixture (`local[2]` mode) defined in `tests/conftest.py`. `tests/test_algorithm.py` checks every algorithm's translation against fake estimators (no JVM packages needed); `tests/test_model_contract.py` runs the fit/predict/validate/save/load contract per (algorithm, task) wherever the package is installed; `tests/test_legacy_artifacts.py` loads models saved by 0.8.1, 0.9.0 and 0.10.0; `tests/test_public_api.py` pins the public surface; `tests/test_removed.py` checks that every 0.x spelling is gone; `tests/test_user_errors.py` probes bad inputs for raw Python errors; `tests/test_auto.py` covers AutomatedTraining (real Random Forest / XGBoost runs, and fake fits for ranking, failures and skips). Per-algorithm files hold only behaviour unique to that algorithm.

### Code Style

- Formatter: `black` (line-length 100)
- Linter: `ruff` (line-length 100, target py38)
- Type checker: `mypy smallaxe`, strict settings in `pyproject.toml` (untyped defs disallowed, `warn_return_any`, `warn_unused_ignores`; missing imports ignored only for hyperopt, py4j, pandas and catboost_spark). The package ships `py.typed`, and CI blocks on it. A `# type: ignore` needs an error code and a reason.

### Key Conventions

- All model classes follow the `fit(dataframe, label_col, ...)` / `predict(dataframe)` API pattern; every public parameter holding a DataFrame is named `dataframe`
- Set parameters with `set_params(**params)`
- Feature columns are auto-inferred from numeric and boolean columns if not specified; other columns are reported in one `UserWarning` and in `metadata["ignored_cols"]`
- Validation strategies: `'none'`, `'train_test'`, `'kfold'`
- Cache strategies (per `fit` call): `None` (default: cache only when fit reads the data more than once), `'none'`, `'memory'`, `'disk'`.
- Task types: `'simple_regression'`, `'binary'`, `'multiclass'` (`AutomatedTraining`: the first two)
- Optional deps are extras: `smallaxe[xgboost]`, `smallaxe[lightgbm]`, `smallaxe[catboost]`, `smallaxe[viz]` (plotly, for `smallaxe.viz`; imported lazily inside `smallaxe/viz/_style.py`, so `import smallaxe.viz` works without it and a chart raises `DependencyError`; the dev extra installs it so CI draws every chart), `smallaxe[all]`. Hard dependencies are exactly what smallaxe imports at runtime (pyspark, py4j, numpy, pandas, hyperopt; setuptools where PySpark needs `distutils` (3.12) or hyperopt 0.2.7 needs `pkg_resources` (3.8, 3.9)), each with a `>=` floor that the test-lowest CI job installs (`.github/scripts/floor_pins.py`); `tests/test_packaging.py` checks the set per Python and that `requirements-dev.txt` matches
- In the test suite a `DeprecationWarning` attributed to a smallaxe module is an error (`filterwarnings` in `pyproject.toml`)

## Contributing Guidelines

- Add tests for new features in `tests/test_<module>.py`
- Maintain the mixin-based architecture for model behaviors
- Keep the scikit-learn-like API consistent across all model types
- Use `ValidationError` / `ColumnNotFoundError` / `ModelNotFittedError` from `smallaxe.exceptions`; a public method never raises a raw `ValueError` / `KeyError` / `TypeError` for a user error
- A new public name goes in its module's `__all__` and in the `tests/test_public_api.py` snapshot, with a CHANGELOG entry
