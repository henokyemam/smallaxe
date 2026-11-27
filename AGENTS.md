# smallaxe - AI Agent Instructions

## Overview

smallaxe is a PySpark MLOps library that provides a scikit-learn-like interface for model training, evaluation, and optimization on PySpark DataFrames. It supports Random Forest, XGBoost, LightGBM and CatBoost, with preprocessing steps and pipelines that save and load as versioned artifacts, train/test and k-fold validation, hyperparameter search (hyperopt), automated training across algorithms (`smallaxe.auto`) and plotly charts (`smallaxe.viz`). It runs on PySpark 3.3 to 3.5 and 4.0 to 4.3, on a classic Spark session.

## Repository Structure

```
smallaxe/
├── smallaxe/                  # Main package (ships py.typed)
│   ├── __init__.py            # Global settings (verbosity, seed) and load / load_model / load_pipeline
│   ├── _artifact.py           # Versioned metadata.json header of every saved artifact
│   ├── _checks.py             # Shared argument checks (a DataFrame, a list of column names)
│   ├── _config.py             # Internal config state (verbosity, seed)
│   ├── _fs.py                 # Hadoop FileSystem IO (local, dbfs:/, s3://, hdfs://); the one JVM access point (refuses Spark Connect)
│   ├── _sql.py                # Spark 3's default (non-ANSI) string-to-integer cast, in either ANSI mode; quoted()
│   ├── _warn.py               # caller_stacklevel(): warnings point at the user's line
│   ├── auto/                  # AutomatedTraining: every algorithm on one split, best kept
│   │   ├── __init__.py
│   │   └── _automated.py
│   ├── datasets/              # Sample datasets (load_sample_regression / _classification, dataset_info)
│   │   ├── __init__.py
│   │   └── _data.py
│   ├── exceptions/            # Custom exception hierarchy (__init__.py)
│   ├── metrics/               # Evaluation metrics; TASK_METRICS
│   │   ├── __init__.py
│   │   ├── _checks.py         # Column checks and ANSI-independent comparisons and casts
│   │   ├── classification.py  # accuracy, precision, recall, f1_score, multiclass_scores, auc_roc, auc_pr, log_loss
│   │   └── regression.py      # mse, rmse, mae, r2, mape
│   ├── pipeline/              # Pipeline: preprocessing steps and an optional model, saved as one unit
│   │   ├── __init__.py
│   │   └── pipeline.py
│   ├── preprocessing/         # Data transformers, each with save / load
│   │   ├── __init__.py
│   │   ├── _common.py         # Column naming and artifact IO shared by the steps
│   │   ├── encoder.py         # Categorical encoding (onehot, label)
│   │   ├── imputer.py         # Missing value imputation
│   │   └── scaler.py          # Feature scaling (standard, minmax, maxabs)
│   ├── search/                # Hyperparameter optimization via hyperopt
│   │   ├── __init__.py
│   │   └── optimize.py        # optimize.run, SearchResult; metric rules shared with auto and viz
│   ├── training/              # Model training classes
│   │   ├── __init__.py
│   │   ├── algorithm.py       # Algorithm/Param records, task names, translation rule, registry
│   │   ├── base.py            # Model, Regressor, Classifier (on BaseModel and the mixins)
│   │   ├── catboost.py        # CatBoost ALGORITHM record and the SynapseML compatibility shim
│   │   ├── classifiers.py     # classifiers.<algorithm>() / load() / algorithms() / available_models()
│   │   ├── lightgbm.py        # LightGBM ALGORITHM record
│   │   ├── mixins/            # Composable model behaviors
│   │   │   ├── __init__.py
│   │   │   ├── metadata_mixin.py
│   │   │   ├── param_mixin.py
│   │   │   ├── persistence_mixin.py
│   │   │   ├── spark_model_mixin.py
│   │   │   └── validation_mixin.py
│   │   ├── random_forest.py   # Random Forest ALGORITHM record
│   │   ├── regressors.py      # regressors.<algorithm>() / load() / algorithms() / available_models()
│   │   └── xgboost.py         # XGBoost ALGORITHM record
│   └── viz/                   # Plotly charts that aggregate in Spark (extra `viz`)
│       ├── __init__.py
│       ├── _classification.py # confusion_matrix, roc_curve, precision_recall_curve
│       ├── _regression.py     # residuals, actual_vs_predicted
│       ├── _models.py         # feature_importances, model_comparison
│       ├── _frames.py         # The rows a chart counts; the one collect of an aggregate
│       └── _style.py          # Lazy plotly import, shared layout, argument checks
├── tests/                     # pytest test suite (session-scoped SparkSession in conftest.py)
│   ├── fakes.py               # Fake estimator modules for LightGBM / CatBoost translation tests
│   └── fixtures/              # Artifacts saved by released versions, and the scripts that wrote them
├── examples/                  # Kaggle end-to-end scripts and two Databricks validation notebooks
├── .github/
│   ├── scripts/floor_pins.py  # Turns each dependency floor into an exact pin for the test-lowest job
│   └── workflows/
│       ├── ci.yml             # Lint, type check, test matrix, test-lowest, test-xgboost
│       └── publish.yml        # Package publishing (runs ci.yml first)
├── pyproject.toml             # Build config, dependencies, tool settings
├── requirements-dev.txt       # Dev dependencies (kept equal to the dev extra by tests/test_packaging.py)
├── MANIFEST.in                # What the source distribution contains
├── README.md                  # User documentation; its Python blocks run in tests/test_readme_examples.py
├── CHANGELOG.md               # Release notes, with the 0.x to 1.0 migration list
├── GLOSSARY.md                # Domain vocabulary used in code and docs
├── GOTCHAS.md                 # Behaviour that can surprise users, and what to do
├── HOWTOAI.md                 # Common tasks for AI agents working on the code
├── CONTRIBUTING.md            # Contribution workflow
├── Goals.md                   # Product goals and the status of the v1 acceptance criteria
└── TODO.md                    # What is left after 1.0
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

- Python 3.8 - 3.12 (3.11 needs PySpark 3.4 or later; 3.12 needs PySpark 3.5 or later; PySpark 3.3 supports 3.8 - 3.10; PySpark 4.0 needs 3.9+, 4.1 and 4.2 3.10+, 4.3 3.11+)
- PySpark 3.3 - 3.5 and 4.0 - 4.3 (`pyspark>=3.3,<4.4`; the bound is one past the newest Spark minor with CI legs, which `tests/test_packaging.py` checks against `ci.yml`)
- Java 11 for Spark 3 (CI also tests Java 8 with PySpark 3.3 and Java 17 with PySpark 3.5); Java 17 for Spark 4 (CI also tests Java 25 with PySpark 4.3)
- A classic (JVM, py4j) Spark session. Spark Connect is not supported: `smallaxe._fs`, the one place that reaches the JVM, raises `ConfigurationError` on a Connect session. On Spark 4, LightGBM and CatBoost are validated on Spark 4.0 only (Databricks, not CI); XGBoost needs xgboost 3.3+
- Linux (CI runs on Ubuntu only)

### Setup

```bash
pip install -e ".[dev,all]"
```

### Running Tests

```bash
pytest
ruff check smallaxe tests examples     # CI lints smallaxe and tests
black --check smallaxe tests
mypy smallaxe
```

Tests use a session-scoped `SparkSession` fixture (`local[2]` mode) defined in `tests/conftest.py`. `tests/test_algorithm.py` checks every algorithm's translation against fake estimators (no JVM packages needed); `tests/test_model_contract.py` runs the fit/predict/validate/save/load contract per (algorithm, task) wherever the package is installed; `tests/test_legacy_artifacts.py` loads models saved by 0.8.1, 0.9.0 and 0.10.0; `tests/test_public_api.py` pins the public surface; `tests/test_removed.py` checks that every 0.x spelling is gone; `tests/test_user_errors.py` probes bad inputs for raw Python errors; `tests/test_auto.py` covers AutomatedTraining (real Random Forest / XGBoost runs, and fake fits for ranking, failures and skips); `tests/test_viz.py` asserts on chart data and on the row counts the charts collect; `tests/test_ansi_mode.py` runs each ANSI-sensitive edge case with ANSI mode on and off; `tests/test_classic_session.py` checks that a Spark Connect session is refused; `tests/test_packaging.py` checks the dependency set, the floors and the PySpark bound against `ci.yml`; `tests/test_readme_examples.py` runs every Python block of `README.md` that has no `<!-- test: ... -->` marker. `tests/test_pipeline_persistence.py` and `tests/test_preprocessing_persistence.py` also load the 0.10 fixtures under `tests/fixtures/`. Per-algorithm files hold only behaviour unique to that algorithm.

### Code Style

- Formatter: `black` (line-length 100)
- Linter: `ruff` (line-length 100, target py38)
- Type checker: `mypy smallaxe`, strict settings in `pyproject.toml` (untyped defs disallowed, `warn_return_any`, `warn_unused_ignores`; missing imports ignored only for hyperopt, py4j, pandas, catboost_spark and plotly). The package ships `py.typed`, and CI blocks on it. A `# type: ignore` needs an error code and a reason.

### Key Conventions

- All model classes follow the `fit(dataframe, label_col, ...)` / `predict(dataframe)` API pattern; every public parameter holding a DataFrame is named `dataframe`
- Set parameters with `set_params(**params)`
- Feature columns are auto-inferred from numeric and boolean columns if not specified; other columns are reported in one `UserWarning` and in `metadata["ignored_cols"]`
- Validation strategies: `'none'`, `'train_test'`, `'kfold'`
- Cache strategies (per `fit` call): `None` (default: cache only when fit reads the data more than once), `'none'`, `'memory'`, `'disk'`.
- Task types: `'simple_regression'`, `'binary'`, `'multiclass'` (`AutomatedTraining`: the first two)
- Optional deps are extras: `smallaxe[xgboost]`, `smallaxe[lightgbm]`, `smallaxe[catboost]`, `smallaxe[viz]` (plotly, for `smallaxe.viz`; imported lazily inside `smallaxe/viz/_style.py`, so `import smallaxe.viz` works without it and a chart raises `DependencyError`; the dev extra installs it so CI draws every chart), `smallaxe[all]`. Hard dependencies are exactly what smallaxe imports at runtime (pyspark, py4j, numpy, pandas, hyperopt; setuptools where PySpark needs `distutils` (3.12) or hyperopt 0.2.7 needs `pkg_resources` (3.8, 3.9)), each with a `>=` floor that the test-lowest CI job installs (`.github/scripts/floor_pins.py`); `tests/test_packaging.py` checks the set per Python and that `requirements-dev.txt` matches
- In the test suite a `DeprecationWarning` attributed to a smallaxe module is an error (`filterwarnings` in `pyproject.toml`)
- Results must not depend on Spark's ANSI SQL mode (on by default from Spark 4), and smallaxe never sets `spark.sql.ansi.enabled`: the result is Spark 3's default (ANSI off) result, with the casts made explicit: compare columns of different types through `smallaxe.metrics._checks.comparable` / `is_one`, read a string as an integer with `smallaxe._sql.integral_from_string` (ANSI mode rejects `"7.0"`), do integer arithmetic in `decimal(20,0)` (`exact_integers`), cast an integer beside a float column to float (`widened`; ANSI mode widens to double), use `try_cast` (Spark 3.2) / `try_subtract` (3.3) / `try_divide` (3.2) where the default mode gives null, guard denominators, and look up a map key only when it is present. Never change a Spark 3 result to get mode independence. `tests/test_ansi_mode.py` runs each edge case with ANSI mode on and off

## Contributing Guidelines

- Add tests for new features in `tests/test_<module>.py`
- Maintain the mixin-based architecture for model behaviors
- Keep the scikit-learn-like API consistent across all model types
- Use `ValidationError` / `ColumnNotFoundError` / `ModelNotFittedError` from `smallaxe.exceptions`; a public method never raises a raw `ValueError` / `KeyError` / `TypeError` for a user error
- A new public name goes in its module's `__all__` and in the `tests/test_public_api.py` snapshot, with a CHANGELOG entry
