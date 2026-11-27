> Note for AI agents working on this repository:
>
> Your job is to move smallaxe toward the goals in this document while keeping the library
> readable, simple to use, and extensible. Prefer clear APIs, small focused abstractions,
> and implementation patterns that match the existing codebase.
>
> Work on a feature branch from `main` (see `CONTRIBUTING.md`). The development setup,
> the test, lint and type-check commands, and the repository map are in `AGENTS.md`.
> Always validate changes with tests: run the relevant focused tests after each change,
> and the full suite (`pytest`) before considering work complete.
>
> If you are unsure about the current behavior, API, or implementation details of a
> dependency or library, use the available DeepWiki MCP tools to inspect authoritative
> project documentation before making assumptions.

# smallaxe Goals

## Product Goal

smallaxe should make common supervised modeling on PySpark DataFrames feel as simple as
scikit-learn on pandas, while keeping execution distributed through Spark-native and
Spark-compatible ML libraries.

The first stable target is:

- Binary classification.
- Standard continuous regression across Random Forest, LightGBM, XGBoost, and CatBoost
  regressors.
- A simple, consistent user API for preprocessing, training, evaluation, prediction,
  persistence, and pipeline composition.

Longer-term expansion should add multilabel classification and specialized regression
tasks such as quantile regression. Multiclass classification shipped with the first
stable target.

## Where 1.0 stands

The 1.0 API is on `main`, and the owner tags 1.0.0rc1 next. It has:

- The frozen public API: the `regressors` / `classifiers` factory modules, `Model`,
  `Regressor` and `Classifier`, `set_params`, and `dataframe=` on every public callable.
  `tests/test_public_api.py` pins it; `tests/test_removed.py` checks that every 0.x
  spelling is gone.
- Random Forest (PySpark MLlib), XGBoost (`xgboost.spark`), LightGBM (SynapseML) and
  CatBoost (`catboost-spark`), for `simple_regression`, `binary` and `multiclass`.
- `Imputer`, `Scaler`, `Encoder` and `Pipeline`, each saved and loaded as a versioned
  artifact through the Hadoop FileSystem; `smallaxe.load`, `load_model` and
  `load_pipeline`.
- `evaluate()` on models and pipelines, macro and weighted multiclass scores, and
  hyperparameter search (`smallaxe.search.optimize`).
- `smallaxe.auto.AutomatedTraining` and the seven charts of `smallaxe.viz`.
- PySpark 3.3 to 3.5 and 4.0 to 4.3, each tested in CI (`.github/workflows/ci.yml`), on
  a classic Spark session.

Earlier status notes, and the Databricks runs on 0.8 to 0.11 builds, are in the git
history of this file.

## Missing For v1

Each item keeps its original goal and has one status note for 1.0.

### 1. Align Public API With Actual Capabilities

- Update README to describe only implemented APIs, or implement the advertised APIs
  before release.
- Decide whether the first regression API is called "regression" or "linear regression."
  Random Forest, XGBoost, LightGBM, and CatBoost are not linear models. If true linear
  regression is a first-class goal, add a Spark `LinearRegression` baseline separately.

**Status (1.0):** done. Every README example runs in `tests/test_readme_examples.py`, and
the README Roadmap lists only what is not in 1.0. The regression task is named
`simple_regression` in code and "regression" in the docs. There is no linear-regression
baseline.

### 2. Finish The Four-Algorithm Training Surface

- Add CatBoost regressor and binary classifier support.
- Add factory methods for every algorithm in the regressor and classifier factories.
- Make optional dependency handling explicit: `available_models()` reports installed and
  unavailable models with install hints; factories raise clear `DependencyError` messages
  when a requested optional model is missing; tests verify missing optional dependency
  behavior without being globally skipped.
- Normalize model parameter names across algorithms where possible (`n_estimators`,
  `max_depth`, `learning_rate`, `seed`), with each algorithm translating them to its
  own names.

**Status (1.0):** done. `regressors` and `classifiers` have one function per algorithm,
each documenting the algorithm's whole parameter table (`tests/test_factories.py`). A
missing package raises `DependencyError` with the install hint
(`tests/test_factories.py`, `tests/test_algorithm.py`), and `tests/test_algorithm.py`
checks the translation of all four algorithms against fake estimator modules, so
LightGBM and CatBoost are covered without their JVM packages.

### 3. Make Preprocessing Production-Ready

- Split categorical and numeric preprocessing into predictable steps: numeric
  imputation, categorical imputation, categorical encoding, numeric scaling, feature
  vector assembly.
- Add a fitted preprocessing schema artifact: input columns, output feature columns,
  encoded category mappings, unknown-category behavior, null handling behavior.
- Replace Python UDF extraction in Scaler/Encoder where practical with Spark SQL/vector
  functions.
- Ensure transform-time behavior is stable for unseen categories, missing columns, and
  changed schemas.
- Avoid silently dropping rows during feature assembly.

**Status (1.0):** done. `Imputer`, `Scaler` and `Encoder` each save their fitted state
(columns, fill values, category mappings) as a versioned artifact. The `Encoder`
reserves an `__UNKNOWN__` index, so an unseen category never fails one-hot encoding or
becomes null. Rows with a null or NaN feature are never dropped: Random Forest raises
`ValidationError` naming the columns, and the boosting algorithms keep the rows
(`tests/test_missing_values.py`). No preprocessing step uses a Python UDF.

### 4. Harden Pipeline Semantics

- Pipeline should own feature-column construction instead of passing all non-label
  columns to the model.
- Pipelines should support both preprocessing-only `fit/transform` and end-to-end
  `fit/predict/evaluate/save/load` with a model step.
- Save/load must preserve preprocessing state, model artifacts, feature schema, label
  column, task type, model params, and validation/evaluation metadata.
- Add tests for saving and loading full pipelines.

**Status (1.0):** done. The model is fitted on exactly `numerical_cols` plus the
Encoder's output columns. A pipeline with a model step saves and loads with identical
predictions (`tests/test_pipeline_persistence.py`), and `Pipeline.fit` takes the
model's validation options. Nothing is pickled; the one pickle still read is the
`Encoder` inside a pipeline saved by 0.10.

### 5. Evaluation API

- Add a model-level `evaluate(dataframe, label_col=None)` method and a pipeline-level
  `evaluate(...)` that preprocesses, predicts, and scores in one call.
- For binary classification, support accuracy, precision, recall, F1, ROC AUC, PR AUC,
  log loss and a confusion matrix.
- For regression, support RMSE, MAE, MSE, R2 and MAPE.
- Keep multiclass and multilabel metrics separate from binary metrics, with an explicit
  averaging policy.

**Status (1.0):** done. `Model.evaluate` and `Pipeline.evaluate` return the metrics of
the task (`smallaxe.metrics.TASK_METRICS`; `tests/test_evaluate.py`). Multiclass
validation uses macro averages and also reports `*_macro` and `*_weighted` scores. The
confusion matrix is a chart, `smallaxe.viz.confusion_matrix`; there is no metric
function that returns its counts. Multilabel is not supported.

### 6. Training And Validation

- Move train/test split and k-fold logic into a dedicated validation module, with public
  split utilities.
- Make validation behavior explicit: `validation="none" | "train_test" | "kfold"`, fixed
  seed behavior, empty fold and tiny-class handling.
- Add train/validation metrics and final model metadata in a consistent structure.
- Add an option to cache training data during fitting, with documented tradeoffs.

**Status (1.0):** partly done. A row's split or fold is a hash of its content (or
`fold_key`) and the seed, so it does not depend on partitioning or caching
(`tests/test_validation.py`). An empty split or fold raises. The kept model is refitted
on every row by default (`refit=True`). `cache_strategy` defaults to caching only when
`fit` reads the data more than once, and never uncaches the caller's DataFrame. The
`stratified` argument is gone: a content hash is label-independent, so every class
splits in the same proportion in expectation. Not done: the split logic is still in
`ValidationMixin`, and there are no public split utilities.

### 7. Model Persistence And Registry-Ready Artifacts

- Define a stable artifact layout with version metadata (`smallaxe_version`, Spark
  version, algorithm name, task type, params, feature schema, timestamp).
- Provide `load_model(path)` and `load_pipeline(path)` convenience functions.
- Ensure loaded models produce the same predictions as saved models on deterministic
  test data.
- Design the artifact format so it can later plug into MLflow or a model registry.

**Status (1.0):** done, except the registry. Every artifact is a directory with a
`metadata.json` holding `format_version`, `kind` and `smallaxe_version`; a model's also
records the algorithm, task, parameters, feature columns, Spark version and training
timestamp. Metrics and schema are keys of `metadata.json`, not separate files.
`smallaxe.load`, `load_model` and `load_pipeline` exist. Random Forest and XGBoost models
saved by 0.8.1, 0.9.0 and 0.10.0 load and predict identically
(`tests/test_legacy_artifacts.py`). MLflow
integration is not done.

### 8. Automated Training

- Implement `AutomatedTraining` after the four algorithm wrappers are stable. It should
  train all available compatible algorithms, skip missing optional dependencies with
  warnings and install hints, return a comparison table, select `best_model` by a
  user-specified metric, and persist the winning model or full comparison run.
- Keep the first version constrained to binary classification and continuous regression.

**Status (1.0):** done. `smallaxe.auto.AutomatedTraining` trains every installed
algorithm for `simple_regression` or `binary` on one split seed, skips missing packages
with a warning and install hint, reports training failures and continues, returns the
comparison as a pandas DataFrame, and selects `best_model` by the chosen metric.
Algorithms given a `param_spaces` entry are tuned with `optimize.run`. Persisting is the
winner's own `save` plus `comparison.to_csv`; there is no artifact for the whole run.
Multiclass is not supported. Locally `tests/test_auto.py` runs Random Forest and XGBoost
and covers LightGBM and CatBoost with fake fits; on Databricks Runtime 17.3 LTS
(Spark 4.0) it ranked all four algorithms on binary and regression (see "Databricks
validation" below).

### 9. Hyperparameter Search

- Implement `smallaxe.search.optimize` with a simple API: model, DataFrame, label column,
  search space, metric, validation strategy, max evaluations. Preserve `best_params`,
  `best_score`, and trial history.

**Status (1.0):** done (`tests/test_optimize.py`). Every trial scores on the same split.
The search takes a model, not a Pipeline, so tuning with categoricals means encoding the
frame first.

### 10. Documentation And Examples

- Rewrite README around the v1 user journey: install, build a preprocessing pipeline,
  train a binary classifier, train a regressor, evaluate, save/load, use optional
  algorithms.
- Add examples for Random Forest binary classification, XGBoost regression, LightGBM
  classification when the dependency is installed, and full pipeline save/load.
- Add a compatibility matrix for Python, Spark, Java, and optional algorithm packages.

**Status (1.0):** done. The README covers that journey, and its Python examples run in
`tests/test_readme_examples.py`, including a full pipeline save and load. The Kaggle
scripts in `examples/` cover Random Forest classification and Random Forest and XGBoost
regression; the Databricks notebooks cover LightGBM and CatBoost. The README
compatibility matrix follows `ci.yml` leg by leg.

### 11. Spark 4 Support

**Status (1.0):** done. `pyspark>=3.3,<4.4`, with CI legs for PySpark 4.0 to 4.3 (see
the README compatibility matrix). Results do not depend on ANSI SQL mode
(`tests/test_ansi_mode.py`). Classic sessions only: a Spark Connect session raises
`ConfigurationError` (`tests/test_classic_session.py`). XGBoost on Spark 4 needs xgboost
3.3 or later. LightGBM and CatBoost are validated on Spark 4.0; their Spark 4.1 builds
are not.

## Databricks validation

`examples/databricks_all_algorithms_validation.py` runs all four algorithms on four real
datasets with numeric and categorical features (Diamonds for regression, Telco churn and
Adult income for binary, a 60,000-row Covertype sample for 7-class multiclass). The two
runs below used earlier versions of the notebook, each on a cluster with 2 fixed
m5d.8xlarge workers.

| Date | Runtime | Maven libraries | Commit | Result |
|---|---|---|---|---|
| 2026-11-26 | DBR 16.4 LTS (Spark 3.5.2, Scala 2.12, Python 3.12.3) | `synapseml-lightgbm_2.12:1.1.3`, `catboost-spark_3.5_2.12:1.2.10` | e41b8eb (the 1.0 API) | 122 of 122 checks passed for Random Forest, XGBoost, LightGBM and CatBoost on regression, binary and multiclass, including Pipeline save and load on `dbfs:/` |
| 2026-11-26 | DBR 17.3 LTS (Spark 4.0.0, Scala 2.13.16, Python 3.12.3) | `synapseml-lightgbm_2.13:1.1.3-spark4.0` (repository `https://mmlspark.azureedge.net/maven`), `catboost-spark_4.0_2.13:1.2.10`; xgboost 3.4.1 | a7c8ce7 | all 124 library checks passed for the four algorithms, including AutomatedTraining ranking all four on binary and regression; a separate run built seven charts per algorithm and `model_comparison` from an AutomatedTraining |
| 2026-11-26 | DBR 16.4 LTS (Spark 3.5.2, Scala 2.12, Python 3.12.3) | as the first row | da2a4cc (release candidate; the same files as main at 97d4cd2) | 122 of 128 checks passed. Random Forest, XGBoost and CatBoost passed every check, as did AutomatedTraining and the charts. One LightGBM fit (binary, 3-fold) failed with `java.net.ConnectException: Connection refused` after an executor exited with code 134; that fit accounts for the 2 failed and 4 blocked checks. LightGBM-only reruns lost executors the same way on this commit and on e41b8eb, so the failure is not specific to this commit; see "Worker failures can be intermittent" in `GOTCHAS.md` |

In one earlier run a CatBoost regression fit failed once with "Error while executing
workers" and then passed 20 of 20 repeats. `GOTCHAS.md` treats CatBoost-Spark worker
failures as intermittent.

The notebook is the only run that exercises LightGBM and CatBoost against their JVM
packages; CI does not install them. Rerun it before each release.

## Architecture Deepening (2026-10-05)

An architecture review traced most of the early Databricks gaps to six structural
problems. Their state at 1.0:

1. One model module with one adapter per algorithm. Done: each algorithm is a
   declarative `Algorithm` record (`GLOSSARY.md`), `Model` / `Regressor` / `Classifier`
   replace the eight per-algorithm classes, and `_estimator_kwargs()` is the one
   translation rule. Not done: an `extra_params` pass-through for estimator parameters
   outside the table.
2. One persistence module for models, Imputer, Scaler, Encoder, and Pipeline. Done: every
   step has `save` / `load` through `smallaxe._fs` and `smallaxe._artifact`, and
   `smallaxe.load(path)` dispatches on the artifact's kind.
3. Pipeline passes declared columns from step to step. Done: whether a model needs an
   Encoder comes from `accepts_raw_categoricals` on its record.
4. The Encoder owns unseen and null categories. Done.
5. Task-aware scoring in one place. Done: `smallaxe.metrics.TASK_METRICS`, read by `fit`,
   `evaluate`, `optimize.run` and `AutomatedTraining`.
6. Fold the five mixins into the model module. Not done: `BaseModel` still composes the
   five mixins of `smallaxe.training.mixins`.

## v1 Acceptance Criteria

- A new user can train, evaluate, save, load, and predict with Random Forest on a PySpark
  DataFrame in under 20 lines of code.
- The same user-facing workflow works for XGBoost, LightGBM, and CatBoost when optional
  dependencies are installed.
- Binary classification and continuous regression have clear metrics and stable output
  schemas.
- A full preprocessing-plus-model pipeline can be saved and loaded with identical
  predictions on deterministic data.
- Missing optional dependencies fail with actionable install instructions.
- Documentation does not advertise unimplemented APIs.
- CI runs core tests on supported Python/Spark versions and optional algorithm tests in
  separate dependency-enabled jobs.

Status (1.0):

1. Random Forest in under 20 lines: met. The README Quick Start is 19 lines and trains,
   evaluates (`validation_scores` on a holdout), saves, loads and predicts.
   `tests/test_readme_examples.py` runs it and checks its length
   (`test_quick_start_is_one_runnable_block_under_20_lines`).
2. The same workflow for XGBoost, LightGBM and CatBoost: met. `tests/test_model_contract.py`
   runs fit, predict, train/test and k-fold validation, `predict_proba`, feature
   importances and save/load for each (algorithm, task) where the package is installed;
   the CI `test-xgboost` job runs it for XGBoost. LightGBM and CatBoost passed the
   Databricks runs above on Spark 3.5 and 4.0.
3. Clear metrics and stable output schemas for binary and regression: met. `evaluate` and
   `validation_scores` return the task's `TASK_METRICS` (`tests/test_evaluate.py`,
   `tests/test_metrics_classification.py`, `tests/test_metrics_regression.py`).
   `predict` returns one row per input row and keeps the user's columns
   (`tests/test_missing_values.py`, `tests/test_user_columns.py`).
4. Pipeline save and load with identical predictions: met.
   `tests/test_pipeline_persistence.py` checks it locally for Random Forest and XGBoost on
   all three tasks. The DBR 16.4 LTS run, which covered all four algorithms, included
   Pipeline save and load on `dbfs:/`.
5. Actionable install instructions: met. A missing package raises `DependencyError`
   naming the install command, and `available_models()` lists the install hints
   (`tests/test_factories.py`, `tests/test_algorithm.py`).
6. Documentation does not advertise unimplemented APIs: met. Every README example runs in
   `tests/test_readme_examples.py`, and the Roadmap lists what 1.0 does not have.
7. CI with separate dependency-enabled jobs: met for XGBoost, not for LightGBM and
   CatBoost. The `test` job runs the core suite on 24 Python / PySpark / Java legs and
   `test-lowest` on the lowest allowed versions; `test-xgboost` runs the suite with
   XGBoost on PySpark 3.3, 3.5 and 4.3. LightGBM and CatBoost have no CI job, because
   they need JVM packages; the Databricks notebook stands in for one, and
   `tests/test_algorithm.py` checks their translation with fake estimators.

Release-candidate run on Databricks (DBR 16.4 LTS, Spark 3.5) for da2a4cc, on
2026-11-26: 122 of 128 checks passed. Random Forest, XGBoost and CatBoost passed every check, as did AutomatedTraining and the charts. One LightGBM fit (binary, 3-fold) failed with `java.net.ConnectException: Connection refused` after an executor exited with code 134; that fit accounts for the 2 failed and 4 blocked checks. LightGBM-only reruns lost executors the same way on this commit and on e41b8eb, so the failure is not specific to this commit; see "Worker failures can be intermittent" in `GOTCHAS.md`.

## Later Goals

- Multilabel classification.
- Quantile regression and other specialized regression objectives.
- Calibration and threshold tuning for binary classifiers.
- A threshold curve and multiclass (one-vs-rest) ROC curves in `smallaxe.viz`.
- MLflow integration for experiment tracking and model registry workflows.
- Distributed hyperparameter tuning with Spark-aware execution.
- Hyperparameter search over a whole `Pipeline`, native CatBoost categoricals, multiclass
  in `AutomatedTraining`, and Spark Connect sessions.
