> Note for AI agents working on this repository:
>
> Your job is to move smallaxe toward the goals in this document while keeping the library readable, simple to use, and extensible. Prefer clear APIs, small focused abstractions, and implementation patterns that match the existing codebase.
>
> Make goal-related changes on a git branch named `goals`. If the branch does not exist, create it before editing.
>
> Always validate changes with tests. The Python environment is managed with UV and should be activated with:
>
> ```bash
> source ~/Desktop/basic/bin/activate
> ```
>
> If a new Python library is needed in the environment, install it with:
>
> ```bash
> uv pip install <library-name>
> ```
>
> For PySpark tests on this machine, use OpenJDK 11:
>
> ```bash
> export JAVA_HOME=/opt/homebrew/opt/openjdk@11
> export PATH="$JAVA_HOME/bin:$PATH"
> ```
>
> Run relevant focused tests after each change, and run the full suite before considering work complete:
>
> ```bash
> pytest -q
> ```
>
> If you are unsure about the current behavior, API, or implementation details of a dependency or library, use the available DeepWiki MCP tools to inspect authoritative project documentation before making assumptions.

# smallaxe Goals

## Product Goal

smallaxe should make common supervised modeling on PySpark DataFrames feel as simple as scikit-learn on pandas, while keeping execution distributed through Spark-native and Spark-compatible ML libraries.

The first stable target is:

- Binary classification.
- Standard continuous regression across Random Forest, LightGBM, XGBoost, and CatBoost regressors.
- A simple, consistent user API for preprocessing, training, evaluation, prediction, persistence, and pipeline composition.

Longer-term expansion should add multiclass classification, multilabel classification, and specialized regression tasks such as quantile regression.

## Current Baseline

The current implementation already has useful foundations:

- Global configuration, custom exceptions, sample datasets, metrics, preprocessing, pipeline, and training modules.
- Random Forest regressors/classifiers backed by PySpark ML.
- Optional XGBoost, LightGBM (SynapseML), and CatBoost (catboost-spark) wrappers, all reachable from the `Regressors` / `Classifiers` factories.
- Imputer, Scaler, Encoder, and Pipeline classes.
- Model metadata, validation scores, feature importance (Random Forest only, see below), and save/load support for individual models.
- Hyperparameter search (`smallaxe.search.optimize`, hyperopt-backed).
- A substantial test suite. With `~/Desktop/basic`, PySpark 3.5.x, and OpenJDK 11, the current suite passes: 556 passed, 67 skipped (2026-09-30). The skipped tests need the LightGBM/CatBoost JVM packages, or are the "dependency not installed" variants.
- An end-to-end Databricks validation notebook covering all four algorithms on real mixed-type data: `examples/databricks_all_algorithms_validation.py`. See "Databricks Validation (2026-09-30)" below.

## Missing For v1

### 1. Align Public API With Actual Capabilities

- Update README to describe only implemented APIs, or implement the advertised APIs before release.
- Current README advertises `smallaxe.search.optimize`, `smallaxe.auto.AutomatedTraining`, visualization, and CatBoost, but those modules are empty or missing.
- Decide whether the first regression API is called "regression" or "linear regression." Random Forest, XGBoost, LightGBM, and CatBoost are not linear models. If true linear regression is a first-class goal, add a Spark `LinearRegression` baseline separately.

**Status (2026-09-30):** mostly done. `search.optimize` and CatBoost are implemented, and the README lists `AutomatedTraining` and visualization only under "Roadmap". Still open: the regression naming decision.

### 2. Finish The Four-Algorithm Training Surface

- Add CatBoost regressor and binary classifier support, or remove CatBoost from public docs until implemented.
- Add factory methods for LightGBM in `Regressors` and `Classifiers`; the classes exist, but the factories only expose Random Forest and XGBoost.
- Make optional dependency handling explicit:
  - `available_models()` should report installed and unavailable models with install hints.
  - Factories should raise clear `DependencyError` messages when a requested optional model is missing.
  - Tests should verify missing optional dependency behavior without being globally skipped.
- Normalize model parameter names across algorithms where possible:
  - User-facing: `n_estimators`, `max_depth`, `learning_rate`, `seed`.
  - Internal adapters translate to Spark/XGBoost/LightGBM/CatBoost-specific names.

**Status (2026-09-30):** done, including three fixes found by the Databricks validation:
- CatBoost now works in the same session as LightGBM. SynapseML monkeypatches PySpark's `JavaParams` process-wide, which broke every CatBoost fit and CatBoost model load; `_ensure_synapseml_compat()` in `training/catboost.py` routes CatBoost's values back to CatBoost's converter.
- `LightGBMClassifier(task="multiclass")` now passes `objective="multiclass"`. It previously trained a binary model on multiclass labels without any error.
- The `smallaxe[lightgbm]` / `smallaxe[all]` extras now depend on `synapseml`. They previously named `synapse-ml-lightgbm`, which does not exist on PyPI, so both extras failed to install.

Still open: "tests should verify missing optional dependency behavior without being globally skipped". `tests/test_lightgbm.py` is skipped wholesale without SynapseML, which is how the multiclass bug went unnoticed.

### 3. Make Preprocessing Production-Ready

- Split categorical and numeric preprocessing into predictable steps:
  - Numeric imputation.
  - Categorical imputation.
  - Categorical encoding.
  - Numeric scaling when useful.
  - Feature vector assembly.
- Add a fitted preprocessing schema artifact:
  - Input columns.
  - Output feature columns.
  - Encoded category mappings.
  - Unknown-category behavior.
  - Null handling behavior.
- Replace Python UDF extraction in Scaler/Encoder where practical with Spark SQL/vector functions for performance.
- Ensure transform-time behavior is stable for unseen categories, missing columns, and changed schemas.
- Avoid silently dropping rows during feature assembly. Current `VectorAssembler(handleInvalid="skip")` can change row counts during training or prediction.

**Status (2026-09-30):** open. Confirmed on Databricks:
- One-hot `Encoder` crashes the whole `predict` when a category was not seen during fit (unless the column has more than `max_categories` levels, which routes unseen values to `__OTHER__`).
- Label encoding turns an unseen category into null, and assembly then drops that row.
- `predict` on rows with nulls and no Imputer silently dropped 513 of 10,892 rows.
- `Encoder` has no `save()` / `load()`.

### 4. Harden Pipeline Semantics

- Pipeline should own feature-column construction instead of passing all non-label columns to the model.
- Pipelines should support both:
  - Preprocessing-only `fit/transform`.
  - End-to-end `fit/predict/evaluate/save/load` with a model step.
- Add robust pipeline persistence for model pipelines, not only preprocessing pipelines.
- Save/load must preserve:
  - Preprocessing state.
  - Model artifacts.
  - Feature schema.
  - Label column.
  - Task type.
  - Model params.
  - Validation/evaluation metadata.
- Add tests for saving and loading full pipelines with Random Forest first, then optional algorithm-specific tests.

**Status (2026-09-30):** open. Confirmed locally and on Databricks:
- Any pipeline containing an `Encoder` fails to save (`TypeError: cannot pickle '_thread.RLock'`). Pipeline falls back to pickling the Encoder, which holds live Spark models.
- A pipeline with a model step saves, but `Pipeline.load` fails (`Could not load step .../step_N_model`).
- `Pipeline.save` / `load` use local `os` / `open`, not the Hadoop filesystem layer that model save/load already uses, so `/dbfs/...` paths fail too.
- Pipeline passes every non-label column to the model, so an unlisted string ID column (e.g. `customerID`) makes fit fail.

### 5. Evaluation API

- Add a model-level `evaluate(df, label_col=None, metrics=None)` method.
- Add a pipeline-level `evaluate(...)` method that preprocesses, predicts, and scores in one call.
- For binary classification, support at least:
  - Accuracy.
  - Precision.
  - Recall.
  - F1.
  - ROC AUC.
  - PR AUC.
  - Log loss.
  - Confusion matrix.
- For regression, support at least:
  - RMSE.
  - MAE.
  - MSE.
  - R2.
  - MAPE.
- Keep multiclass and multilabel metrics separate from binary metrics. The current binary precision/recall/F1 implementation should not be reused for multiclass without explicit averaging policy.

**Status (2026-09-30):** open, and the multiclass metrics are actively wrong. For `task="multiclass"`, the precision, recall, and F1 in `validation_scores` equal class 1's one-vs-rest values, not an average. This was checked against scikit-learn on 7-class Covertype for all four algorithms. For example, Random Forest reported F1 0.754 against a macro F1 of 0.373. It also affects `search.optimize` when it optimizes those metrics. Regression and binary metrics match scikit-learn exactly. There is no `evaluate()` and no confusion matrix yet.

### 6. Training And Validation

- Move train/test split and k-fold logic into a dedicated validation module.
- Add public split utilities for reuse and testing.
- Make validation behavior explicit:
  - `validation="none" | "train_test" | "kfold"`.
  - `stratified=True` only for classification.
  - Fixed seed behavior.
  - Empty fold and tiny-class handling.
- Add train/validation metrics and final model metadata in a consistent structure.
- Add an option to cache training data during fitting, with documented tradeoffs.

**Status (2026-09-30):** partially done. `validation` / `stratified` / `n_folds` / `cache_strategy` work for all four algorithms on Databricks, but the split logic still lives in `ValidationMixin`, with no public split utilities. Bug: `fit(cache_strategy="memory" | "disk")` calls `unpersist()` on the caller's own DataFrame, so a DataFrame the user cached before `fit` comes back uncached.

### 7. Model Persistence And Registry-Ready Artifacts

- Define a stable artifact layout:
  - `metadata.json`.
  - `preprocessing/`.
  - `model/`.
  - `metrics.json`.
  - `schema.json`.
- Include a `smallaxe_version`, Spark version, algorithm name, task type, params, feature schema, and timestamp.
- Provide `load_model(path)` and `load_pipeline(path)` convenience functions.
- Ensure loaded models produce the same predictions as saved models on deterministic test data.
- Design the artifact format so it can later plug into MLflow or a model registry.

**Status (2026-09-30):** partially done. Model-level `save` / `load` (via the factories) round-trips with identical predictions for all four algorithms and all three tasks on `dbfs:/` paths. The stable artifact layout, version metadata, and `load_model` / `load_pipeline` helpers are not done. Pipeline persistence is broken (see item 4).

### 8. Automated Training

- Implement `AutomatedTraining` after the four algorithm wrappers are stable.
- It should:
  - Train all available compatible algorithms.
  - Skip missing optional dependencies with warnings and install hints.
  - Return a comparison table as a Spark or pandas DataFrame.
  - Select `best_model` by a user-specified metric.
  - Persist the winning model or full comparison run.
- Keep the first version constrained to binary classification and continuous regression.

**Status (2026-09-30):** not started (`smallaxe/auto/` is empty).

### 9. Hyperparameter Search

- Implement `smallaxe.search.optimize`.
- Start with a simple, predictable API:
  - model instance.
  - DataFrame.
  - label column.
  - search space.
  - metric.
  - validation strategy.
  - max evaluations.
- Preserve `best_params`, `best_score`, and trial history.
- Make search optional and clearly dependency-gated if using Hyperopt.

**Status (2026-09-30):** done. Validated on Databricks for all four algorithms, for binary (`auc_roc`) and regression (`rmse`). The search takes a model, not a Pipeline, so tuning with categoricals means pre-encoding the frame first.

### 10. Documentation And Examples

- Rewrite README around the actual v1 user journey:
  - Install.
  - Build a preprocessing pipeline.
  - Train binary classifier.
  - Train regressor.
  - Evaluate.
  - Save/load.
  - Use optional algorithms.
- Add examples for:
  - Random Forest binary classification.
  - XGBoost regression.
  - LightGBM classification when dependency is installed.
  - Full pipeline save/load.
- Add a compatibility matrix for Python, Spark, Java, and optional algorithm packages.

**Status (2026-09-30):** partially done. The README has the compatibility matrix, with the Databricks requirements corrected by this validation. Kaggle end-to-end scripts cover Random Forest classification and Random Forest + XGBoost regression, and `examples/databricks_all_algorithms_validation.py` covers all four algorithms. Still missing: a full pipeline save/load example (blocked on item 4).

## Databricks Validation (2026-09-30)

`examples/databricks_all_algorithms_validation.py` ran all four algorithms on four real datasets with both numeric and categorical features:
- Diamonds: regression.
- Telco churn: binary, with numeric missing values.
- Adult income: binary, with categorical missing values.
- Covertype: 7 classes, with 40-level categoricals.

For each algorithm it covers the full Pipeline, row preservation, metrics cross-checked against scikit-learn, train_test and stratified k-fold validation, `predict_proba`, `feature_importances`, save/load, `search.optimize`, and label encoding. It also probes each known gap.

| Run | Build | Cluster | PASS | GAP | FAIL | BLOCKED |
|---|---|---|---|---|---|---|
| 1 | PyPI 0.8.0 | autoscaling 2–8 | 47 | 18 | 11 | 8 |
| 2 | 0.8.0 + fixes above | autoscaling 2–8 | 55 | 20 | 6 | 3 |
| 3 | 0.8.0 + fixes above | fixed 2 workers | 62 | 21 | 1 | 0 |
| 4 | 0.8.2.dev4 (`deepen-algorithm-seam`, 2026-10-05) | fixed 2 workers | 69 | 15 | 0 | 0 |

GAP means a known library gap (item status notes above); the probe starts passing once the gap is fixed.

Run 4 (the algorithm-seam refactor) kept every run-3 PASS, turned the six XGBoost/LightGBM `feature_importances` probes into PASS, and the run-3 CatBoost Diamonds RMSE overflow did not recur. Its CatBoost `feature_importances` probes stayed GAP for a new reason: SynapseML also patches `JavaWrapper._call_java`, which pickles CatBoost's `EFstrType` enum argument. Fixed in 0.8.2.dev5 by routing CatBoost model calls through the compatibility shim too.

Environment requirements:
- DBR 16.4 LTS **Scala 2.12** (Spark 3.5.2) is the only LTS runtime that can host all four algorithms, because SynapseML has no Scala 2.13 or Spark 4 build.
- Maven packages: `com.microsoft.azure:synapseml-lightgbm_2.12:1.1.3` and `ai.catboost:catboost-spark_3.5_2.12:1.2.10`. On Databricks, the SynapseML jar also puts `synapse.ml` on the Python path.
- A **fixed-size cluster** for CatBoost. CatBoost-Spark training fails ("Error while executing workers", worker exit 134) whenever executors join or leave mid-fit. All 13 CatBoost failures under autoscaling lined up with resize events, and there were none on a fixed cluster.

Findings not covered by the items above:
- `feature_importances` returned `None` for XGBoost, LightGBM, and CatBoost. It only read `featureImportances`, which only Random Forest exposes. **Fixed on `deepen-algorithm-seam`:** each algorithm record now carries its own importances reader (XGBoost gain, LightGBM split counts, CatBoost PredictionValuesChange; raw library scales, not normalised).
- CatBoost native categoricals are not implemented. Pipeline exempts CatBoost from needing an `Encoder`, but raw string columns then fail at vector assembly.
- Databricks MLflow autologging logs every internal Spark ML fit, including each fold, each hyperopt trial, and the StandardScaler/OneHotEncoder fits: about 250 runs per validation run. Setting `spark.databricks.mlflow.autologging.enabled=false` at runtime did not stop it.
- The SynapseML/CatBoost shim assumes SynapseML is imported last, as smallaxe's factories do. If catboost_spark only becomes importable later (the lazy `_load_catboost_spark()` path), its `_from_java` wins instead, and loading a saved LightGBM model may break.
- `pyspark<4.0` is stricter than necessary on Databricks: Random Forest and XGBoost ran on DBR 17.3 (Spark 4.0). `catboost-spark_4.0_2.13` exists; SynapseML has no Spark 4 build.
- The one remaining FAIL: in one CatBoost Diamonds pipeline fit, RMSE/MSE overflowed to infinity while MAE and R² matched scikit-learn. It did not reproduce in 4 identical reruns and is unexplained. CatBoost-Spark is also not fully deterministic for a fixed seed across distributed runs.
- LightGBM's multiclass bug slipped through because `tests/test_lightgbm.py` only asserted `_is_fitted`, and the whole file is skipped without SynapseML. That test now checks the probability vector length.

## Recommended Order Of Execution

Rerun `examples/databricks_all_algorithms_validation.py` after each step. A step is done when its GAP probes become PASS with no new FAIL, and the local suite still passes.

1. **Multiclass metrics (item 5).** Add macro and weighted precision, recall, and F1 for `task="multiclass"`, and keep the binary formulas for binary only. This comes first because it silently reports wrong numbers today, including inside `search.optimize`. Done when the four multiclass metric probes match scikit-learn.
2. **Pipeline persistence (items 4 and 7).** Give `Encoder` its own `save` / `load`, load model steps through the `Regressors` / `Classifiers` factories, and route pipeline IO through `smallaxe._fs`. This is a v1 acceptance criterion. Done when all three pipeline save/load probes pass and a reloaded pipeline predicts identically, locally and on `dbfs:/`.
3. **Unseen categories and row preservation (item 3).** Map unseen categories to an all-zeros vector or `__OTHER__` instead of crashing, and stop dropping rows at assembly: keep them, or fail loudly. Done when the unseen-category and null-row probes pass.
4. **Pipeline owns its feature columns (item 4).** Build the model's features from `numerical_cols` and the Encoder's output columns, and ignore unlisted columns. Done when the ID-column probe passes.
5. **Small correctness fixes.** Stop `fit(cache_strategy=...)` from unpersisting the caller's DataFrame (item 6), and implement `feature_importances` for XGBoost, LightGBM, and CatBoost.
6. **Evaluation API (item 5).** Model- and pipeline-level `evaluate()`, plus a confusion matrix.
7. **CatBoost categoricals.** Either implement native categorical handling or remove the Pipeline's CatBoost exemption from the Encoder requirement.
8. **Databricks ergonomics.** An option to suppress MLflow autologging during internal fits, a warning when CatBoost trains on an autoscaling cluster, and hardening of the SynapseML/CatBoost shim for the reverse import order.
9. **Validation module and artifact layout (items 6 and 7).** Public split utilities, a versioned artifact layout, and `load_model` / `load_pipeline`.
10. **AutomatedTraining (item 8), then visualization.**
11. **Spark 4 support.** Relax `pyspark<4.0` once RF, XGBoost, and CatBoost (`catboost-spark_4.0_2.13`) pass the validation notebook on DBR 17.x. LightGBM stays Spark 3.5-only until SynapseML ships a Spark 4 build.

Cross-cutting: make the Databricks validation notebook a release gate. It is currently the only run that exercises LightGBM and CatBoost against their JVM packages; CI skips those tests.

## Architecture Deepening (2026-10-05)

An architecture review traced most of the Databricks gaps to six structural problems. Fixing each removes a class of bug instead of one instance. They overlap with the execution order above; matching steps are noted.

1. **One model module with one adapter per algorithm. Implemented 2026-10-05 on `deepen-algorithm-seam`, ahead of step 1; merge gate is the Databricks validation rerun.**
   - Found on the way: `smallaxe[xgboost]` installed only `xgboost`, but `xgboost.spark` imports scikit-learn and runs on pyarrow, so on a clean machine `Regressors.xgboost()` reported "xgboost is not installed". The extra now includes both; Databricks had them preinstalled, which is why it never showed there.
   - Today: 8 near-identical Regressor/Classifier classes (about 420 duplicated lines), 2 factories, and availability checks in 5 places each repeat param translation, model-class choice, and loading.
   - Per-algorithm differences have no hook. That produced the `feature_importances` gap, the LightGBM multiclass objective bug, and CatBoost's full `_fit_spark_model` override.
   - Target: the model module owns fit, predict, save, and importances. The RF, XGBoost, LightGBM, and CatBoost adapters each declare a param table, an estimator builder, importances, and session setup.
   - Decided design (see `GLOSSARY.md`): each algorithm is a declarative `Algorithm` record in its existing file, with a strict `Param` table, estimator and model classes per task as lazily resolved dotted paths, task-fixed params (LightGBM `objective`, CatBoost `lossFunction`), a per-algorithm importances callable, and at most two code hooks (`prepare` for the CatBoost JVM shim, `fit_context` for its temp dir). One pure translation function in `smallaxe/training/algorithm.py` serves all four. `Regressor` / `Classifier` in `base.py` replace the eight classes, which stay as deprecated aliases until v1; factories return the plain classes. New artifacts record `algorithm` + `task`; 0.8.x artifacts load through a name map, locked by fixtures saved with 0.8.1. `extra_params` pass-through is a later follow-up.
   - Tests: one contract test parametrized over (algorithm, task) against a fake estimator module, so LightGBM and CatBoost translation runs without a JVM; per-algorithm files keep only their specifics.
   - Stage RF + XGBoost first, and add xgboost to the CI install. CI installs only `.[dev]`, so every XGBoost, LightGBM, and CatBoost test is skipped there. One PR; the Databricks validation rerun is the merge gate.
   - Covers the `feature_importances` half of step 5, and supplies the "needs an Encoder" flag for steps 4 and 7.
2. **One persistence module for models, Imputer, Scaler, Encoder, and Pipeline.**
   - Today: only models use `smallaxe._fs`. Imputer and Scaler write with `os`, the Encoder has no `save`, and Pipeline pickles it. Its Spark OneHotEncoder models cannot be pickled, and Pipeline cannot load model steps.
   - Target: one `save(obj, path)` / `load(path)` keyed by a type tag. Each step supplies its state, and all IO goes through `_fs`.
   - Same goal as step 2, done once for every step type.
3. **Pipeline passes declared columns from step to step.**
   - Today: `_get_feature_cols` hands the model every non-label column. Steps are dispatched on `type(step).__name__`, including the wrong CatBoost exemption from the Encoder.
   - Target: each step takes and returns the numerical, categorical, and encoded column lists, and the model gets only numerical + encoded columns. Whether a model needs an Encoder comes from its adapter (needs 1).
   - Same goal as step 4.
4. **The Encoder owns unseen and null categories.**
   - Today: three Spark stages apply three null policies (`error`, `skip`, `keep`). So onehot crashes on new categories, and label encoding drops rows.
   - Target: the Encoder always reserves an unknown index and builds one-hot columns from its own mapping, and one shared assembly step carries one null policy.
   - Same goal as step 3. It also removes the unpicklable object behind 2.
5. **Task-aware scoring in one place.**
   - Today: the metric functions are correct, but `base.py` calls the binary formulas for every task. `search/optimize.py` keeps its own copy of metric names, directions, and the k-fold key format.
   - Target: `evaluate(task, df)` returns scores that both `fit` and `optimize` read.
   - Same goal as step 1, extended to `optimize`.
6. **Fold the five mixins into the model module.**
   - Today: `BaseModel` is their only consumer, and they share state through implicit attributes, so following `fit` means reading seven files.
   - Mostly falls out of 1 and 2. The caller's-cache fix (step 5) is a few lines on its own.

## v1 Acceptance Criteria

- A new user can train, evaluate, save, load, and predict with Random Forest on a PySpark DataFrame in under 20 lines of code.
- The same user-facing workflow works for XGBoost, LightGBM, and CatBoost when optional dependencies are installed.
- Binary classification and continuous regression have clear metrics and stable output schemas.
- A full preprocessing-plus-model pipeline can be saved and loaded with identical predictions on deterministic data.
- Missing optional dependencies fail with actionable install instructions.
- Documentation does not advertise unimplemented APIs.
- CI runs core tests on supported Python/Spark versions and optional algorithm tests in separate dependency-enabled jobs.

Status (2026-09-30):
- **Met:** the Random Forest workflow in under 20 lines; the same workflow for XGBoost, LightGBM, and CatBoost, on the runtime and cluster described above; actionable install errors; README advertises only implemented APIs.
- **Met for binary and regression only:** clear metrics and stable output schemas. Multiclass metrics are wrong (item 5).
- **Not met:** full pipeline save/load (item 4); dependency-enabled CI jobs, which the Databricks validation notebook currently stands in for.

## Later Goals

- Multiclass classification with explicit averaging options for metrics.
- Multilabel classification.
- Quantile regression and other specialized regression objectives.
- Calibration and threshold tuning for binary classifiers.
- Feature importance and model comparison visualizations.
- MLflow integration for experiment tracking and model registry workflows.
- Distributed hyperparameter tuning with Spark-aware execution.
