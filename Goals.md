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
- Optional XGBoost and LightGBM wrappers.
- Imputer, Scaler, Encoder, and Pipeline classes.
- Model metadata, validation scores, feature importance, and save/load support for individual models.
- A substantial test suite. With `~/Desktop/basic`, PySpark 3.5.x, and OpenJDK 11, the current suite passes: 485 passed, 102 skipped. The skipped tests are optional XGBoost/LightGBM coverage when those libraries are not installed.

## Missing For v1

### 1. Align Public API With Actual Capabilities

- Update README to describe only implemented APIs, or implement the advertised APIs before release.
- Current README advertises `smallaxe.search.optimize`, `smallaxe.auto.AutomatedTraining`, visualization, and CatBoost, but those modules are empty or missing.
- Decide whether the first regression API is called "regression" or "linear regression." Random Forest, XGBoost, LightGBM, and CatBoost are not linear models. If true linear regression is a first-class goal, add a Spark `LinearRegression` baseline separately.

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

### 8. Automated Training

- Implement `AutomatedTraining` after the four algorithm wrappers are stable.
- It should:
  - Train all available compatible algorithms.
  - Skip missing optional dependencies with warnings and install hints.
  - Return a comparison table as a Spark or pandas DataFrame.
  - Select `best_model` by a user-specified metric.
  - Persist the winning model or full comparison run.
- Keep the first version constrained to binary classification and continuous regression.

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

## v1 Acceptance Criteria

- A new user can train, evaluate, save, load, and predict with Random Forest on a PySpark DataFrame in under 20 lines of code.
- The same user-facing workflow works for XGBoost, LightGBM, and CatBoost when optional dependencies are installed.
- Binary classification and continuous regression have clear metrics and stable output schemas.
- A full preprocessing-plus-model pipeline can be saved and loaded with identical predictions on deterministic data.
- Missing optional dependencies fail with actionable install instructions.
- Documentation does not advertise unimplemented APIs.
- CI runs core tests on supported Python/Spark versions and optional algorithm tests in separate dependency-enabled jobs.

## Later Goals

- Multiclass classification with explicit averaging options for metrics.
- Multilabel classification.
- Quantile regression and other specialized regression objectives.
- Calibration and threshold tuning for binary classifiers.
- Feature importance and model comparison visualizations.
- MLflow integration for experiment tracking and model registry workflows.
- Distributed hyperparameter tuning with Spark-aware execution.
