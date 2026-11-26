# smallaxe

[![CI](https://github.com/henokyemam/smallaxe/actions/workflows/ci.yml/badge.svg)](https://github.com/henokyemam/smallaxe/actions/workflows/ci.yml)

smallaxe trains, evaluates, tunes and saves machine learning models on PySpark DataFrames
with a scikit-learn-like API.

## Why smallaxe?

PySpark MLlib is powerful but verbose. smallaxe gives the common supervised workflows one
small interface: `fit`, `predict`, `evaluate`, `save` and `load`. The work still runs in
Spark, so it scales with the cluster.

## Features

- Models with `fit()` / `predict()` / `evaluate()`: Random Forest (native PySpark),
  XGBoost, LightGBM and CatBoost, for regression, binary and multiclass classification.
- Preprocessing: `Imputer`, `Scaler` and `Encoder`, chained with a model in a `Pipeline`
  that saves and loads as one unit.
- Validation: train/test split and k-fold, reproducible from a seed. Each row's split is
  a hash of its content, so every class splits in the same proportion in expectation.
- Metrics: accuracy, precision, recall, F1, AUC-ROC, AUC-PR and log loss for
  classification; MSE, RMSE, MAE, R² and MAPE for regression.
- Hyperparameter search over any model's parameters, backed by hyperopt.
- Automated training: every installed algorithm on the same validation rows, best one
  kept.
- Charts with plotly (confusion matrix, ROC and precision-recall curves, residuals,
  actual vs predicted, feature importances, model comparison). They aggregate in Spark,
  so they work on DataFrames of any size.

## Installation

```bash
pip install smallaxe
```

smallaxe runs on PySpark 3.3 to 3.5 and 4.0 to 4.3, on a classic Spark session. Spark 4
needs Java 17 or later. pip installs the newest PySpark your Python supports; pin it to
stay on Spark 3 (`pip install smallaxe "pyspark~=3.5.0"`). The exact tested combinations
are in the [compatibility matrix](#compatibility-matrix).

Optional algorithms and charts are extras:

```bash
pip install smallaxe[xgboost]    # XGBoost
pip install smallaxe[lightgbm]   # LightGBM (SynapseML)
pip install smallaxe[catboost]   # CatBoost
pip install smallaxe[viz]        # plotly, for the smallaxe.viz charts
pip install smallaxe[all]        # all of the above
```

Random Forest and XGBoost run anywhere PySpark runs. On Spark 4, XGBoost needs xgboost 3.3
or later (`pip install "xgboost>=3.3"`). LightGBM and CatBoost also need a JVM Spark
package built for your Spark and Scala version, plus native libraries, so they are meant
for Linux Spark clusters such as Databricks. See [Optional algorithms on a
cluster](#optional-algorithms-on-a-cluster).

## Quick Start

Train, evaluate, save, load and predict with Random Forest:

```python
import tempfile

from pyspark.sql import SparkSession

import smallaxe
from smallaxe.datasets import load_sample_regression
from smallaxe.training import regressors

spark = SparkSession.builder.getOrCreate()
smallaxe.set_seed(42)  # reproducible splits and trees
houses = load_sample_regression(spark)  # 10,000 rows of synthetic house prices

model = regressors.random_forest(n_estimators=50, max_depth=8)
model.fit(houses, label_col="price", exclude_cols=["location", "condition"],
          validation="train_test")
print(model.validation_scores)  # rmse, mae, mse, r2 and mape on the held-out 20%
path = tempfile.mkdtemp() + "/house_rf"
model.save(path)
predictions = regressors.load(path).predict(houses)  # adds a predict_label column
```

Feature columns are inferred from the numeric and boolean columns when `feature_cols` is
not given. Here the two string columns are excluded; otherwise `fit` warns once that it
ignores them. With `validation="train_test"` the scores come from a 20% holdout, and the
model you keep is then refitted on every row.

## Usage

Every example below runs as written against a local Spark session; the test suite runs
them (`tests/test_readme_examples.py`). Examples marked as needing XGBoost or plotly are
skipped where that package is not installed.

### Classification with k-fold validation

<!-- test: needs xgboost -->
```python
from smallaxe.datasets import load_sample_classification
from smallaxe.training import classifiers

churn = load_sample_classification(n_rows=2000)  # uses the active SparkSession
model = classifiers.xgboost(task="binary", n_estimators=50)
model.fit(
    churn,
    label_col="churn",
    exclude_cols=["contract", "payment_method"],
    validation="kfold",
    n_folds=3,
)
print(model.validation_scores)  # mean_ and std_ of each metric across the folds
probabilities = model.predict_proba(churn)  # adds the probability vector
```

For `task="multiclass"`, the validation `precision`, `recall` and `f1_score` are macro
averages: each class has equal weight. The scores also include explicit `*_macro` and
`*_weighted` keys, where weighted averages use each class's true-label count. Undefined
per-class scores are zero, and the classes are all true and predicted labels. Label and
prediction columns must be numeric; rows where either is null are ignored.

The standalone `smallaxe.metrics.precision`, `recall` and `f1_score` accept
`average="macro"` or `average="weighted"`; their default is `average="binary"` (class 1).
`smallaxe.metrics.multiclass_scores(dataframe, label_col, prediction_col)` computes
accuracy and all macro and weighted scores together.

### Evaluating on new data

`evaluate` scores a fitted model on any DataFrame with the metrics its validation uses:

```python
from smallaxe.datasets import load_sample_classification
from smallaxe.training import classifiers

churn = load_sample_classification(n_rows=2000)
train, test = churn.randomSplit([0.8, 0.2], seed=7)
model = classifiers.random_forest(n_estimators=30)
model.fit(train, label_col="churn", feature_cols=["tenure", "monthly_charges", "total_charges"])
print(model.evaluate(test))  # accuracy, precision, recall, f1_score, auc_roc, auc_pr, log_loss
```

### Hyperparameter search

`smallaxe.search.optimize.run` tunes any model with
[hyperopt](https://github.com/hyperopt/hyperopt). You pass a model template, a search
space and the metric to optimize. It returns the best parameters, the best validation
score, a `best_model` refitted on every row, and the trial history. Every trial scores on
the same split.

Multiclass search defaults to `accuracy`. You can select `precision`, `recall` or
`f1_score` for macro averages, or an explicit name such as `f1_score_weighted`.

```python
from hyperopt import hp

from smallaxe.datasets import load_sample_regression
from smallaxe.search import optimize
from smallaxe.training import regressors

houses = load_sample_regression(n_rows=2000)
result = optimize.run(
    regressors.random_forest(seed=42),
    houses,
    label_col="price",
    param_space={
        "n_estimators": hp.quniform("n_estimators", 20, 60, 20),
        "max_depth": hp.quniform("max_depth", 3, 10, 1),
        # a plain list is a set of discrete choices (hp.choice)
        "feature_subset_strategy": ["sqrt", "log2", "onethird"],
    },
    metric="rmse",  # minimized; r2, accuracy and auc_* are maximized
    exclude_cols=["location", "condition"],
    validation="kfold",
    n_folds=3,
    max_evals=5,
    seed=42,
)
print(result.best_params, result.best_score)
predictions = result.best_model.predict(houses)  # best_model is already fitted
```

`optimize.run` takes a model, not a `Pipeline`. To tune on data with categorical columns,
impute and encode the frame first, as in the last example of the next section.

### Automated training

`smallaxe.auto.AutomatedTraining` trains every installed algorithm on the same data,
scores them all on exactly the same validation rows (one split seed for the whole run),
compares them on one metric, and keeps the best model, refitted on every row. It supports
`"simple_regression"` and `"binary"`. The default metric is `rmse` (minimized) for
regression and `auc_roc` (maximized) for binary, as in `optimize.run`. An algorithm whose
optional package is missing is skipped with a warning naming its install hint. One that
fails while training is reported in the table, and the others go on.

```python
from smallaxe.auto import AutomatedTraining
from smallaxe.datasets import load_sample_regression

houses = load_sample_regression(n_rows=2000)
auto = AutomatedTraining(task="simple_regression", params={"random_forest": {"n_estimators": 30}})
auto.fit(houses, label_col="price", exclude_cols=["location", "condition"])

print(auto.comparison)  # one row per algorithm, best first
print(auto.best_algorithm, auto.best_score)
predictions = auto.predict(houses)  # uses auto.best_model
```

Give an algorithm fixed `params`, or a hyperopt `param_spaces` entry to tune it with
`optimize.run` (`max_evals` trials, on the same split); the others are fitted once.
`seed` makes the whole comparison reproducible:

```python
from hyperopt import hp

from smallaxe.auto import AutomatedTraining
from smallaxe.datasets import load_sample_classification

churn = load_sample_classification(n_rows=2000)
auto = AutomatedTraining(
    task="binary",
    algorithms=["random_forest", "xgboost"],
    metric="auc_roc",
    params={"xgboost": {"n_estimators": 30}},
    param_spaces={"random_forest": {"max_depth": hp.quniform("max_depth", 3, 8, 1)}},
    max_evals=3,
    seed=42,
)
auto.fit(
    churn,
    label_col="churn",
    exclude_cols=["contract", "payment_method"],
    validation="kfold",
    n_folds=3,
)
print(auto.comparison[["algorithm", "status", "auc_roc", "params"]])  # params: what was used
probabilities = auto.predict_proba(churn)
```

`comparison` is a pandas DataFrame with the columns `algorithm`, `status` (`ok`,
`failed` or `skipped`), the compared metric, the task's other validation metrics,
`fit_seconds`, `params` and `error`. A searched algorithm reports the compared metric
only. An undefined score (AUC on a one-class split) is NaN and never wins.
AutomatedTraining has no `save`: keep the winner with `auto.best_model.save(path)` and the
table with `auto.comparison.to_csv(path)`.

Like `fit`, it trains on numeric and boolean features, so impute and encode the other
columns first:

```python
from smallaxe.auto import AutomatedTraining
from smallaxe.datasets import load_sample_classification
from smallaxe.preprocessing import Encoder, Imputer

churn = load_sample_classification(n_rows=2000)
filled = Imputer(numerical_strategy="median").fit_transform(
    churn, numerical_cols=["tenure", "monthly_charges", "total_charges"]
)
encoded = Encoder(method="onehot").fit_transform(
    filled, categorical_cols=["contract", "payment_method"]
)
auto = AutomatedTraining(
    task="binary", algorithms=["random_forest"], params={"random_forest": {"n_estimators": 30}}
)
auto.fit(encoded, label_col="churn")
print(auto.best_score)
```

### Preprocessing pipeline

A `Pipeline` chains preprocessing steps and a model. The model is fitted on exactly
`numerical_cols` plus the columns the `Encoder` writes; other columns pass through
untouched. `save` writes every step's own artifact and `load_pipeline` reads them back:

```python
import tempfile

import smallaxe
from smallaxe.datasets import load_sample_regression
from smallaxe.pipeline import Pipeline
from smallaxe.preprocessing import Encoder, Imputer, Scaler
from smallaxe.training import regressors

houses = load_sample_regression(n_rows=2000)
pipeline = Pipeline([
    ("imputer", Imputer(numerical_strategy="median", categorical_strategy="most_frequent")),
    ("scaler", Scaler(method="standard")),
    ("encoder", Encoder(method="onehot")),
    ("model", regressors.random_forest(n_estimators=30, seed=42)),
])
pipeline.fit(
    houses,
    label_col="price",
    numerical_cols=["bedrooms", "bathrooms", "sqft", "age"],
    categorical_cols=["location", "condition"],
)

path = tempfile.mkdtemp() + "/house_pipeline"
pipeline.save(path)
reloaded = smallaxe.load_pipeline(path)
predictions = reloaded.predict(houses)  # the same predictions as pipeline.predict(houses)
print(reloaded.evaluate(houses))
```

`smallaxe.load(path)` loads any saved artifact (model, pipeline, `Imputer`, `Scaler`,
`Encoder`) by the kind its `metadata.json` records. Artifacts are written through the
Hadoop FileSystem, so local paths, `dbfs:/`, `s3://` and `hdfs://` all work. Nothing is
pickled.

### Validating a pipeline

`Pipeline.fit` takes the model's validation options and hands them to the model step:

```python
from smallaxe.datasets import load_sample_classification
from smallaxe.pipeline import Pipeline
from smallaxe.preprocessing import Encoder, Scaler
from smallaxe.training import classifiers

churn = load_sample_classification(n_rows=2000)
pipeline = Pipeline([
    ("scaler", Scaler()),
    ("encoder", Encoder()),
    ("model", classifiers.random_forest(n_estimators=30, seed=42)),
])
pipeline.fit(
    churn,
    label_col="churn",
    numerical_cols=["tenure", "monthly_charges", "total_charges"],
    categorical_cols=["contract", "payment_method"],
    validation="kfold",
    n_folds=3,
)
print(pipeline.validation_scores)  # the model step's validation_scores
```

The preprocessing steps are fitted on every row before the model is validated, so their
statistics (imputed values, scaling ranges, encoder categories) include the rows that
validation holds out. The scores measure the model on that preprocessed data.

### Visualization

`smallaxe.viz` draws evaluation charts with [plotly](https://plotly.com/python/)
(`pip install smallaxe[viz]`). Every chart aggregates in Spark and collects only the
aggregated rows: a confusion matrix is one `groupBy(label, prediction).count()`, and a
curve, histogram or density is one `groupBy` over score or value bins. What reaches the
driver is bounded by the bins or classes, never by the number of rows. Each function
returns a `plotly.graph_objects.Figure` (call `.show()`, `.write_html(path)` or
`.to_json()` on it) and takes `title=` to replace its default title.

Classification, on the output of `predict` and `predict_proba`:

<!-- test: needs plotly -->
```python
from smallaxe import viz
from smallaxe.datasets import load_sample_classification
from smallaxe.training import classifiers

churn = load_sample_classification(n_rows=2000)
train, test = churn.randomSplit([0.8, 0.2], seed=7)
model = classifiers.random_forest(n_estimators=30)
model.fit(train, label_col="churn", feature_cols=["tenure", "monthly_charges", "total_charges"])

predictions = model.predict(test)  # adds predict_label
viz.confusion_matrix(predictions, label_col="churn", normalize=True).show()

probabilities = model.predict_proba(test)  # adds the probability vector
viz.roc_curve(probabilities, label_col="churn").show()  # exact AUC-ROC in the legend
viz.precision_recall_curve(probabilities, label_col="churn", n_bins=500).show()
```

`confusion_matrix` works for binary and multiclass labels (at most 50 classes). The
curves are for binary 0/1 labels. They accept the `predict_proba` vector or a numeric
column of any numeric type holding the positive-class score. Their legend carries the
exact `smallaxe.metrics.auc_roc` / `auc_pr`, not the binned approximation.

Regression, on the output of `predict`:

<!-- test: needs plotly -->
```python
from smallaxe import viz
from smallaxe.datasets import load_sample_regression
from smallaxe.training import regressors

houses = load_sample_regression(n_rows=2000)
train, test = houses.randomSplit([0.8, 0.2], seed=7)
regressor = regressors.random_forest(n_estimators=30)
regressor.fit(train, label_col="price", exclude_cols=["location", "condition"])

predictions = regressor.predict(test)
viz.residuals(predictions, label_col="price").show()  # histogram of label - prediction
viz.actual_vs_predicted(predictions, label_col="price").show()  # 2-D density with y = x
```

Models, from a fitted model, a `Pipeline`, an `AutomatedTraining` or a mapping of scores:

<!-- test: needs plotly -->
```python
from smallaxe import viz
from smallaxe.auto import AutomatedTraining
from smallaxe.datasets import load_sample_regression
from smallaxe.training import regressors

houses = load_sample_regression(n_rows=2000)
small = regressors.random_forest(n_estimators=10, max_depth=4)
large = regressors.random_forest(n_estimators=40, max_depth=8)
for model in (small, large):
    model.fit(houses, label_col="price", exclude_cols=["location", "condition"],
              validation="train_test")

viz.feature_importances(large, top_n=15).show()
viz.model_comparison(
    {"small": small.validation_scores, "large": large.validation_scores}, metric="rmse"
).show()

auto = AutomatedTraining(
    task="simple_regression", algorithms=["random_forest"],
    params={"random_forest": {"n_estimators": 20}},
)
auto.fit(houses, label_col="price", exclude_cols=["location", "condition"])
viz.model_comparison(auto).show()  # a fitted AutomatedTraining
```

`model_comparison` sorts best first by the metric's direction (`rmse`, `mse`, `mae`,
`mape` and `log_loss` are minimized, the others maximized), highlights the best bar and
names the models whose score is undefined. Rows with a null or NaN label, prediction or
score are left out of every chart. The regression charts also leave out rows with an
infinite value (they cannot be binned) and say how many in the subtitle.

## Upgrading to 1.0

smallaxe 1.0 removes every spelling that 0.11 deprecated, and a few internal names that
0.x exposed. The fastest path: run your code on 0.11 first and fix each
`DeprecationWarning`. Each one names its replacement, and code without warnings on 0.11
runs on 1.0. In 1.0 a removed name raises `ImportError` or `AttributeError`, and a
removed argument raises `TypeError`.

### Renamed and removed names

| 0.x spelling | 1.0 replacement |
|---|---|
| `RandomForestRegressor()`, `XGBoostRegressor()`, `LightGBMRegressor()`, `CatBoostRegressor()` (from `smallaxe.training` or their algorithm modules) | `regressors.random_forest()`, `regressors.xgboost()`, `regressors.lightgbm()`, `regressors.catboost()`, after `from smallaxe.training import regressors` |
| `RandomForestClassifier()`, `XGBoostClassifier()`, `LightGBMClassifier()`, `CatBoostClassifier()` | `classifiers.random_forest(task=...)`, `classifiers.xgboost(task=...)`, `classifiers.lightgbm(task=...)`, `classifiers.catboost(task=...)` |
| `<OldClass>.load(path)` | `regressors.load(path)`, `classifiers.load(path)` or `smallaxe.load_model(path)` |
| `Regressors.<algorithm>(...)`, `Classifiers.<algorithm>(...)`, `Regressors.load`, `Classifiers.load`, `Regressors.available_models()`, `Classifiers.available_models()` (from `smallaxe.training`, `smallaxe.training.regressors` or `smallaxe.training.classifiers`) | the functions of the same name in the `regressors` / `classifiers` modules |
| `Regressors.list_models()`, `Classifiers.list_models()` | `regressors.algorithms()` (every algorithm) and `regressors.available_models()` (which are installed); the same in `classifiers` |
| the `"class_name"` entry of `available_models()` | the entry's key, which is the algorithm name |
| `model.set_param({...})` | `model.set_params(**params)` |
| `df=` on `Model.fit` / `predict` / `evaluate`, `Classifier.predict_proba`, every `Pipeline`, `Imputer`, `Scaler` and `Encoder` method, and every `smallaxe.metrics` function | `dataframe=`, or pass the DataFrame positionally |
| `stratified=` on `Model.fit` and `optimize.run` | delete it; it had no effect |
| `metrics.log_loss(dataframe, label_col, probability_col, eps)` with `eps` positional | `eps=` by keyword |
| `dataset_info(name)` printing by default | it returns the dict and prints nothing; pass `verbose=True` to print |
| `except ValueError` around `dataset_info` | `except ValidationError` |
| `smallaxe.set_cache_strategy()`, `smallaxe.get_cache_strategy()`, `smallaxe.VALID_CACHE_STRATEGIES` | `cache_strategy=` on `fit` |
| `smallaxe.set_spark_session()`, `smallaxe.get_spark_session()` | `dataframe.sparkSession` or `SparkSession.getActiveSession()` |
| `smallaxe.warnings` | `import warnings` |
| `smallaxe.training.BaseModel`, `BaseRegressor`, `BaseClassifier` | `smallaxe.training.Model`, `Regressor`, `Classifier` |
| `smallaxe.training.base.infer_feature_cols`, `warn_ignored_cols`, `check_training_data`, `resolve_split_seed`; `smallaxe.search.optimize.infer_feature_cols`, `warn_ignored_cols`, `check_training_data`, `resolve_split_seed` | `Model.fit` and `optimize.run` do each of these; set the seed with `smallaxe.set_seed()` or the model's `seed` |
| `smallaxe.training.mixins.validation_mixin.hashable_columns`, `split_hash`, `resolve_split_seed` | `fit(validation=..., fold_key=...)`; `smallaxe.set_seed()` or the model's `seed` |
| `smallaxe.training.algorithm.resolve` | `algorithm.get(name).estimator_class(task)` / `algorithm.get(name).model_class(task)` |
| `algorithm.estimator_kwargs` | a factory and `set_params()` |
| `algorithm.legacy_class_name` | the algorithm name and the model's `task` |
| `algorithm.package_version` | `metadata["algorithm_package_version"]` of a fitted model |
| `algorithm.COLUMN_ROLES` | `algorithm.get(name).cols` |
| `algorithm.LEGACY_CLASS_NAMES` | `smallaxe.load_model()`, which reads 0.8.x artifacts |
| `HYPEROPT_AVAILABLE` (in `smallaxe.search` and `smallaxe.search.optimize`), `optimize.HYPEROPT_INSTALL_HINT` | nothing: hyperopt is a required dependency |
| `optimize.MAXIMIZE_METRICS`, `MINIMIZE_METRICS`, `VALID_METRICS`; `MULTICLASS_AVERAGED_METRICS` in `smallaxe.metrics` and `smallaxe.metrics.classification` (not deprecated in 0.11) | `smallaxe.metrics.TASK_METRICS` (`TASK_METRICS["multiclass"]` for the multiclass names) |
| `XGBOOST_AVAILABLE`, `LIGHTGBM_AVAILABLE`, `CATBOOST_AVAILABLE`, `smallaxe.training.catboost.is_catboost_available()` | `smallaxe.training.algorithm.get(name).available()` or `regressors.available_models()` |
| `smallaxe.training.catboost.catboost_install_hint()` | `algorithm.get("catboost").dependency.install_hint` |
| `smallaxe.preprocessing.encoder.VALID_METHODS`, `VALID_HANDLE_RARE`; `smallaxe.preprocessing.scaler.VALID_METHODS` | the values listed in the class docstrings; an invalid value raises `ValidationError` |
| `smallaxe._deprecation` (private) | nothing |

These were public in 0.11 without a deprecation warning and are private in 1.0:
`smallaxe.preprocessing.imputer.VALID_NUMERICAL_STRATEGIES` and
`VALID_CATEGORICAL_STRATEGIES` (listed in the `Imputer` docstring),
`smallaxe.preprocessing.encoder.OTHER` and `UNKNOWN` (the strings `"__OTHER__"` and
`"__UNKNOWN__"`), `smallaxe.pipeline.pipeline.step_kind`, the artifact layout constants of
the preprocessing, pipeline and persistence modules, and the `REGRESSION_TASKS`,
`CLASSIFICATION_TASKS`, `VALID_TASKS`, `VALID_VALIDATION` and `VALID_CACHE_STRATEGIES`
attributes of the model classes. Use `smallaxe.training.algorithm.REGRESSION_TASKS`,
`CLASSIFICATION_TASKS` and `ALL_TASKS` for the task names.

Removing `stratified` moved the `fit` arguments after `n_folds` (`cache_strategy`,
`fold_key`, `refit`) one place left. Pass them by keyword.

### Saved artifacts

`tests/test_legacy_artifacts.py`, `tests/test_pipeline_persistence.py` and
`tests/test_preprocessing_persistence.py` load artifacts written by released versions
(`tests/fixtures/`) and check that they predict or transform as they did when saved:

- Models: Random Forest and XGBoost models saved by 0.8.1, 0.9.0 and 0.10.0, and models in
  the current format (`format_version` 1, which 0.11 and 1.0 write), load with
  `regressors.load`, `classifiers.load`, `smallaxe.load_model` and `smallaxe.load`. The
  fixtures cover Random Forest and XGBoost only. Re-saving an older model writes the
  current format.
- XGBoost models saved before 0.11 keep the 100 boosting rounds they were trained with;
  those releases trained 100 rounds whatever `n_estimators` said. Refit them to get the
  number you ask for.
- Multiclass models whose saved scores predate the macro averages (0.8.x) load with a
  `UserWarning`: their `precision`, `recall` and `f1_score` are class 1's scores. Refit to
  get macro and weighted scores.
- Preprocessing: an `Imputer` and a `Scaler` saved by 0.10 load and transform as before.
  0.10's `Encoder` had no `save`.
- Pipelines: a pipeline saved by 0.10 with `Imputer`, `Scaler` and `Encoder` steps loads.
  Its pickled `Encoder` is the one pickle smallaxe still reads, and it now maps a
  category it did not see at fit to `__UNKNOWN__` instead of null. A 0.10 pipeline with a
  model step does not load (0.10 could not load it either), and any other pickled step
  raises `ValidationError`. Pipelines in the current format load. Pipelines saved before
  0.10 are not covered by a test.

### Behaviour you will notice

These are the 1.0 changes most likely to affect working 0.11 code. The full list is in
"Changed" and "Fixed" under 1.0.0 in [CHANGELOG.md](CHANGELOG.md).

- User errors raise smallaxe exceptions. Code that caught `AttributeError`, `TypeError`,
  `KeyError` or Spark's `IllegalArgumentException` / `AnalysisException` for a bad
  argument now gets `ValidationError` or `ColumnNotFoundError`. Examples: a value that is
  not a DataFrame, a column list given as a string, a non-numeric label, a feature column
  missing at predict time, and `pipeline["name"]` for a missing step.
- A save or load path must be a string. A `pathlib.Path` raises `ValidationError`; pass
  `str(path)`.
- `set_params` checks every value before it sets any, rejects a bool for an int or float
  parameter, and requires `seed` to be an int or None. `set_seed(True)` raises
  `ConfigurationError`.
- `fit`, `optimize.run` and `AutomatedTraining.fit` on a DataFrame with no rows raise
  `ValidationError`.
- `plotly` is no longer installed with smallaxe. Install `smallaxe[viz]` for the charts.
- A Spark Connect session raises `ConfigurationError` when smallaxe saves or loads.
- `mape` and `log_loss` on integer columns no longer overflow near the integer limits.
  Results do not depend on Spark's ANSI SQL mode, which Spark 4 turns on by default.

## Stability and versioning

From 1.0, smallaxe follows [semantic versioning](https://semver.org): a patch release
fixes bugs, a minor release adds to the API without breaking it, and only a major release
removes or changes it incompatibly.

The public API is:

- every name listed in the `__all__` of a public module (one whose dotted path has no
  part starting with `_`, such as `smallaxe.training.regressors`; `smallaxe._fs` is
  private), and the public methods and attributes of the classes among them;
- the arguments those functions and methods document, with their defaults and behaviour;
- the artifact formats: every artifact written by a 1.x release (models, pipelines,
  `Imputer`, `Scaler`, `Encoder`) loads in every later 1.x release, and the older
  artifacts listed under [Saved artifacts](#saved-artifacts) keep loading. A release
  that cannot read an artifact written by a newer one says so instead of guessing.

Names starting with `_`, the private modules, the exact text of messages and warnings,
and `metadata` keys not listed in a docstring are not part of it. Nor are the
implementation classes `BaseModel`, `BaseRegressor` and `BaseClassifier` in
`smallaxe.training.base` and the five mixins in `smallaxe.training.mixins`. They are
listed in their modules' `__all__` so that type hints and `isinstance` checks can name
them, but what they provide is public only as members of `Model`, `Regressor` and
`Classifier`. Build on `Model`, not on them.

Before anything public is removed or changed incompatibly, it is deprecated: it keeps
working and emits a `DeprecationWarning` naming its replacement for at least one minor
release, and it is removed only in the next major release. `tests/test_public_api.py`
pins the public surface, so an accidental change fails CI.

The supported Python, PySpark and Java versions are exactly the ones CI tests, listed in
the [compatibility matrix](#compatibility-matrix). Dropping one is announced in the
CHANGELOG and happens only in a minor or major release.

## End-to-End Examples

Scripts on real Kaggle datasets live in [`examples/`](examples/). They download their data
on first run:

| Example | Task | Dataset |
|---------|------|---------|
| [`titanic_classification.py`](examples/titanic_classification.py) | Binary classification | Kaggle Titanic |
| [`house_prices_regression.py`](examples/house_prices_regression.py) | Regression | Kaggle King County house sales |

Each script covers load, train, evaluate, save and load, search, and predict.

```bash
python examples/titanic_classification.py
python examples/house_prices_regression.py
```

Two Databricks notebooks (source format) are there too:
[`databricks_validation.py`](examples/databricks_validation.py), a short check of each
algorithm, and
[`databricks_all_algorithms_validation.py`](examples/databricks_all_algorithms_validation.py),
the full validation described under [Optional algorithms on a
cluster](#optional-algorithms-on-a-cluster).

## Supported Algorithms

| Algorithm | Regressor | Classifier | Dependencies |
|-----------|-----------|------------|--------------|
| Random Forest | ✓ | ✓ | None (native PySpark) |
| XGBoost | ✓ | ✓ | `smallaxe[xgboost]` |
| LightGBM | ✓ | ✓ | `smallaxe[lightgbm]` + SynapseML Spark package |
| CatBoost | ✓ | ✓ | `smallaxe[catboost]` + `catboost-spark` Spark package |

`regressors.available_models()` and `classifiers.available_models()` show which
algorithms are installed in the current environment, with install hints for the rest:

```python
from smallaxe.training import classifiers, regressors

print(regressors.algorithms())  # ['random_forest', 'xgboost', 'lightgbm', 'catboost']
for name, entry in classifiers.available_models().items():
    print(name, entry["available"], entry["install_hint"])
```

The factory of an algorithm that is not installed raises `DependencyError` with the
install command.

## Compatibility Matrix

| Component | Spark 3 | Spark 4 |
|-----------|---------|---------|
| PySpark | 3.3 – 3.5 | 4.0 – 4.3 |
| Python | 3.8 – 3.12 (by PySpark line, below) | 3.9 – 3.12 (by PySpark line, below) |
| Java | 11; also 8 with PySpark 3.3 and 17 with PySpark 3.5 | 17; also 25 with PySpark 4.3 |
| Session | classic | classic only (no Spark Connect) |
| Random Forest | ✓ in CI | ✓ in CI |
| XGBoost | ✓ in CI on PySpark 3.3 and 3.5 | ✓ in CI on PySpark 4.3 (xgboost 3.3 or later); validated on Databricks, Spark 4.0 |
| LightGBM (SynapseML) | validated on Databricks, Spark 3.5, Scala 2.12 | validated on Databricks, Spark 4.0, Scala 2.13 |
| CatBoost (catboost-spark) | validated on Databricks, Spark 3.5, Scala 2.12 | validated on Databricks, Spark 4.0, Scala 2.13 |
| OS | Linux | Linux |

The `test` job in `.github/workflows/ci.yml` runs the whole suite on Linux for each row
below, 24 legs in all. The pairs are the ones PySpark itself supports, per the Python
classifiers of each PySpark release, within smallaxe's Python 3.8 to 3.12. One more pair
is tested: Python 3.12 with PySpark 3.5 and Java 17. No PySpark 3.x release declares
Python 3.12; it is tested because Databricks Runtime 16.4 LTS ships it.

| PySpark | Python | Java | `test` legs |
|---------|--------|------|-------------|
| 3.3 | 3.8 – 3.10 | 8, 11 | 3.8, 3.9 and 3.10 on Java 11; 3.10 on Java 8 |
| 3.4 | 3.8 – 3.11 | 11 | 3.8, 3.9, 3.10 and 3.11 on Java 11 |
| 3.5 | 3.8 – 3.12 | 11, 17 | 3.8, 3.9, 3.10 and 3.11 on Java 11; 3.10 and 3.12 on Java 17 |
| 4.0 | 3.9 – 3.12 | 17 | 3.9, 3.10, 3.11 and 3.12 on Java 17 |
| 4.1 | 3.12 | 17 | 3.12 on Java 17 |
| 4.2 | 3.12 | 17 | 3.12 on Java 17 |
| 4.3 | 3.11 – 3.12 | 17, 25 | 3.11 and 3.12 on Java 17; 3.12 on Java 25; 3.12 on Java 17 with pandas 3 |

Two more jobs run the suite:

- `test-lowest`: Python 3.8, PySpark 3.3.0, Java 11, and every other runtime dependency
  at the lowest version `pyproject.toml` allows.
- `test-xgboost`: the suite with the `xgboost` extra installed, on PySpark 3.3 and 3.5
  (Python 3.10, Java 11) and PySpark 4.3 (Python 3.12, Java 17).

Notes on the matrix:

- PySpark 4.0 needs Python 3.9 or later, 4.1 and 4.2 need Python 3.10, and 4.3 needs
  Python 3.11. All of them need Java 17 or later. pip installs an older PySpark on an
  older Python. On PySpark 4.1 and 4.2 only Python 3.12 is tested; PySpark supports 3.10
  and 3.11 there too.
- smallaxe declares Python up to 3.12, so Python 3.13 and 3.14, which PySpark 4
  supports, are not covered.
- Spark 4.0 to 4.3 document pandas below 3. The Spark 4 legs install pandas below 3,
  except one PySpark 4.3 / Python 3.12 leg that installs pandas 3. A plain `pip install`
  resolves pandas 3 on Python 3.11 and later.
- PySpark 3.5 imports `distutils`, which Python 3.12 removed, so smallaxe requires
  `setuptools` on Python 3.12. On Python 3.8 and 3.9 the newest hyperopt is 0.2.7, which
  imports `pkg_resources`, so smallaxe requires a `setuptools` older than 82 (the release
  that removed it) there. From Python 3.10 it requires hyperopt 0.3.0 or later, which
  does not need it.
- Classic sessions only. smallaxe reads and writes artifacts through the Hadoop
  FileSystem on the JVM, so it needs a classic Spark session: the default of
  `pip install pyspark` on every supported release. With a Spark Connect session
  (`SparkSession.builder.remote(...)`, `spark.api.mode=connect`, or the
  `pyspark-connect` / `pyspark-client` packages), saving or loading raises
  `ConfigurationError("smallaxe needs a classic Spark session; Spark Connect is not
  supported")`.
- ANSI SQL mode is on by default from Spark 4 (`spark.sql.ansi.enabled`). smallaxe never
  changes the setting, and its results are the same with it on or off: the ones Spark 3's
  default mode (ANSI off) gives. Where ANSI mode would change a cast or an operator (a
  string compared with a number, `"7.0"` read as an integer, integer overflow, an integer
  subtracted from a float), smallaxe spells out the default mode's cast.
  `tests/test_ansi_mode.py` checks the metric, preprocessing, chart, split and dataset
  edge cases both ways, and on Spark 3 against Spark's own implicit casts.
- XGBoost on Spark 4 needs xgboost 3.3 or later, the first release that declares PySpark 4
  support; it needs Python 3.12 or later. `smallaxe[xgboost]` keeps its floor (1.7.0) for
  Spark 3.3, so on Spark 4 install `xgboost>=3.3`.

### Optional algorithms on a cluster

LightGBM and CatBoost depend on JVM packages compiled for one Spark and Scala build and on
native libraries, so they run on Linux Spark clusters, not locally on Apple Silicon. CI
does not run them; they are validated on Databricks with
[`examples/databricks_all_algorithms_validation.py`](examples/databricks_all_algorithms_validation.py).
Random Forest and XGBoost have no such constraint.

| Runtime | LightGBM Maven package | CatBoost Maven package |
|---|---|---|
| Spark 3.5, Scala 2.12 (Databricks Runtime 16.4 LTS) | `com.microsoft.azure:synapseml-lightgbm_2.12:1.1.3` | `ai.catboost:catboost-spark_3.5_2.12:1.2.10` |
| Spark 4.0, Scala 2.13 (Databricks Runtime 17.3 LTS) | `com.microsoft.azure:synapseml-lightgbm_2.13:1.1.3-spark4.0`, from the `https://mmlspark.azureedge.net/maven` repository | `ai.catboost:catboost-spark_4.0_2.13:1.2.10` |

- SynapseML publishes only Scala 2.12 builds for Spark 3.5, so LightGBM is not available
  on Spark 3 runtimes with Scala 2.13. CatBoost publishes both `catboost-spark_3.5_2.12`
  and `catboost-spark_3.5_2.13`; only the 2.12 build is validated.
- SynapseML also publishes a `-spark4.1` build and catboost-spark a
  `catboost-spark_4.1_2.13` build. Neither is validated with smallaxe, and neither
  project publishes a build for Spark 4.2 or 4.3.
- LightGBM needs SynapseML's Python API on the path. On Databricks the
  `synapseml-lightgbm` Maven package provides it. `smallaxe[lightgbm]` also installs the
  `synapseml` Python package, whose version should match the jar.
- The validation runs used a cluster with 2 fixed-size workers. Autoscaling clusters are
  not validated, so train LightGBM and CatBoost on a fixed-size cluster.

Validation runs on Databricks, both on a cluster with 2 fixed m5d.8xlarge workers:

- 2026-11-26, Databricks Runtime 16.4 LTS (Spark 3.5.2, Scala 2.12, Python 3.12.3), with
  `synapseml-lightgbm_2.12:1.1.3` and `catboost-spark_3.5_2.12:1.2.10`, the 1.0 API
  (commit e41b8eb): 122 of 122 checks passed for Random Forest, XGBoost, LightGBM and
  CatBoost on regression, binary and multiclass, including Pipeline save and load on
  `dbfs:/`.
- 2026-11-26, Databricks Runtime 17.3 LTS (Spark 4.0.0, Scala 2.13.16, Python 3.12.3),
  with `synapseml-lightgbm_2.13:1.1.3-spark4.0`, `catboost-spark_4.0_2.13:1.2.10` and
  xgboost 3.4.1 (commit a7c8ce7): all 124 library checks passed for the four algorithms,
  including AutomatedTraining ranking all four on binary and regression. A separate run
  built seven charts per algorithm and `model_comparison` from an AutomatedTraining.
- Release candidate, Databricks Runtime 16.4 LTS (Spark 3.5), commit [RC COMMIT]: [RC
  RESULT].

## Development

```bash
# PySpark needs Java: 11 for Spark 3 (as in most of CI), 17 or later for Spark 4
export JAVA_HOME=/path/to/jdk11
pip install -e ".[dev]"        # add ,xgboost or ,all for the optional algorithms

# Spark workers must use the same Python as the driver
export PYSPARK_PYTHON=$(which python)
export PYSPARK_DRIVER_PYTHON=$(which python)

pytest -q                                  # the test suite
black --check smallaxe tests               # formatting, as CI checks it
ruff check smallaxe tests examples         # lint
mypy smallaxe                              # type check (strict; settings in pyproject.toml)
```

The optional-algorithm tests skip when their package is not installed, and the Kaggle
end-to-end tests skip without network access, so the core suite runs offline.

## Roadmap

Not in 1.0:

- Hyperparameter search over a whole `Pipeline` (`optimize.run` takes a model).
- Passing estimator parameters that are not in an algorithm's parameter table
  (`extra_params`).
- Native CatBoost categorical features (CatBoost needs an `Encoder` today).
- Multilabel classification and quantile regression.
- MLflow integration.
- Spark Connect sessions.
- Multiclass tasks in `AutomatedTraining`.
- A threshold curve (precision, recall and F1 against the threshold) and multiclass ROC
  curves in `smallaxe.viz`.

## License

MIT License
