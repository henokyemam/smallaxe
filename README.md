# smallaxe

[![CI](https://github.com/henokyemam/smallaxe/actions/workflows/ci.yml/badge.svg)](https://github.com/henokyemam/smallaxe/actions/workflows/ci.yml)

A PySpark MLOps library that simplifies model training, evaluation, and optimization for PySpark DataFrames.

## Why smallaxe?

PySpark MLlib has a steep learning curve and verbose API. **smallaxe** provides a clean, scikit-learn-like interface for common ML workflows while leveraging the distributed power of Spark.

## Features

- **Simple API** - Train models with familiar `fit()`/`predict()` patterns
- **Multiple Algorithms** - Random Forest (native), XGBoost, LightGBM, and CatBoost
- **Preprocessing Pipeline** - Imputer, Scaler, Encoder with chainable pipelines
- **Cross-Validation** - Train/test split and k-fold, reproducible from a seed; each class is split in the same proportion
- **Metrics** - Classification (accuracy, precision, recall, F1, AUC-ROC, AUC-PR, log loss) and regression (MSE, RMSE, MAE, R², MAPE)
- **Hyperparameter Optimization** - hyperopt-backed search over any model's parameters

## Installation

```bash
pip install smallaxe
```

Install with optional algorithm dependencies:

```bash
pip install smallaxe[xgboost]    # XGBoost support
pip install smallaxe[lightgbm]   # LightGBM support (SynapseML)
pip install smallaxe[catboost]   # CatBoost support
pip install smallaxe[viz]        # plotly, for the planned visualizations
pip install smallaxe[all]        # All of the above
```

> **Note on optional algorithms.** Random Forest and XGBoost run anywhere PySpark
> runs. LightGBM (via SynapseML) and CatBoost (via `catboost-spark`) additionally
> require **JVM Spark packages built for your Spark/Scala version** and native
> libraries — they are intended for Linux Spark clusters (e.g. Databricks). See the
> [compatibility matrix](#compatibility-matrix).

## Quick Start

```python
import smallaxe
from smallaxe.training import regressors
from smallaxe.datasets import load_sample_regression

smallaxe.set_seed(42)  # reproducible splits

# Load sample data
dataframe = load_sample_regression()   # uses the active SparkSession

# Train, evaluate, save, load, and predict in a handful of lines.
# Feature columns are inferred from the numeric columns when not specified.
model = regressors.random_forest(n_estimators=100, max_depth=10)
model.fit(dataframe, label_col="price", validation="train_test")
print(model.validation_scores)        # {'rmse': ..., 'r2': ..., ...}

model.save("/tmp/my_model")
reloaded = regressors.load("/tmp/my_model")
predictions = reloaded.predict(dataframe)     # identical to the original model
```

## Usage Examples

### Training with Cross-Validation

```python
from smallaxe.training import classifiers

model = classifiers.xgboost(task="binary")
model.fit(
    dataframe,
    label_col="churn",
    validation="kfold",
    n_folds=5,
)

print(model.validation_scores)   # includes mean_/std_ per metric across folds
```

For `task="multiclass"`, validation `precision`, `recall`, and `f1_score` use macro
averages: each class has equal weight. Scores also include explicit `*_macro` and
`*_weighted` keys, where weighted averages use each class's true-label count.
Undefined per-class scores are zero, and classes include all true and predicted labels.
Label and prediction columns must be numeric; rows where either is null are ignored.

> **Changed since 0.8.x:** multiclass `precision`, `recall`, and `f1_score` in
> `validation_scores` were class 1's one-vs-rest scores, not averages. Models saved by
> 0.8.x keep those old values under the same keys, so loading one warns; refit it to get
> macro and weighted scores. See [CHANGELOG.md](CHANGELOG.md).

Standalone `smallaxe.metrics.precision`, `recall`, and `f1_score` accept
`average="macro"` or `average="weighted"`; their default remains `average="binary"`
(class 1). Use `smallaxe.metrics.multiclass_scores(dataframe, label_col, prediction_col)`
to compute accuracy and all macro/weighted scores together.

### Hyperparameter Optimization

`smallaxe.search.optimize` tunes any model using [hyperopt](https://github.com/hyperopt/hyperopt).
You pass a model template, a search space, and the metric to optimize; it returns
the best params, the best validation score, a refit `best_model`, and the full
trial history.

Multiclass search defaults to `accuracy`. You can select `precision`, `recall`, or
`f1_score` for macro averaging, or an explicit name such as `f1_score_weighted`.

```python
from hyperopt import hp
from smallaxe.search import optimize
from smallaxe.training import regressors

result = optimize.run(
    regressors.random_forest(seed=42),
    dataframe,
    label_col="price",
    param_space={
        "n_estimators": hp.quniform("n_estimators", 40, 200, 20),
        "max_depth": hp.quniform("max_depth", 3, 15, 1),
        # a plain list is treated as discrete choices (hp.choice)
        "feature_subset_strategy": ["sqrt", "log2", "onethird"],
    },
    metric="rmse",            # minimized; r2/accuracy/auc_* are maximized
    validation="kfold",
    n_folds=5,
    max_evals=25,
    seed=42,
)

print(result.best_params, result.best_score)
predictions = result.best_model.predict(dataframe)   # best_model is already fitted
```

### Preprocessing Pipeline

```python
from smallaxe.pipeline import Pipeline
from smallaxe.preprocessing import Imputer, Scaler, Encoder
from smallaxe.training import regressors

pipeline = Pipeline([
    ("imputer", Imputer(numerical_strategy="median")),
    ("scaler", Scaler(method="standard")),
    ("encoder", Encoder(method="onehot")),
    ("model", regressors.xgboost()),
])

pipeline.fit(
    dataframe,
    label_col="target",
    numerical_cols=["age", "income"],
    categorical_cols=["city", "category"],
)

predictions = pipeline.predict(new_df)
```

### Validating a Pipeline

`Pipeline.fit` takes the model's validation options and hands them to the model step:

```python
pipeline.fit(
    dataframe,
    label_col="target",
    numerical_cols=["age", "income"],
    categorical_cols=["city", "category"],
    validation="kfold",
    n_folds=5,
)
print(pipeline.validation_scores)   # the model step's validation_scores
```

The preprocessing steps are fitted on every row before the model is validated, so
their statistics (imputed means, scaling ranges, encoder categories) include the rows
validation holds out. The scores measure the model on that preprocessed data.

## Upgrading to 1.0

smallaxe 1.0 removes every spelling that 0.11 deprecated. Run your code on 0.11 first
and fix each `DeprecationWarning`: it names the replacement, and then the code runs
unchanged on 1.0. The "Upgrading from 0.x" list in [CHANGELOG.md](CHANGELOG.md) names
every removed item and what to use instead. Models and pipelines saved by 0.8.1 and
later still load.

## Stability and versioning

From 1.0, smallaxe follows [semantic versioning](https://semver.org): a patch release
fixes bugs, a minor release adds to the API without breaking it, and only a major
release removes or changes it incompatibly.

The public API is:

- every name listed in the `__all__` of a public module (one whose dotted path has no
  part starting with `_`, such as `smallaxe.training.regressors`; `smallaxe._fs` is
  private), and the public methods and attributes of the classes among them;
- the arguments those functions and methods document, with their defaults and
  behaviour;
- the artifact formats: every artifact written by a 1.x release (models, pipelines,
  `Imputer`, `Scaler`, `Encoder`) loads in every later 1.x release, and artifacts
  written by 0.8.1 and later keep loading. A release that cannot read an artifact
  written by a newer one says so instead of guessing.

Names starting with `_`, the private modules, the exact text of messages and warnings,
and `metadata` keys not listed in a docstring are not part of it. Nor are the
implementation classes `BaseModel`, `BaseRegressor` and `BaseClassifier` in
`smallaxe.training.base` and the five mixins in `smallaxe.training.mixins`: they are
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

Runnable scripts on real Kaggle datasets live in [`examples/`](examples/):

| Example | Task | Dataset |
|---------|------|---------|
| [`titanic_classification.py`](examples/titanic_classification.py) | Binary classification | Kaggle Titanic |
| [`house_prices_regression.py`](examples/house_prices_regression.py) | Regression | Kaggle King County house sales |

Each script covers the full workflow: load → train → evaluate → save/load → optimize → predict.

```bash
python examples/titanic_classification.py
python examples/house_prices_regression.py
```

## Supported Algorithms

| Algorithm | Regressor | Classifier | Dependencies |
|-----------|-----------|------------|--------------|
| Random Forest | ✓ | ✓ | None (native PySpark) |
| XGBoost | ✓ | ✓ | `smallaxe[xgboost]` |
| LightGBM | ✓ | ✓ | `smallaxe[lightgbm]` + SynapseML Spark package |
| CatBoost | ✓ | ✓ | `smallaxe[catboost]` + `catboost-spark` Spark package |

Use `regressors.available_models()` / `classifiers.available_models()` to see which
algorithms are installed in the current environment, with install hints for the rest.

## Compatibility Matrix

| Component | Supported |
|-----------|-----------|
| Python | 3.8 – 3.12 (which PySpark lines, below) |
| PySpark | 3.3 – 3.5 (Spark < 4.0) |
| Java | 11; also 8 with PySpark 3.3 and 17 with PySpark 3.5 |
| OS | Linux |
| LightGBM (SynapseML) | Spark 3.x / **Scala 2.12 only** (no Scala 2.13 build), Linux |
| CatBoost (catboost-spark) | Spark 3.5 / Scala **2.12 or 2.13**, Linux |

CI tests every pair below on Linux with Java 11, plus Java 8 with PySpark 3.3 and
Java 17 with PySpark 3.5 (both on Python 3.10). The pairs are the ones PySpark itself
supports, per the Python classifiers of each PySpark release, plus Python 3.12 with
PySpark 3.5 and Java 17. No PySpark 3.x release declares Python 3.12; that combination
is tested because Databricks Runtime 16.4 LTS ships it. PySpark 3.5 imports
`distutils`, which Python 3.12 removed, so smallaxe installs `setuptools` there. On
Python 3.8 and 3.9 the newest hyperopt is 0.2.7, which imports `pkg_resources`, so
smallaxe installs a `setuptools` older than 82 (the release that removed it) there; from
Python 3.10 it requires hyperopt 0.3.0 or later, which does not need it.

| PySpark | Python |
|---------|--------|
| 3.3 | 3.8 – 3.10 |
| 3.4 | 3.8 – 3.11 |
| 3.5 | 3.8 – 3.12 |

CI also runs the oldest versions smallaxe allows (Python 3.8, PySpark 3.3.0 and the
lowest allowed version of every other runtime dependency), and XGBoost on PySpark 3.3
and 3.5.

> LightGBM and CatBoost depend on JVM packages compiled for a specific
> Spark/Scala build and on native shared libraries, so they run on Linux
> Spark/Databricks clusters rather than locally on Apple Silicon. CatBoost
> publishes both `catboost-spark_3.5_2.12` and `catboost-spark_3.5_2.13`;
> SynapseML (LightGBM) publishes only Scala 2.12 builds for Spark 3.5, so
> LightGBM is unavailable on Scala 2.13 runtimes. Random Forest and XGBoost have
> no such constraints.
>
> **LightGBM** needs SynapseML's Python API on the path. On Databricks the
> `synapseml-lightgbm_2.12` Maven package provides it; `smallaxe[lightgbm]` also
> installs the `synapseml` Python package, whose version should match the jar.
>
> **CatBoost** training fails if executors join or leave mid-fit (for example,
> cluster autoscaling), so train CatBoost on a fixed-size cluster.
>
> Validated end-to-end on Databricks 16.4 LTS (Spark 3.5.2, Scala 2.12) with
> `com.microsoft.azure:synapseml-lightgbm_2.12:1.1.3` and
> `ai.catboost:catboost-spark_3.5_2.12:1.2.10`: all four algorithms in the same
> session, for regression, binary, and multiclass tasks (train, validate,
> predict, save/load, and tune via `search.optimize`). See
> [`examples/databricks_all_algorithms_validation.py`](examples/databricks_all_algorithms_validation.py).

## Development

```bash
# Set up the environment (PySpark needs Java; CI uses 11)
export JAVA_HOME=/path/to/jdk11
pip install -e ".[dev]"        # add ,xgboost / ,all for optional algorithms

# Ensure Spark workers use the same Python as the driver
export PYSPARK_PYTHON=$(which python)
export PYSPARK_DRIVER_PYTHON=$(which python)

pytest -q                      # run the test suite
black . && ruff check .        # format + lint
mypy smallaxe                  # type check (strict; settings in pyproject.toml)
```

The optional-algorithm and end-to-end Kaggle tests skip automatically when their
dependencies or network are unavailable, so the core suite always runs offline.

## Roadmap

Planned for future releases (not yet available):

- `AutomatedTraining` — train all available algorithms and compare them.
- Plotly-based evaluation visualizations.
- Multiclass/multilabel metrics, quantile regression, MLflow integration.

## License

MIT License
