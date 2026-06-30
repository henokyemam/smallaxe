# smallaxe

[![CI](https://github.com/henokyemam/smallaxe/actions/workflows/ci.yml/badge.svg)](https://github.com/henokyemam/smallaxe/actions/workflows/ci.yml)

A PySpark MLOps library that simplifies model training, evaluation, and optimization for PySpark DataFrames.

## Why smallaxe?

PySpark MLlib has a steep learning curve and verbose API. **smallaxe** provides a clean, scikit-learn-like interface for common ML workflows while leveraging the distributed power of Spark.

## Features

- **Simple API** - Train models with familiar `fit()`/`predict()` patterns
- **Multiple Algorithms** - Random Forest (native), XGBoost, LightGBM, and CatBoost
- **Preprocessing Pipeline** - Imputer, Scaler, Encoder with chainable pipelines
- **Cross-Validation** - Train/test split and k-fold with stratified sampling
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
pip install smallaxe[all]        # All algorithms
```

> **Note on optional algorithms.** Random Forest and XGBoost run anywhere PySpark
> runs. LightGBM (via SynapseML) and CatBoost (via `catboost-spark`) additionally
> require **JVM Spark packages built for your Spark/Scala version** and native
> libraries — they are intended for Linux Spark clusters (e.g. Databricks). See the
> [compatibility matrix](#compatibility-matrix).

## Quick Start

```python
import smallaxe
from smallaxe.training import Regressors
from smallaxe.datasets import load_sample_regression

smallaxe.set_seed(42)  # reproducible splits

# Load sample data
df = load_sample_regression(spark)

# Train, evaluate, save, load, and predict in a handful of lines.
# Feature columns are inferred from the numeric columns when not specified.
model = Regressors.random_forest(n_estimators=100, max_depth=10)
model.fit(df, label_col="price", validation="train_test")
print(model.validation_scores)        # {'rmse': ..., 'r2': ..., ...}

model.save("/tmp/my_model")
reloaded = Regressors.load("/tmp/my_model")
predictions = reloaded.predict(df)     # identical to the original model
```

## Usage Examples

### Training with Cross-Validation

```python
from smallaxe.training import Classifiers

model = Classifiers.xgboost(task="binary")
model.fit(
    df,
    label_col="churn",
    validation="kfold",
    n_folds=5,
    stratified=True,
)

print(model.validation_scores)   # includes mean_/std_ per metric across folds
```

### Hyperparameter Optimization

`smallaxe.search.optimize` tunes any model using [hyperopt](https://github.com/hyperopt/hyperopt).
You pass a model template, a search space, and the metric to optimize; it returns
the best params, the best validation score, a refit `best_model`, and the full
trial history.

```python
from hyperopt import hp
from smallaxe.search import optimize
from smallaxe.training import Regressors

result = optimize.run(
    Regressors.random_forest(seed=42),
    df,
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
predictions = result.best_model.predict(df)   # best_model is already fitted
```

### Preprocessing Pipeline

```python
from smallaxe.pipeline import Pipeline
from smallaxe.preprocessing import Imputer, Scaler, Encoder
from smallaxe.training import Regressors

pipeline = Pipeline([
    ("imputer", Imputer(numerical_strategy="median")),
    ("scaler", Scaler(method="standard")),
    ("encoder", Encoder(method="onehot")),
    ("model", Regressors.xgboost()),
])

pipeline.fit(
    df,
    label_col="target",
    numerical_cols=["age", "income"],
    categorical_cols=["city", "category"],
)

predictions = pipeline.predict(new_df)
```

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

Use `Regressors.available_models()` / `Classifiers.available_models()` to see which
algorithms are installed in the current environment, with install hints for the rest.

## Compatibility Matrix

| Component | Supported |
|-----------|-----------|
| Python | 3.8 – 3.12 |
| PySpark | 3.3 – 3.5 (Spark < 4.0) |
| Java | 8 or 11 (required by Spark) |
| LightGBM (SynapseML) | Spark 3.x / **Scala 2.12 only** (no Scala 2.13 build), Linux |
| CatBoost (catboost-spark) | Spark 3.5 / Scala **2.12 or 2.13**, Linux |

> LightGBM and CatBoost depend on JVM packages compiled for a specific
> Spark/Scala build and on native shared libraries, so they run on Linux
> Spark/Databricks clusters rather than locally on Apple Silicon. CatBoost
> publishes both `catboost-spark_3.5_2.12` and `catboost-spark_3.5_2.13`;
> SynapseML (LightGBM) publishes only `synapseml_2.12`, so LightGBM is
> unavailable on Scala 2.13 runtimes. Neither supports Spark 4.0 yet. Random
> Forest and XGBoost have no such constraints.
>
> Validated on Databricks (DBR 16.4, Spark 3.5.2): Random Forest, XGBoost, and
> CatBoost train/evaluate/predict and tune via `search.optimize`. See
> [`examples/databricks_validation.py`](examples/databricks_validation.py).

## Development

```bash
# Set up the environment (PySpark needs Java 8/11)
export JAVA_HOME=/path/to/jdk11
pip install -e ".[dev]"        # add ,xgboost / ,all for optional algorithms

# Ensure Spark workers use the same Python as the driver
export PYSPARK_PYTHON=$(which python)
export PYSPARK_DRIVER_PYTHON=$(which python)

pytest -q                      # run the test suite
black . && ruff check .        # format + lint
mypy smallaxe/                 # type check
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
