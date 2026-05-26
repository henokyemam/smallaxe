# smallaxe

[![CI](https://github.com/henokyemam/smallaxe/actions/workflows/ci.yml/badge.svg)](https://github.com/henokyemam/smallaxe/actions/workflows/ci.yml)

A PySpark MLOps library that simplifies model training, evaluation, and optimization for PySpark DataFrames.

## Why smallaxe?

PySpark MLlib has a steep learning curve and verbose API. **smallaxe** provides a clean, scikit-learn-like interface for common ML workflows while leveraging the distributed power of Spark.

## Features

- **Simple API** - Train models with familiar `fit()`/`predict()` patterns
- **Multiple Algorithms** - XGBoost, LightGBM, CatBoost, and Random Forest
- **Preprocessing Pipeline** - Imputer, Scaler, Encoder with chainable pipelines
- **Cross-Validation** - Train/test split and k-fold with stratified sampling
- **Metrics** - Classification (accuracy, precision, recall, F1, AUC-ROC, AUC-PR, log loss) and regression (MSE, RMSE, MAE, R², MAPE)
- **Visualization** - Plotly-based charts for model evaluation

## Installation

```bash
pip install smallaxe
```

Install with optional algorithm dependencies:

```bash
pip install smallaxe[xgboost]    # XGBoost support
pip install smallaxe[lightgbm]   # LightGBM support
pip install smallaxe[catboost]   # CatBoost support
pip install smallaxe[all]        # All algorithms
```

## Quick Start

```python
from smallaxe.training import Regressors
from smallaxe.datasets import load_sample_regression

# Load sample data
df = load_sample_regression(spark)

# Train a model
model = Regressors.random_forest()
model.fit(df, label_col='price', exclude_cols=['id'])

# Make predictions
predictions = model.predict(df)
```

## Usage Examples

### Training with Cross-Validation

```python
from smallaxe.training import Classifiers

model = Classifiers.xgboost(task='binary')
model.fit(
    df,
    label_col='churn',
    validation='kfold',
    n_folds=5,
    stratified=True
)

print(model.validation_scores)
```

### Preprocessing Pipeline

```python
from smallaxe.pipeline import Pipeline
from smallaxe.preprocessing import Imputer, Scaler, Encoder
from smallaxe.training import Regressors

pipeline = Pipeline([
    ('imputer', Imputer(numerical_strategy='median')),
    ('scaler', Scaler(method='standard')),
    ('encoder', Encoder(method='onehot')),
    ('model', Regressors.xgboost())
])

pipeline.fit(
    df,
    label_col='target',
    numerical_cols=['age', 'income'],
    categorical_cols=['city', 'category']
)

predictions = pipeline.predict(new_df)
```

## Supported Algorithms

| Algorithm | Regressor | Classifier | Dependencies |
|-----------|-----------|------------|--------------|
| Random Forest | ✓ | ✓ | None (native PySpark) |
| XGBoost | ✓ | ✓ | `smallaxe[xgboost]` |
| LightGBM | ✓ | ✓ | `smallaxe[lightgbm]` |
| CatBoost | ✓ | ✓ | `smallaxe[catboost]` |

## Requirements

- Python 3.8 - 3.12
- PySpark 3.3+
- Java 8 or 11 (required by Spark)

## License

MIT License
