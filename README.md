# smallaxe

A PySpark MLOps library that simplifies model training, evaluation, and optimization for PySpark DataFrames.

## Why smallaxe?

PySpark MLlib has a steep learning curve and verbose API. **smallaxe** provides a clean, scikit-learn-like interface for common ML workflows while leveraging the distributed power of Spark.

## Features

- **Simple API** - Train models with familiar `fit()`/`predict()` patterns
- **Multiple Algorithms** - XGBoost, LightGBM, CatBoost, and Random Forest
- **Preprocessing Pipeline** - Imputer, Scaler, Encoder with chainable pipelines
- **Hyperparameter Optimization** - Built-in hyperopt integration with early stopping
- **Automated Training** - Train all algorithms and compare with one call
- **Visualization** - Plotly-based charts for model evaluation
- **Cross-Validation** - Train/test split and k-fold with stratified sampling

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

### Hyperparameter Optimization

```python
from smallaxe.search import optimize
from hyperopt import hp

param_grid = {
    'max_depth': hp.choice('max_depth', [3, 5, 7, 10]),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.3)
}

best_model = optimize.run(
    model=Regressors.xgboost(),
    dataframe=df,
    label_col='target',
    param_grid=param_grid,
    metric='rmse',
    max_evals=50
)

print(best_model.best_params)
```

### Automated Training

```python
from smallaxe.auto import AutomatedTraining

auto = AutomatedTraining(model_type='classification', metrics=['f1_score', 'auc_roc'])
auto.fit(
    df,
    label_col='churn',
    numerical_cols=['tenure', 'monthly_charges'],
    categorical_cols=['contract'],
    n_folds=5
)

# Compare all models
auto.metrics.show()

# Use best model
predictions = auto.predict(new_df)
```

## Supported Algorithms

| Algorithm | Regressor | Classifier | Dependencies |
|-----------|-----------|------------|--------------|
| Random Forest | Yes | Yes | None (native PySpark) |
| XGBoost | Yes | Yes | `smallaxe[xgboost]` |
| LightGBM | Yes | Yes | `smallaxe[lightgbm]` |
| CatBoost | Yes | Yes | `smallaxe[catboost]` |

## Requirements

- Python 3.8, 3.9, or 3.10
- PySpark 3.3, 3.4, or 3.5

## License

MIT License
