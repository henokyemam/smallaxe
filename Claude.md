## Definition and Objective

smallaxe is a pyspark MLOPs library designed to streamline model training and optimization. The reason for such a need is the pyspark MLlib API is not user friendly and hence requires steep-learning curve. smallaxe aims to simplify model training, evaluation and model optimization for pyspark data frames. It will initially do so for two ML training types: simple linear regression and binary classifiers. It will do this for 4 learning algorithms: catboost, xgboost, random forest and light GBM.

## Platform
The library will be installable from pip as 'pip install smallaxe' and it could be run on databricks or any other IDE or even the terminal.

## Dependencies
Core dependencies: pyspark, hyperopt, plotly

Optional dependencies (installed separately based on algorithm needs):
- `pip install smallaxe[xgboost]` - includes xgboost, xgboost-spark
- `pip install smallaxe[lightgbm]` - includes lightgbm, mmlspark/synapse.ml
- `pip install smallaxe[catboost]` - includes catboost-spark
- `pip install smallaxe[all]` - includes all algorithm dependencies

Random Forest uses native PySpark MLlib and requires no additional dependencies.

## Library Features

### 1. Model Training

```python
from smallaxe.training import Regressors, Classifiers

# Specify task type for future extensibility
model = Regressors.xgboost(task='simple_regression')
# task options for Regressors: 'simple_regression', 'quantile_regression' (future)
# task options for Classifiers: 'binary', 'multiclass' (future), 'multilabel' (future)

# View parameter definitions
model.params
# -> returns dictionary of {'param_name': 'definition/description'}

# View default parameter values
model.default_params
# -> returns dictionary of {'param_name': default_value}

# Set parameters (with validation)
model.set_param({'param_name': value})
# -> validates and sets parameters, raises error if value not allowed

# Check task type
model.task_type  # -> 'simple_regression'

# Train the model
model.fit(
    dataframe,
    label_col='label',           # specify target column name
    feature_cols=None,           # None = all other columns, or list of column names
    exclude_cols=['id', 'timestamp']  # columns to exclude from features (useful when feature_cols=None)
)

# Make predictions
model.predict(dataframe)
# -> returns dataframe with additional 'predict_label' column
# -> if label_col exists in dataframe, it is ignored during prediction

# Access model metadata (available after fit)
model.metadata
# -> returns dict with:
#    {
#        'training_timestamp': '2024-01-15T10:30:00',
#        'spark_version': '3.5.0',
#        'smallaxe_version': '1.0.0',
#        'dataset_schema_hash': 'abc123...',
#        'training_metrics': {'rmse': 0.15, 'r2': 0.92},
#        'row_count': 10000,
#        'feature_cols': ['age', 'income', ...],
#        'label_col': 'target'
#    }
```

**Task Types (v1.0 and future):**
| Type | Regressors | Classifiers |
|------|------------|-------------|
| v1.0 | `simple_regression` | `binary` |
| Future | `quantile_regression` | `multiclass`, `multilabel` |

### 2. Model Persistence

```python
# Save trained model
model.save('/path/to/model')

# Load model
from smallaxe.training import Regressors
model = Regressors.load('/path/to/model')
```

### 3. Preprocessing Pipeline

```python
from smallaxe.pipeline import Pipeline
from smallaxe.preprocessing import Imputer, Scaler, Encoder

# Define preprocessing pipeline with model
pipeline = Pipeline([
    ('imputer', Imputer(
        numerical_strategy='mean',      # 'mean', 'median', or custom value (e.g., -999)
        categorical_strategy='mode'     # 'mode' or custom value (e.g., 'UNKNOWN')
    )),
    ('scaler', Scaler(method='standard')),  # 'standard', 'minmax', 'maxabs'
    ('encoder', Encoder(
        method='onehot',        # 'onehot', 'label'
        max_categories=100,     # limit categories to prevent dimension explosion
        handle_rare='other'     # 'other' (group rare), 'keep', or 'error'
    )),
    ('model', Regressors.xgboost())
])

pipeline.fit(
    dataframe,
    label_col='target',
    numerical_cols=['age', 'income', 'score'],
    categorical_cols=['gender', 'city', 'category']
)

pipeline.predict(dataframe)
pipeline.save('/path/to/pipeline')

# Preprocessing-only pipeline (no model)
preprocessing_pipeline = Pipeline([
    ('imputer', Imputer(numerical_strategy='median', categorical_strategy='UNKNOWN')),
    ('scaler', Scaler(method='minmax')),
    ('encoder', Encoder(method='label'))
])

preprocessing_pipeline.fit(
    dataframe,
    numerical_cols=['age', 'income'],
    categorical_cols=['gender', 'city']
)

# Transform data without training a model
df_transformed = preprocessing_pipeline.transform(dataframe)
```

**Preprocessing Requirements by Algorithm:**
- **Random Forest**: Requires scaling and encoding (enforced)
- **XGBoost**: Encoding required, scaling optional
- **LightGBM**: Encoding required, scaling optional
- **CatBoost**: No preprocessing required (handles categoricals natively)

If required preprocessing steps are missing for an algorithm, the library will raise an informative error.

### 4. Model Metrics

```python
from smallaxe.metrics import (
    # Regression metrics
    rmse, mae, mse, r2, mape,
    # Classification metrics
    f1_score, accuracy, precision, recall, auc_roc, auc_pr, log_loss
)

# Usage: dataframe must have 'label' and 'predict_label' columns
score = f1_score(dataframe, label_col='label', prediction_col='predict_label')
```

### 5. Cross-Validation

```python
# For manual model training - user chooses validation strategy
model.fit(
    dataframe,
    label_col='target',
    validation='train_test',    # 'train_test', 'kfold', or None
    test_size=0.2,              # for train_test split
    n_folds=5,                  # for kfold
    stratified=True             # stratified sampling for classification (default: True)
)

# Access validation results
model.validation_scores  # -> dict of metric scores from validation
```

**Stratified Sampling:**
- For classification tasks, `stratified=True` (default) ensures class distribution is preserved in splits
- For regression tasks, `stratified` is ignored
- Works with both `train_test` and `kfold` validation strategies

### 6. Model Optimization

```python
from smallaxe.search import optimize

# Define parameter grid for hyperopt
param_grid = {
    'max_depth': hp.choice('max_depth', [3, 5, 7, 10]),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
    'n_estimators': hp.quniform('n_estimators', 50, 500, 50)
}

# Run optimization (displays progress bar)
best_model = optimize.run(
    model=Regressors.xgboost(),
    dataframe=df,
    label_col='target',
    param_grid=param_grid,
    metric='rmse',              # metric to optimize
    max_evals=100,              # maximum number of trials
    validation='kfold',         # validation strategy
    n_folds=5,
    early_stopping=True,        # enable early stopping (default: True)
    early_stopping_rounds=20    # stop if no improvement after N trials (default: 20)
)

# best_model is a trained model with optimal hyperparameters
best_model.best_params    # -> dict of optimal parameters found
best_model.trials_history # -> list of all trials with params and scores
best_model.best_score     # -> best metric score achieved
```

**Early Stopping:**
- When `early_stopping=True`, optimization stops if no improvement is found after `early_stopping_rounds` consecutive trials
- Saves compute time on large search spaces
- Progress bar shows current best score and trials remaining

### 7. Automated Training

```python
from smallaxe.auto import AutomatedTraining

auto = AutomatedTraining(
    model_type='regression',    # 'regression' or 'classification'
    metrics=['rmse', 'mae']     # str or list of metrics to evaluate
)

# Trains all 4 algorithms with default hyperparams using KFold CV
auto.fit(
    dataframe,
    label_col='target',
    numerical_cols=['age', 'income'],
    categorical_cols=['gender', 'city'],
    n_folds=5                   # KFold cross-validation (required for AutomatedTraining)
)

# Make predictions with best performing model
auto.predict(dataframe)

# View comparison metrics
auto.metrics
# -> returns PySpark DataFrame with columns:
#    ['model_name', 'rmse', 'mae', ...]

# Access individual trained models
auto.models  # -> dict of {'model_name': trained_model}
auto.best_model  # -> best performing model based on first metric
```

### 8. Visualization

```python
from smallaxe.viz import Visualizer

viz = Visualizer(theme='light')  # 'light' or 'dark'

# --- Classification Visualizations ---

# Confusion matrix heatmap
viz.confusion_matrix(
    df,
    label_col='label',
    prediction_col='predict_label'
).show()

# ROC curve with AUC score
viz.roc_curve(
    df,
    label_col='label',
    probability_col='probability'  # model must output probabilities
).show()

# Precision-Recall curve
viz.precision_recall_curve(
    df,
    label_col='label',
    probability_col='probability'
).show()

# Threshold analysis curve (precision, recall, f1 vs threshold)
viz.threshold_curve(
    df,
    label_col='label',
    probability_col='probability'
).show()
# -> helps find optimal classification threshold for your use case

# --- Regression Visualizations ---

# Actual vs Predicted scatter plot
viz.actual_vs_predicted(
    df,
    label_col='label',
    prediction_col='predict_label'
).show()

# Residuals plot (residuals vs predicted values)
viz.residuals(
    df,
    label_col='label',
    prediction_col='predict_label'
).show()

# --- Feature Analysis ---

# Feature importance (horizontal bar chart)
viz.feature_importance(model).show()

# --- Model Comparison (for AutomatedTraining) ---

# Compare metrics across all trained models (grouped bar chart)
viz.compare_models(auto.metrics).show()

# --- Output Options ---

# Display inline (notebooks/Databricks)
viz.confusion_matrix(df, ...).show()

# Save to HTML file
viz.confusion_matrix(df, ...).save('confusion_matrix.html')

# Get Plotly figure object for further customization
fig = viz.confusion_matrix(df, ...).figure
fig.update_layout(title='Custom Title')
fig.show()
```

**Visualization Features:**
- Built on Plotly for interactive charts
- Works in Jupyter notebooks, Databricks, and can export to standalone HTML
- Light and dark theme support
- All visualizations return a wrapper object with `.show()`, `.save()`, and `.figure` methods

### 9. Configuration & Logging

```python
import smallaxe

# Set global verbosity level
smallaxe.set_verbosity('normal')  # 'quiet', 'normal', 'verbose'

# quiet   - Only errors, no progress bars or info messages
# normal  - Progress bars and key info (default)
# verbose - Detailed logging for debugging

# Check current verbosity
smallaxe.get_verbosity()  # -> 'normal'

# Context manager for temporary verbosity change
with smallaxe.verbosity('quiet'):
    model.fit(df, label_col='target')  # runs silently

# Configure Spark session (optional - uses existing session if available)
smallaxe.set_spark_session(spark)

# Set global random seed for reproducibility
smallaxe.set_seed(42)
# -> affects all random operations: train/test splits, kfold, hyperopt sampling

# Configure caching strategy for PySpark operations
smallaxe.set_cache_strategy('auto')  # 'auto', 'always', 'never'

# auto   - Smart caching: cache after preprocessing, unpersist after training (default)
# always - Cache at every stage (use for debugging or small datasets)
# never  - No automatic caching (manual control)
```

**Caching Strategy Details:**
- Cross-validation + hyperopt can cause compute explosion without caching
- `'auto'` mode:
  - Caches dataframe after preprocessing pipeline completes
  - Unpersists after model training finishes
  - Caches intermediate results during kfold
- For large datasets, `'auto'` prevents redundant recomputation

### 10. Error Handling

```python
from smallaxe.exceptions import (
    SmallaxeError,              # Base exception for all smallaxe errors
    ValidationError,            # Invalid input parameters or data
    PreprocessingError,         # Missing required preprocessing steps
    ModelNotFittedError,        # Calling predict() before fit()
    ColumnNotFoundError,        # Required column missing from dataframe
    DependencyError,            # Missing optional dependency (e.g., xgboost not installed)
    ConfigurationError          # Invalid configuration settings
)

# Example: Informative error messages
try:
    model.predict(df)
except ModelNotFittedError as e:
    print(e)
    # -> "Model has not been fitted. Call model.fit() before predict()."

try:
    pipeline.fit(df, label_col='target', numerical_cols=['age'])
except PreprocessingError as e:
    print(e)
    # -> "Random Forest requires Scaler in pipeline. Add Scaler(method='standard')
    #     or Scaler(method='minmax') before the model step."

try:
    model = Regressors.xgboost()
except DependencyError as e:
    print(e)
    # -> "XGBoost is not installed. Install with: pip install smallaxe[xgboost]"
```

**Exception Hierarchy:**
```
SmallaxeError (base)
├── ValidationError
├── PreprocessingError
├── ModelNotFittedError
├── ColumnNotFoundError
├── DependencyError
└── ConfigurationError
```

### 11. Sample Datasets

```python
from smallaxe.datasets import (
    load_sample_regression,
    load_sample_classification
)

# Load sample regression dataset (housing prices)
df_regression = load_sample_regression(spark)
# -> PySpark DataFrame with columns:
#    ['bedrooms', 'bathrooms', 'sqft', 'age', 'location', 'condition', 'price']
#    Numerical: bedrooms, bathrooms, sqft, age
#    Categorical: location, condition
#    Label: price

# Load sample classification dataset (customer churn)
df_classification = load_sample_classification(spark)
# -> PySpark DataFrame with columns:
#    ['tenure', 'monthly_charges', 'total_charges', 'contract', 'payment_method', 'churn']
#    Numerical: tenure, monthly_charges, total_charges
#    Categorical: contract, payment_method
#    Label: churn (0 or 1)

# Dataset info
from smallaxe.datasets import dataset_info

dataset_info('regression')
# -> prints column descriptions, data types, and suggested usage

dataset_info('classification')
# -> prints column descriptions, data types, and suggested usage
```

**Sample Dataset Features:**
- Pre-built PySpark DataFrames for quick demos and testing
- Includes mix of numerical and categorical features
- Realistic data distributions for meaningful model training
- Small enough to run quickly, large enough to be useful (~10,000 rows each)

## Module Structure

```
smallaxe/
├── __init__.py         # set_verbosity, get_verbosity, set_spark_session, set_seed, set_cache_strategy
├── _config.py          # Internal config state
├── training/
│   ├── __init__.py
│   ├── base.py         # BaseModel, BaseRegressor, BaseClassifier
│   ├── regressors.py   # Regressors factory
│   ├── classifiers.py  # Classifiers factory
│   ├── random_forest.py
│   ├── xgboost.py
│   ├── lightgbm.py
│   ├── catboost.py
│   └── mixins/         # Mixin classes for separation of concerns
│       ├── __init__.py
│       ├── param_mixin.py
│       ├── persistence_mixin.py
│       ├── validation_mixin.py
│       ├── metadata_mixin.py
│       └── spark_model_mixin.py
├── preprocessing/      # Imputer, Scaler, Encoder
├── pipeline/           # Pipeline
├── metrics/            # All metric functions
├── search/             # optimize
├── auto/               # AutomatedTraining
├── viz/                # Visualizer
├── exceptions/         # Custom exception classes
└── datasets/           # Sample datasets
```

## Architecture

### Model Class Hierarchy with Mixins

BaseModel has heavy responsibilities. To keep code clean and maintainable, we use a mixin-based architecture:

```python
# Mixins (smallaxe/training/mixins/)
class ParamMixin:
    """Parameter management: params, default_params, set_param, validation"""
    @property
    def params(self): ...
    @property
    def default_params(self): ...
    def set_param(self, params: dict): ...
    def _validate_param(self, name, value): ...

class PersistenceMixin:
    """Save/load functionality"""
    def save(self, path): ...
    @classmethod
    def load(cls, path): ...
    def _save_metadata(self, path): ...
    def _load_metadata(self, path): ...

class ValidationMixin:
    """Cross-validation: train_test_split, kfold"""
    def _train_test_split(self, df, test_size, stratified): ...
    def _kfold_split(self, df, n_folds, stratified): ...
    def _compute_validation_scores(self, df, metrics): ...

class MetadataMixin:
    """Training metadata tracking"""
    @property
    def metadata(self): ...
    def _capture_metadata(self, df, label_col, feature_cols): ...

class SparkModelMixin:
    """PySpark MLlib model wrapper"""
    def _assemble_features(self, df, feature_cols): ...
    def _create_spark_model(self): ...
    def _fit_spark_model(self, df): ...
    def _predict_spark_model(self, df): ...

# Base classes compose mixins
class BaseModel(ParamMixin, PersistenceMixin, ValidationMixin, MetadataMixin, SparkModelMixin):
    def __init__(self, task: str): ...
    def fit(self, df, label_col, feature_cols, exclude_cols, validation, ...): ...
    def predict(self, df): ...

class BaseRegressor(BaseModel):
    """Base for all regressors"""
    pass

class BaseClassifier(BaseModel):
    """Base for all classifiers - adds probability output"""
    def predict_proba(self, df): ...

# Algorithm implementations
class RandomForestRegressor(BaseRegressor): ...
class RandomForestClassifier(BaseClassifier): ...
class XGBoostRegressor(BaseRegressor): ...
# etc.
```

**Benefits of Mixin Architecture:**
- Single responsibility: each mixin handles one concern
- Testable: test each mixin independently
- Reusable: share logic across algorithms
- Extensible: add new capabilities without modifying base class

### Broadcast Safety for Encoders

Small categorical mappings (from StringIndexer) are automatically broadcast to prevent shuffle overhead:

```python
# Internal implementation in Encoder
def _broadcast_mapping(self, mapping_df, spark):
    if mapping_df.count() < 10000:  # threshold for broadcast
        return spark.sparkContext.broadcast(mapping_df.collect())
    return mapping_df
```

---

## Documentation

### Algorithm Selection Guide

| Scenario | Recommended Algorithm | Reason |
|----------|----------------------|--------|
| Many categorical features | CatBoost | Native categorical handling, no encoding needed |
| Large sparse data | LightGBM | Efficient leaf-wise growth, handles sparsity well |
| Quick baseline | Random Forest | No external deps, robust defaults |
| Tabular competition | XGBoost | Battle-tested, extensive tuning options |
| Small dataset (<10k rows) | Random Forest | Less prone to overfitting |
| High cardinality categoricals | CatBoost or LightGBM | Better than one-hot explosion |

### Preprocessing Decision Guide

| Data Characteristic | Imputer | Scaler | Encoder |
|--------------------|---------|--------|---------|
| Outliers present | `median` | `maxabs` or `robust` | - |
| Normal distribution | `mean` | `standard` | - |
| Bounded features | `mean` | `minmax` | - |
| Low cardinality (<20) | - | - | `onehot` |
| High cardinality (>100) | - | - | `label` or CatBoost native |
| Unknown categories at inference | custom value | - | `handle_rare='other'` |

### Anti-Patterns to Avoid

**1. Scaling Before Split (Data Leakage)**
```python
# WRONG - leaks test data statistics into training
scaler.fit(full_df)
train_df, test_df = split(full_df)

# CORRECT - fit only on training data
train_df, test_df = split(full_df)
scaler.fit(train_df)
test_df = scaler.transform(test_df)
```
*smallaxe handles this automatically when using Pipeline with validation.*

**2. Including Non-Feature Columns**
```python
# WRONG - including ID or timestamp columns as features
model.fit(df, label_col='target')  # df contains 'user_id', 'created_at'

# CORRECT - explicitly exclude non-feature columns
model.fit(df, label_col='target', exclude_cols=['user_id', 'created_at', 'row_hash'])
```

**3. One-Hot Encoding High Cardinality**
```python
# WRONG - can create millions of columns
Encoder(method='onehot')  # on a column with 50,000 categories

# CORRECT - limit categories or use label encoding
Encoder(method='onehot', max_categories=100, handle_rare='other')
# OR use CatBoost which handles this natively
```

**4. Not Caching in Cross-Validation**
```python
# WRONG - recomputes preprocessing for each fold
for fold in folds:
    preprocess(df)  # expensive!
    train(df)

# CORRECT - cache after preprocessing (smallaxe does this with cache_strategy='auto')
df_preprocessed = preprocess(df).cache()
for fold in folds:
    train(df_preprocessed)
```

**5. Ignoring Class Imbalance**
```python
# WRONG - training on heavily imbalanced data without handling
model.fit(df)  # 99% class 0, 1% class 1

# CORRECT - use stratified splits and consider class weights
model.fit(df, validation='kfold', stratified=True)
# Consider: class_weight='balanced' (future feature)
```

---

## Development Setup

**Supported Versions:**
- Python: 3.8, 3.9, 3.10
- PySpark: 3.3, 3.4, 3.5

**Branch Strategy:**
- Feature branches → PR to `main`
- Each PR triggers CI/CD (GitHub Actions)
- Tests must pass before merge

**Testing:**
- Framework: pytest with PySpark fixtures
- One test file per module (e.g., `test_imputer.py`, `test_scaler.py`)
- Local testing before each commit

**CI/CD Strategy (Realistic for Speed):**
- **PR builds**: Run fast - Python 3.10 + Spark 3.5 only
- **Main branch / Release**: Run full matrix (Python [3.8, 3.9, 3.10] × Spark [3.3, 3.4, 3.5])
- This prevents slow PR feedback while ensuring compatibility before release

---



