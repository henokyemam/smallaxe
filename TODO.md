## TODO LIST - Build Phases

### Phase 8: Training Module - External Algorithms (v0.7.0)

#### Step 8.2: LightGBM
- [ ] Create `smallaxe/training/lightgbm.py`
- [ ] Implement `LightGBMRegressor` and `LightGBMClassifier`
- [ ] Handle optional dependency
- [ ] Create `tests/test_lightgbm.py`
- [ ] Commit: "Add LightGBM support"
- [ ] PR → main

#### Step 8.3: CatBoost
- [ ] Create `smallaxe/training/catboost.py`
- [ ] Implement `CatBoostRegressor` and `CatBoostClassifier`
- [ ] Note: CatBoost handles categoricals natively (no preprocessing required)
- [ ] Create `tests/test_catboost.py`
- [ ] Commit: "Add CatBoost support"
- [ ] PR → main
- [ ] **Tag: v0.7.0**

---

### Phase 9: Cross-Validation Enhancement (v0.8.0)

#### Step 9.1: Validation Module
- [ ] Create `smallaxe/training/validation.py`:
  ```python
  def train_test_split(df, test_size, stratified, label_col): ...
  def kfold_split(df, n_folds, stratified, label_col): ...
  ```
- [ ] Implement stratified sampling for classification
- [ ] Create `tests/test_validation.py`:
  - Test train_test_split ratios
  - Test stratified class distribution
  - Test kfold yields correct number of folds
  - Test kfold stratification
- [ ] Commit: "Add validation utilities"
- [ ] PR → main

#### Step 9.2: Integrate with Training
- [ ] Update BaseModel.fit() to use validation module
- [ ] Add `validation_scores` attribute to models
- [ ] Update all model tests to verify validation_scores
- [ ] Commit: "Integrate validation with training models"
- [ ] PR → main
- [ ] **Tag: v0.8.0**

---

### Phase 10: Model Persistence (v0.9.0)

#### Step 10.1: Serialization
- [ ] Implement robust save/load for all models:
  - Save model artifacts (native format)
  - Save metadata (params, column info, fitted state)
  - Save preprocessing state if in pipeline
- [ ] Create `tests/test_persistence.py`:
  - Test save/load for each algorithm
  - Test load produces identical predictions
  - Test load preserves params
  - Test pipeline save/load
- [ ] Commit: "Implement model persistence"
- [ ] PR → main
- [ ] **Tag: v0.9.0**

---

### Phase 11: Optimization Module (v0.10.0)

#### Step 11.1: Optimize Class
- [ ] Create `smallaxe/search/optimize.py`
- [ ] Implement `optimize.run()`:
  ```python
  def run(model, dataframe, label_col, param_grid, metric,
          max_evals, validation, n_folds,
          early_stopping, early_stopping_rounds): ...
  ```
- [ ] Integrate with hyperopt (fmin, Trials)
- [ ] Implement progress bar (tqdm)
- [ ] Implement early stopping logic
- [ ] Return trained best model with:
  - `best_params`
  - `best_score`
  - `trials_history`
- [ ] Create `tests/test_optimize.py`:
  - Test basic optimization finds reasonable params
  - Test early stopping triggers correctly
  - Test progress bar doesn't break output
  - Test best_model is trained and ready for predict
  - Test trials_history contains all trials
- [ ] Commit: "Add hyperparameter optimization"
- [ ] PR → main
- [ ] **Tag: v0.10.0**

---

### Phase 12: Automated Training (v0.11.0)

#### Step 12.1: AutomatedTraining Class
- [ ] Create `smallaxe/auto/automated.py`
- [ ] Implement `AutomatedTraining`:
  ```python
  class AutomatedTraining:
      def __init__(self, model_type, metrics): ...
      def fit(self, df, label_col, numerical_cols, categorical_cols, n_folds): ...
      def predict(self, df): ...
      @property
      def metrics(self): ...  # returns PySpark DataFrame
      @property
      def models(self): ...   # returns dict
      @property
      def best_model(self): ...
  ```
- [ ] Train all 4 algorithms with appropriate preprocessing
- [ ] Handle missing optional dependencies gracefully (skip algorithm, warn user)
- [ ] Create `tests/test_automated.py`:
  - Test fit trains multiple models
  - Test metrics DataFrame has correct schema
  - Test best_model is correctly selected
  - Test predict uses best_model
  - Test with only some algorithms installed
- [ ] Commit: "Add AutomatedTraining"
- [ ] PR → main
- [ ] **Tag: v0.11.0**

---

### Phase 13: Visualization Module (v0.12.0)

#### Step 13.1: Visualizer Core
- [ ] Create `smallaxe/viz/visualizer.py`
- [ ] Implement `Visualizer` class with theme support
- [ ] Create plot wrapper class:
  ```python
  class PlotResult:
      def __init__(self, figure): ...
      def show(self): ...
      def save(self, path): ...
      @property
      def figure(self): ...
  ```
- [ ] Create `tests/test_viz_core.py`:
  - Test Visualizer instantiation
  - Test theme switching
  - Test PlotResult methods
- [ ] Commit: "Add Visualizer core"
- [ ] PR → main

#### Step 13.2: Classification Visualizations
- [ ] Implement `confusion_matrix()`
- [ ] Implement `roc_curve()`
- [ ] Implement `precision_recall_curve()`
- [ ] Implement `threshold_curve()` (precision, recall, f1 vs threshold)
- [ ] Create `tests/test_viz_classification.py`:
  - Test each visualization produces valid Plotly figure
  - Test correct data mapping
  - Test threshold_curve shows all metrics correctly
- [ ] Commit: "Add classification visualizations"
- [ ] PR → main

#### Step 13.3: Regression Visualizations
- [ ] Implement `actual_vs_predicted()`
- [ ] Implement `residuals()`
- [ ] Create `tests/test_viz_regression.py`
- [ ] Commit: "Add regression visualizations"
- [ ] PR → main

#### Step 13.4: Feature and Comparison Visualizations
- [ ] Implement `feature_importance()`
- [ ] Implement `compare_models()`
- [ ] Create `tests/test_viz_features.py`
- [ ] Update `smallaxe/viz/__init__.py`
- [ ] Commit: "Add feature importance and model comparison visualizations"
- [ ] PR → main
- [ ] **Tag: v0.12.0**

---

### Phase 14: Integration & Polish (v1.0.0)

#### Step 14.1: End-to-End Integration Tests
- [ ] Create `tests/test_integration.py`:
  - Full regression workflow: load data → preprocess → train → evaluate → visualize
  - Full classification workflow
  - AutomatedTraining workflow
  - Optimization workflow
- [ ] Ensure all modules work together seamlessly
- [ ] Commit: "Add end-to-end integration tests"
- [ ] PR → main

#### Step 14.2: README Documentation
- [ ] Write comprehensive README.md:
  - Installation instructions (pip, optional deps)
  - Quick start example
  - Feature overview with code snippets
  - API reference summary
  - Contributing guidelines
  - License (choose appropriate license)
- [ ] Add badges (CI status, PyPI version, Python versions)
- [ ] Commit: "Add comprehensive README"
- [ ] PR → main

#### Step 14.3: Final Review & Cleanup
- [ ] Code review all modules for consistency
- [ ] Ensure all docstrings are complete
- [ ] Run full test suite across all Python/Spark versions
- [ ] Fix any remaining issues
- [ ] Update version to 1.0.0 in pyproject.toml
- [ ] Commit: "Prepare v1.0.0 release"
- [ ] PR → main

#### Step 14.4: PyPI Publication
- [ ] Create PyPI account (if needed)
- [ ] Configure PyPI tokens in GitHub secrets
- [ ] Create GitHub release v1.0.0
- [ ] Verify GitHub Action publishes to PyPI
- [ ] Test installation: `pip install smallaxe`
- [ ] **Tag: v1.0.0** 🎉

---

## Version Summary

| Version | Features |
|---------|----------|
| v0.1.0 | Project setup, exceptions, config |
| v0.2.0 | Sample datasets |
| v0.3.0 | Preprocessing (Imputer, Scaler, Encoder) |
| v0.4.0 | Pipeline |
| v0.5.0 | Metrics |
| v0.6.0 | Training - Random Forest |
| v0.7.0 | Training - XGBoost, LightGBM, CatBoost |
| v0.8.0 | Cross-validation |
| v0.9.0 | Model persistence |
| v0.10.0 | Optimization (hyperopt) |
| v0.11.0 | AutomatedTraining |
| v0.12.0 | Visualization |
| v1.0.0 | Integration, README, PyPI publish |