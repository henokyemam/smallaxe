# Changelog

## 0.9.0 (unreleased)

Release this as 0.9.0, not 0.8.2: multiclass validation scores change meaning.

### Changed

- **Multiclass `validation_scores` now use macro averages.** For `task="multiclass"`,
  `precision`, `recall`, and `f1_score` (and their k-fold `mean_` / `std_` keys) were
  class 1's one-vs-rest scores in 0.8.x. They are now macro averages. Models saved by
  0.8.x keep the old values under the same keys, and loading one warns; refit to get
  the new scores.
- `optimize.run` rejects a metric the model's task does not produce (for example
  `auc_roc` for multiclass) before running any trial, instead of failing every trial.
- One `Regressor` / `Classifier` class now serves all four algorithms, configured by a
  declarative algorithm record. `Regressors.<algorithm>()` and `Classifiers.<algorithm>()`
  return these classes.
- `smallaxe[xgboost]` also installs scikit-learn and pyarrow, which `xgboost.spark`
  needs.

### Added

- `*_macro` and `*_weighted` precision, recall, and F1 in multiclass `validation_scores`,
  also usable as `optimize.run` metrics.
- `smallaxe.metrics.multiclass_scores` computes accuracy and all averaged scores in one
  Spark pass. It requires numeric label and prediction columns and ignores rows where
  either is null.
- `average="macro"` / `average="weighted"` on `smallaxe.metrics.precision`, `recall`,
  and `f1_score`. The default stays `average="binary"` (class 1).
- `smallaxe.metrics.TASK_METRICS`: the scores validation produces for each task.
- `feature_importances` for XGBoost, LightGBM, and CatBoost.

### Deprecated

- `XGBoostRegressor`, `XGBoostClassifier`, `LightGBMRegressor`, `LightGBMClassifier`,
  `CatBoostRegressor`, `CatBoostClassifier`, `RandomForestRegressor`, and
  `RandomForestClassifier` warn on construction and will be removed in 1.0. Use the
  `Regressors` / `Classifiers` factories. Models they saved still load.

### Removed

- The `XGBOOST_INSTALL_HINT`, `LIGHTGBM_INSTALL_HINT`, and `CATBOOST_INSTALL_HINT`
  constants from `smallaxe.training.regressors` and `smallaxe.training.classifiers`.
- The lazily populated `SparkCatBoost*` module globals in `smallaxe.training.catboost`;
  import the estimators from `catboost_spark` directly.

### Fixed

- CatBoost model calls no longer break when SynapseML patches `JavaWrapper._call_java`.
