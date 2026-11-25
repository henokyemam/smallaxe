# Changelog

## 0.11.0 (unreleased)

### Changed

- **Metrics no longer turn a failure into a score.** Every function in
  `smallaxe.metrics` raises `ValidationError` naming the metric when the DataFrame has
  no rows; earlier releases returned 0.0, a perfect `mse` / `rmse` / `mae`. `mse`,
  `rmse`, `mae`, and `r2` no longer catch every exception and return 0.0: Spark's
  errors propagate. A metric that is undefined on the rows given is `NaN`, not 0.0:
  `auc_roc` with one class present, `auc_pr` with no positive labels, `mape` when every
  label is zero, `log_loss` and `multiclass_scores` when no row is complete, and `r2`
  when Spark's evaluator yields NaN. `r2` keeps scikit-learn's rule for constant labels
  (1.0 on exact predictions, else 0.0), and per-class precision, recall, and F1 with a
  zero denominator are still 0.0. An empty validation split or fold now makes `fit`
  raise instead of scoring it 0.0.
- **Undefined validation scores are None, and k-fold means skip them.** In
  `validation_scores` (and the saved `metadata.json`) an undefined metric is stored as
  `None`. A k-fold `mean_` / `std_` uses only the folds where the metric was defined and
  `fit` emits one `UserWarning` naming each such metric and how many folds were
  undefined; a metric undefined on every fold has `mean_<metric>` of `None`. Earlier
  releases averaged the 0.0 scores in. In `optimize.run`, a trial whose metric is
  undefined fails with an error saying so, instead of competing with a 0.0 score.
- Binary `auc_roc`, `auc_pr`, and `log_loss` in `validation_scores` are `None` only when
  the fitted model writes no probability column. Any other error while computing them
  now propagates; earlier releases replaced every error with `None`.

### Fixed

- **XGBoost trains the number of rounds `n_estimators` asks for.** Every earlier release
  sent it to `xgboost.spark` as `num_round`, which that package does not know: it filed
  the value under `arbitrary_params_dict` and trained its default 100 rounds whatever
  `n_estimators` said. `n_estimators` is now forwarded under its own name, so models
  with a non-default `n_estimators` train a different number of trees and give
  different predictions and validation scores than before. XGBoost models saved by
  earlier releases keep their 100 trees when loaded; refit them to get the requested
  number. XGBoost's `seed` was not affected: it reaches the booster and controls
  training.

## 0.10.1 (unreleased)

### Fixed

- **`fit` and `optimize.run` work on Spark 3.3 again.** 0.10.0 compared the DataFrame's
  storage level with `StorageLevel.NONE`, which Spark only has from 3.4, so every `fit`
  on Spark 3.3 raised `AttributeError`. The check now reads the storage level's fields.
  Pull-request CI now runs on Spark 3.3 as well as 3.5.

## 0.10.0 (2026-10-31)

A minor release rather than 0.9.1: fold assignment, row handling, the kept model, and the
default caching all change behavior. See each entry.

### Changed

- **Folds and train/test splits are assigned from the row's content and the seed**, not
  from `monotonically_increasing_id`. They no longer depend on partitioning, caching,
  or how many times Spark reads the source, so k-fold validation sets are always
  disjoint and complete, and `set_seed` reproduces folds across cluster sizes. Exact
  duplicate rows land in the same fold. Validation scores change as a result. `fit` and
  `optimize.run` take `fold_key=` to hash one column instead, keeping all rows of an
  entity in one fold. Stratified and unstratified splits use the same rule; a content
  hash is independent of the label.
- **The assembled feature vector has an internal name.** A user column named
  `features` no longer replaces the model's features, and `predict` / `predict_proba`
  return every input column unchanged, including user columns named `features`,
  `prediction`, `probability`, or `rawPrediction`. Passing an `output_col` that already
  exists raises `ValidationError`. Models saved by 0.8.x still load and predict.
- **Rows with null or NaN features are never dropped.** Random Forest `fit` and
  `optimize.run` now raise `ValidationError` naming the columns (and their counts) and
  pointing at `Imputer`; infinite values are rejected too. XGBoost, LightGBM, and
  CatBoost keep the rows and treat NaN as missing, so `predict` returns one row per
  input row. A null label is an error for every algorithm. `metadata["missing_counts"]`
  records the missing count per feature.
- **`validation="train_test"` keeps a model trained on all rows.** `fit(refit=True)`
  (the default) trains again on the full DataFrame after scoring, as k-fold already
  did. `refit=False` keeps the model from the last training split. The metadata records
  `refit`, `kept_model` (`"full"`, `"train_split"`, `"last_fold"`), and
  `n_training_rows`.
- **`fit` no longer uncaches a DataFrame the caller cached.** A DataFrame whose
  `storageLevel` is already set is left alone. `cache_strategy` defaults to `None`:
  persist with `MEMORY_AND_DISK` only while `fit` reads the data more than once
  (k-fold, or train_test with refit). A failed `persist` warns and continues.
  `optimize.run` persists once around the whole search and runs each trial with
  `cache_strategy="none"` and `refit=False`, so a trial trains once per split.
- **`set_seed` now reaches everything it promised.** It seeds splits, the estimator
  when the model's `seed` parameter is None, and hyperopt's sampler when
  `optimize.run(seed=None)`. A model's own `seed` takes precedence. The metadata records
  the `seed` the estimator used and the `split_seed`. Without any seed, folds are drawn
  from a fresh random seed on each `fit`.
- **Feature inference includes boolean columns** and warns once, naming the columns it
  ignored and their types. `metadata["feature_cols_inferred"]` and
  `metadata["ignored_cols"]` record the outcome. `optimize.run` infers once and passes
  explicit `feature_cols` to every trial.
- `optimize.run(verbose=None)` follows the global verbosity: `quiet` hides the progress bar.

### Deprecated

- `set_cache_strategy` / `get_cache_strategy` and `set_spark_session` /
  `get_spark_session` warn and have no effect; they will be removed in 1.0. Caching is
  set per `fit` call, and the Spark session comes from the DataFrame.

### Fixed

- `Imputer` ignores NaN when computing the mean, median, or mode, so it no longer fills
  nulls with NaN.

## 0.9.0 (2026-10-26)

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
