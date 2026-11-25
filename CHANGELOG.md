# Changelog

## 0.11.0 (unreleased)

### Changed

- **`Encoder` always has an index for categories it did not see at fit.** If the
  column has an `__OTHER__` bucket (`handle_rare="other"` with more than
  `max_categories` levels), unseen values go there as before. Otherwise the mapping
  ends with a reserved `__UNKNOWN__` entry. Label encoding maps an unseen category to
  that index instead of null; one-hot encoding no longer fails on it. This holds for
  every `handle_rare`, including `keep` and `error`. A null input stays distinct:
  null under label encoding, 0.0 in every one-hot column of the feature. Rows are never
  dropped, and `transform` does not raise on data values. `category_mappings` includes
  the bucket entry.
- **One-hot encoding adds a `{col}_UNKNOWN` column** (last in the column's block)
  whenever the column has no `{col}_OTHER` column. Other one-hot names are unchanged,
  except that a category whose name collides with a bucket column or, ignoring case, with
  another category's column now gets `_{index}` appended instead of overwriting it.
- **Encoder categories are ranked by count, then by the category string**, so ties no
  longer depend on partitioning and the same data always gives the same mapping.
- `Encoder.fit` raises `ValidationError` when a column contains the literal value
  `__OTHER__` or `__UNKNOWN__`.
- **`Imputer`, `Scaler` and `Encoder` artifacts are a directory with a versioned
  `metadata.json`** (`format_version`, `kind`, `smallaxe_version`, the constructor
  arguments and the fitted state); the Scaler keeps its `spark_model/` next to it. All
  IO goes through the Hadoop FileSystem layer, so `dbfs:/`, `s3://` and `hdfs://` paths
  work. Imputer and Scaler artifacts saved by 0.10 and earlier
  (`imputer_metadata.json` / `scaler_metadata.json`) still load. Loading a missing path,
  the wrong kind of artifact, or one written by a newer smallaxe raises
  `ValidationError` with the reason.
- `Imputer.fit` raises `ValidationError` when given neither `numerical_cols` nor
  `categorical_cols`, as `Scaler` and `Encoder` already did for empty input. A `Pipeline`
  fitted with no columns leaves its `Imputer` unfitted and passes the data through.
- `Imputer` `most_frequent` ties go to the smallest value (by its string form), so the
  fill value no longer depends on partitioning. Categorical fill values are stored as
  Spark's string form of the value (`"3"`, `"true"`).
- `Scaler.fit` raises `ValidationError` naming the column and its type when an input
  column is neither numeric nor boolean (booleans scale as 0/1, as before).
- One-hot `Encoder.transform` raises `ValidationError` when an output column would
  overwrite a column the Encoder does not encode (for example a pass-through column
  `c_a` when `c` has the category `a`). 0.10 overwrote it without a message.
- An `Encoder` pickled by 0.10 (as `Pipeline.save` stored it, `step_N_encoder.pkl`)
  still loads. Unpickling adds the `__UNKNOWN__` entry to every mapping without an
  `__OTHER__` bucket, so seen values and nulls encode as before and unseen values follow
  the new rule above instead of becoming null.

### Added

- `Encoder.save(path)` and `Encoder.load(path)`. A fitted Encoder holds only plain data
  (no Spark models), so it also pickles.

### Fixed

- **Boolean, double, date and other non-string categorical columns encode correctly.**
  `Encoder.fit` built category keys with Python's `str()` (`"True"`) while `transform`
  looked them up with Spark's cast (`"true"`), so those categories never matched
  (label encoding returned null for every such value).
- **`Imputer` no longer leaves a non-string categorical column unfilled without a
  message.** 0.10 cast the fill to the column's type, and when the cast failed
  (`categorical_strategy="UNKNOWN"` on an integer column, `"-1"` on a boolean or date
  column, `most_frequent` on an all-null integer column) the nulls stayed null. When the
  cast lost the value (`"1.5"` on an integer column) it filled `1`. `fit` now raises
  `ValidationError` in both cases; a fill that fits the column's type works as before.
- `Imputer` `most_frequent` on a float or double categorical column no longer counts
  NaN as a value (it could pick NaN as the fill). NaN in such a column is filled like
  null, as 0.10 did and as the numerical strategies do.
- **`Scaler` leaves user columns named `_assembled_features`, `_scaled_features` or
  `_scaled_array` untouched.** Its working columns now get names that no input column
  uses; before, such a user column made `fit` fail or was dropped by `transform`.
- One-hot `Encoder.transform` keeps an output column whose name equals another encoded
  input column (with `categorical_cols=["c", "c_a"]`, `c`'s category `a` gives `c_a`).
  0.10 dropped it together with the input `c_a`, so rows with `c = "a"` encoded like a
  null.
- `Imputer.load`, `Scaler.load` and `Encoder.load` raise `ValidationError` instead of
  `FileNotFoundError` / `KeyError` / `TypeError` for missing, incomplete or corrupted
  artifacts, including a `metadata.json` edited after saving whose Hadoop `.crc` no
  longer matches. Errors that are not Spark's own load failures are no longer wrapped.
- `Imputer` and `Scaler` metadata was written with local `open()`, so `save` / `load`
  on `dbfs:/` or other non-local paths wrote the metadata somewhere other than the
  Spark model.

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
