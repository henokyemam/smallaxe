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
  raise instead of scoring it 0.0. `mape` and `log_loss` raise `ValidationError` for a
  non-numeric column, as the evaluator-based metrics already raised; earlier, without
  ANSI mode Spark cast the strings to null and they returned 0.0.
- **Undefined validation scores are None, and k-fold means skip them.** In
  `validation_scores` (and the saved `metadata.json`) an undefined metric is stored as
  `None`. A k-fold `mean_` / `std_` uses only the folds where the metric was defined and
  `fit` emits one `UserWarning` naming each such metric and how many folds were
  undefined; a metric undefined on every fold has `mean_<metric>` of `None`. Earlier
  releases averaged the 0.0 scores in. In `optimize.run`, a trial whose metric is
  undefined fails with an error saying so, instead of competing with a 0.0 score.
- Binary `auc_roc`, `auc_pr`, and `log_loss` in `validation_scores` are `None` in two
  cases only: the fitted model writes no probability column, or it was fitted on a
  training split holding a single class (label 0 only), so its probability vectors have
  no positive-class entry and the metrics are undefined (counted as undefined folds in a
  k-fold mean). Any other error while computing them now propagates; earlier releases
  replaced every error with `None`.
- **The declared support matrix is now exactly what CI tests.** CI runs every
  (Python, PySpark) pair that PySpark itself supports, per the Python classifiers of
  each PySpark release on PyPI: PySpark 3.3 on Python 3.8 – 3.10, PySpark 3.4 and 3.5
  on Python 3.8 – 3.11, each with the latest patch release, on Linux with Java 11.
  Java 8 is tested with PySpark 3.3 and Java 17 with PySpark 3.5. A further job runs
  Python 3.8 with PySpark 3.3.0 and every other runtime dependency at its declared
  floor. Python 3.12 is tested with PySpark 3.5 and Java 17 only: no PySpark 3.x
  release declares Python 3.12, but Databricks Runtime 16.4 LTS ships that combination.
  On Python 3.12 smallaxe now depends on `setuptools`, which supplies the `distutils`
  module PySpark 3.5 imports.
  The classifiers no longer claim "OS Independent" (now "POSIX :: Linux"). README and AGENTS.md state the same matrix. A release now reaches
  PyPI only after this CI passes on the release commit and the built version equals
  the release tag.
- **The source distribution contains only the library, the examples, and the README,
  CHANGELOG and LICENSE.** A new `MANIFEST.in` leaves out `.claude/`, `.github/`, the
  tests (they need binary model fixtures), agent and planning notes, and tooling files.
  Run the tests from a git checkout.
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
  fitted with no columns skips its `Imputer` (see the pipeline entries below).
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
- **Model artifacts carry the versioned header.** `model.save(path)` writes
  `metadata.json` with top-level `format_version` (1), `kind` (`"model"`) and
  `smallaxe_version` next to the keys it wrote before, whose names and meaning are
  unchanged. Model artifacts saved by 0.8.x, 0.9 and 0.10 have no header and still load.
- **`metadata.json` is written after the Spark model**, so a directory that has one
  holds a complete artifact. Saving over an existing artifact first removes its
  `metadata.json`: a save that fails part way leaves a directory that does not load,
  instead of the old metadata next to a new or half-written Spark model.
- The nested `metadata` of a newly fitted model records `algorithm_package` and
  `algorithm_package_version` (for example `"xgboost"`, `"3.2.0"`; `None` for
  Random Forest) next to `smallaxe_version` and `spark_version`, for diagnosis. Loading
  ignores them.
- **Loading a model validates its metadata instead of guessing.** A missing or unknown
  `task` raises `ValidationError` (earlier a missing one silently gave a regression
  model; every smallaxe release has written `task`). A saved parameter that the
  algorithm's table does not have for that task raises `ValidationError` naming the
  parameter and the algorithm (the artifact was written by a newer smallaxe or is
  corrupted); parameters the artifact leaves out take their defaults. An artifact of
  another kind (for example an `Imputer` directory) or one written by a newer smallaxe
  raises `ValidationError` with the reason.
- **`Regressors.load` and `Classifiers.load` raise `ValidationError`, not
  `FileNotFoundError`, for a missing path or a directory without `metadata.json`**, with
  the same message as `Model.load` and `smallaxe.load_model`: `Cannot load model: path
  '...' does not exist.` or `Cannot load model: '...' is not a smallaxe model artifact
  (expected metadata.json in it).`
- `PersistenceMixin` no longer has a `load` classmethod; it provides `save` and reads the
  saved state back, and `Model.load` is the one load rule for models. A class built
  directly on `BaseModel` / `BaseRegressor` / `BaseClassifier` (not on `Regressor` or
  `Classifier`) can still `save` but no longer `load`.
- **A `Pipeline` fits its model on the columns it is given, and nothing else.** The
  model's features are exactly `numerical_cols` plus the Encoder's output columns for
  `categorical_cols` (`Encoder.output_cols`); every other column of the frame passes
  through the preprocessing steps untouched and never reaches the model. Earlier
  releases handed the model every non-label column, so an unlisted string ID column made
  `fit` fail and an unlisted numeric column silently became a feature. With neither
  list, the model infers its features as `model.fit` does without `feature_cols`
  (numeric and boolean columns, one `UserWarning` naming the others). The chosen columns
  are `pipeline.feature_cols`, and predict uses the model's fitted feature columns, so
  extra columns at predict time are harmless. `categorical_cols` given to a duck-typed
  model with no Encoder are handed to it raw, as before. `fit` raises `ValidationError`
  when `label_col` is also listed in `numerical_cols` or `categorical_cols`.
- **Pipeline steps are dispatched on what they are, not on their class name.** An
  `Imputer`, `Scaler`, `Encoder` or smallaxe model (`Regressor` / `Classifier`) is
  recognised with `isinstance`, so subclasses work; a class that merely shares a name
  with one (`class RandomForestRegressor: ...`) is a duck-typed step. Whether a model
  needs an Encoder comes from its algorithm record (`accepts_raw_categoricals`);
  `Pipeline.MODEL_TYPES_REQUIRING_ENCODER` is removed. `PreprocessingError.algorithm`
  is the algorithm name (`"random_forest"`), not the class name.
- **A CatBoost pipeline with `categorical_cols` needs an Encoder**, like every other
  algorithm, and `fit` raises `PreprocessingError` without one. It used to be exempt but
  then failed inside Spark, because features are assembled with `VectorAssembler`, which
  rejects string columns. The CatBoost algorithm record's `accepts_raw_categoricals` is
  now `False`.
- **A pipeline step that receives no columns is skipped explicitly**: an `Imputer` with
  neither list, a `Scaler` without `numerical_cols`, an `Encoder` without
  `categorical_cols`. It stays unfitted, `transform` passes the data through it, and
  `pipeline.skipped_steps` names it; it is saved and loaded as skipped. The `Scaler` and
  `Encoder` are fitted on the lists as given: a listed column that is missing raises
  `ColumnNotFoundError` instead of being dropped from the list without a message.
- **A duck-typed preprocessing step is fitted as `fit(df, numerical_cols=...,
  categorical_cols=...)` and nothing else.** Errors from its `fit` propagate; 0.10 caught
  `TypeError`, retried `fit(df)`, and on a second `TypeError` left the step unfitted
  without a message.
- `Pipeline(steps)` raises `ValidationError` for a step that is neither a preprocessing
  step (`fit` and `transform`) nor a model (`fit` and `predict`), and for more than one
  model step.
- **Pipeline artifacts are versioned and written through the Hadoop FileSystem, with no
  pickle.** `Pipeline.save(path)` saves each fitted step with its own `save` into
  `<path>/steps/<index>_<name>/` and then writes `<path>/metadata.json` (kind
  `"pipeline"`: the ordered steps with name, kind, directory and skipped flag,
  `numerical_cols`, `categorical_cols`, `label_col`, `feature_cols`, `has_model`), so
  pipelines save and load on `dbfs:/`, `s3://` and `file:` URIs as well as local paths.
  Saving over an existing pipeline removes its old metadata and step directories (in
  either layout) first. A step that is not an Imputer, Scaler, Encoder or smallaxe model
  cannot be saved: `save` raises `ValidationError` naming it before writing anything,
  where 0.10 pickled it. `Pipeline.load` loads each step by kind and raises
  `ValidationError` for a missing path, another kind of artifact, a newer
  `format_version`, a missing step directory, or corrupted metadata.
- **Pipelines saved by 0.10 still load** when their steps are an Imputer, a Scaler and an
  Encoder (the only pipelines 0.10 could load). The Encoder 0.10 pickled is the one pickle
  read, through an unpickler that resolves nothing but the `Encoder` class. Any other
  `.pkl` in a 0.10 pipeline (a pickled duck-typed step, or a stray file) raises
  `ValidationError` explaining that pickled steps are no longer loaded, as does a 0.10
  pipeline with a model step (0.10 could not load those either).

### Added

- The wheel ships `smallaxe/py.typed`, so type checkers read smallaxe's annotations.
- `Encoder.save(path)` and `Encoder.load(path)`. A fitted Encoder holds only plain data
  (no Spark models), so it also pickles.
- `smallaxe.load_model(path)` loads any saved model, whatever its algorithm, returning the
  `Regressor` or `Classifier` it was, for artifacts of this and every earlier release.
  `import smallaxe` still imports no training code; it is imported on the first call.
- `model.evaluate(df, label_col=None)` scores a fitted model on any DataFrame with the
  metrics its validation computes for its task (the same keys as `validation_scores`,
  without `validation_type` / `test_size` / fold bookkeeping), undefined metrics as
  `None`. `label_col` defaults to the label the model was fitted with. It raises
  `ModelNotFittedError` before `fit` and `ColumnNotFoundError` when the label or a
  feature column is missing.
- `Encoder.output_cols`: the columns `transform` writes for the encoded features, in
  order (the encoded columns themselves for `'label'`, the generated columns for
  `'onehot'`).
- `Pipeline.predict_proba(df, output_col="probability")` and
  `Pipeline.evaluate(df, label_col=None)` apply the preprocessing steps and delegate to
  the model; `Pipeline.predict` gains `output_col` (default `"predict_label"`, as
  `model.predict`). They raise `ValidationError` on a pipeline without a model, and
  `predict_proba` does on a regressor. `pipeline.feature_cols` and
  `pipeline.skipped_steps` report what `fit` decided.
- `smallaxe.load_pipeline(path)` loads a saved pipeline, and `smallaxe.load(path)` loads
  any smallaxe artifact (model, pipeline, Imputer, Scaler, Encoder, including the
  unversioned model, pipeline, Imputer and Scaler layouts of 0.10 and earlier), choosing
  by the kind the artifact records. `load_model` and `load_pipeline` reject other kinds.
  `import smallaxe` still imports neither.


### Fixed

- **A pipeline with a model step can be saved and loaded.** 0.10's `Pipeline.load` had no
  way to load a model step, so such a pipeline could be saved but never loaded.
- `Pipeline.save` and `Pipeline.load` wrote and read with local `open()` / `os`, so a
  pipeline saved to `dbfs:/` or `s3://` was split between filesystems or failed.
- **XGBoost trains the number of rounds `n_estimators` asks for.** Every earlier release
  sent it to `xgboost.spark` as `num_round`, which that package does not know: it filed
  the value under `arbitrary_params_dict` and trained its default 100 rounds whatever
  `n_estimators` said. `n_estimators` is now forwarded under its own name, so models
  with a non-default `n_estimators` train a different number of trees and give
  different predictions and validation scores than before. XGBoost models saved by
  earlier releases keep their 100 trees when loaded; refit them to get the requested
  number. XGBoost's `seed` was not affected: it reaches the booster and controls
  training.
- **Every `optimize.run` trial scores on the same split, even unseeded.** With no
  `seed` argument, no `set_seed`, and no model `seed`, each trial's `fit` drew its own
  random split seed, so trials were compared on different folds despite the docstring.
  The search now draws one split seed up front and every trial splits with it (the
  trials' `metadata["split_seed"]` is identical). Unseeded estimators stay unseeded, and
  `best_model` does not keep the search's split seed for later fits.
- **`import smallaxe.training` no longer imports the optional algorithm packages.**
  Computing `XGBOOST_AVAILABLE`, `LIGHTGBM_AVAILABLE`, and `CATBOOST_AVAILABLE` at import
  imported `xgboost.spark` (and with it scikit-learn), `synapse.ml.lightgbm`, and
  `catboost_spark`, and the last two patch `pyspark.ml.wrapper` for the whole process.
  The three names are still importable from their modules, but are now probed when
  read. The packages are imported when an algorithm is first checked or used; the first
  time either JVM package is needed, `catboost_spark` is imported before
  `synapse.ml.lightgbm` (each skipped if missing), the order the CatBoost compatibility
  shim needs, whichever algorithm comes first.
- **A broken optional install is reported as such.** When an algorithm's package is
  present but fails to import (for example a missing dependency of its own),
  `DependencyError` now says it could not be imported and includes the original error,
  instead of claiming the package is not installed.
- **Loading a sample dataset no longer reseeds Python's `random` module.**
  `load_sample_regression` and `load_sample_classification` called `random.seed(seed)`,
  which also fixed the random split seed of every later unseeded `fit`. They now use
  their own `random.Random(seed)`; the rows generated for a given seed are unchanged.
- An error while reading a model's parameters no longer silently leaves the model
  unseeded during `fit`; it propagates.
- The Random Forest `max_depth` description said 0 meant unlimited. In PySpark depth 0
  is a single leaf node, and the maximum is 30.
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
