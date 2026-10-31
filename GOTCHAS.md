# Gotchas: Recommended Fixes

Behavior that surprises users today, and the fix for each. `Goals.md` links here from
"Recommended Order Of Execution". Implement in the order below. Each item has its own
"Done when" check.

**Status (2026-10-31):** all seven items are implemented on `goals`, with the local suite
passing (`tests/test_validation.py`, `test_user_columns.py`, `test_missing_values.py`,
`test_fit_behavior.py`) and Databricks validation runs 6 and 6b passing (see `Goals.md`):
the null-row and caller-cache probes are PASS. Item 3's "Open" question is still open.
Changelog entries are under 0.10.0. One finding from run 6: catboost-spark does not persist
`featuresCol`, so `load` now reads the vector column name from the loaded Spark model.

Each item was reviewed on 2026-10-30 against the library's goal (scikit-learn simplicity,
every row scored in production, Databricks as a main runtime) and against ML rigor. The
review changed the design of items 3 to 7 and added items 1 and 2. Claims marked
"(measured)" come from local Spark 3.5.8 experiments run during that review.

Rules for every item:

- Do not drop data or change results silently. Keep the data, or fail with a clear error.
- Put per-algorithm behavior on the `Algorithm` record, not in `if algorithm == ...` code.
- A warning must not add a Spark action to `predict()`. `predict()` stays lazy.
- A split or fold must depend only on the rows and the seed. It must not depend on
  partitioning, caching, or how many times Spark reads the source.
- Add a `CHANGELOG.md` entry for each behavior change.

## 1. K-fold and train/test splits change when Spark reads the source again

**Implemented.** `split_hash` in `validation_mixin.py`: `hash(<hashable columns>, seed) & 0x7FFFFFFF`, one rule for all four variants; `fold_key=` on `fit` and `optimize.run`; maps and ML vectors are excluded from the hash. Stratified and unstratified splits now share the rule (a content hash is label-independent), so `stratified` only validates its arguments.

**Today.** K-fold assigns folds with
`hash(monotonically_increasing_id(), seed) % n_folds` (`validation_mixin.py:165`, and
`:203` for stratified). `train_test` uses `randomSplit` (`:56`, `:90`). Both are lazy.
Each fold's train and validation sets are separate lineages that Spark evaluates in
separate jobs, and `fit` does not cache by default. So each read can assign rows again:

- With a non-deterministic step upstream (a UDF, `rand()`, a table that changes), fold 0's
  train and validation sets shared 311 rows. The five validation sets covered 1,336 of
  2,000 distinct rows. After caching the source: no overlap, and all 2,000 rows
  (measured). Overlapping folds make the cross-validation scores too optimistic.
- The same 2,000 rows in 3 partitions instead of 4 moved 1,220 rows to a different fold
  (measured). `monotonically_increasing_id()` depends on partitioning, so `set_seed` does
  not reproduce folds across cluster sizes, and autoscaling can repartition between reads.
- Search trials can score on different folds, so `optimize` compares noise.

**Do.**

1. Assign folds and the train/test split from a hash of the row's content and the seed:
   `hash(<input columns>, seed)`. Use one rule for `train_test`, `kfold`, and both
   stratified variants.
2. Add an optional `fold_key=` to `fit` and `optimize.run`. If given, hash only that column.
   It keeps all rows of one entity (a customer, a session) in the same fold.
3. Exclude columns that `hash` cannot take, such as ML vectors, and document it.

**Breaking.** Small. Fold and split assignment changes, so validation scores change. Exact
duplicate rows now land in the same fold, which is the correct behavior.

**Done when.** On an uncached source with a non-deterministic UDF upstream, the k-fold
validation sets are pairwise disjoint and cover every row. The same data and seed give the
same folds in 2 and in 8 partitions. Both are tests.

## 2. A `features` column on the input DataFrame replaces the model's features

**Implemented.** The vector is assembled into `__smallaxe_features__`; user columns named `features`, `prediction`, `probability`, or `rawPrediction` are set aside during a transform and restored; an existing `output_col` raises `ValidationError`. 0.8.x artifacts record no `features_col` and keep reading `features`.

**Today.** `_assemble_features` returns early if the DataFrame already has a `features`
column (`spark_model_mixin.py:80`). The model then trains on that vector instead of
`feature_cols`. On a frame with `x`, `z`, a 3-wide `features` vector, and `y`, metadata
reported `feature_cols=['x','z']` and the model had `numFeatures=3` (measured). `predict`
also drops the user's own `features` column from its output (`:164`).

**Do.**

1. Always assemble into an internal column name that cannot collide with user columns,
   and pass that name to the estimator through the existing column kwargs.
2. Leave user columns named `features`, `prediction`, `probability`, or `rawPrediction`
   alone in the output. Raise `ValidationError` if a user column would be overwritten.

**Breaking.** Yes, for users who relied on a prebuilt `features` vector. That use was not
documented, and the metadata about it was wrong.

**Done when.** A frame with a `features` column trains on `feature_cols` only, and the
column is still in the output of `predict`.

## 3. Training and prediction drop rows with null or NaN features

**Implemented** except the Databricks rerun. `Algorithm.accepts_missing` (true for xgboost, lightgbm, catboost); `handleInvalid` is `keep` or `error`; `check_training_data` in `base.py` runs once in `fit` and once in `optimize.run` and replaces the metadata `count()`; `Imputer` ignores NaN. The former `test_fit_spark_model_skips_null_feature_rows` now asserts the error.

**Today.** `_assemble_features` uses `VectorAssembler(handleInvalid="skip")`
(`spark_model_mixin.py:86`). This affects `fit`, validation scoring, `predict`, and
`predict_proba`. Rows with a null or NaN feature are removed without a message. On
Databricks, `predict` dropped 513 of 10,892 rows. `metadata["n_samples"]` counts the raw
DataFrame, so it also counts the skipped rows. Dropping rows is unbiased only when values
are missing completely at random, and the validation scores are then computed on a
non-random subset. `test_fit_spark_model_skips_null_feature_rows` in `tests/test_mixins.py`
locks in the current behavior.

Engine behavior:

- Spark Random Forest training raises a deep `RuntimeException` on NaN or Infinity.
  At predict time the same model scores NaN rows silently (measured).
- xgboost.spark trains with `missing=nan` and predicts every row (measured).
- SynapseML LightGBM (`useMissing=true`) and catboost-spark (`nan_mode=Min`) handle NaN
  natively, according to their docs. Not yet tested here.
- `VectorAssembler` in `error` mode raises on null and NaN, but lets `+inf` through. It
  raises only when an action builds the vector (measured).

**Do.**

1. Add `accepts_missing: bool` to `Algorithm`, next to `accepts_raw_categoricals`. Set it
   to `True` for xgboost, lightgbm, and catboost, and `False` for random_forest. Pass the
   policy into `_assemble_features` as an argument.
2. If `accepts_missing` is true, use `handleInvalid="keep"`. Nulls become NaN, and the
   engine treats NaN as missing.
3. If `accepts_missing` is false, use `handleInvalid="error"`.
4. Check the data once, at the top of `fit`, on the full DataFrame. Fold the check into the
   existing `count()` (`metadata_mixin.py:71`), so it costs no extra pass. Do not repeat it
   per fold.
   - Count null and NaN for each feature column. When `accepts_missing` is false, count
     `±inf` too. Raise `ValidationError` with the column names, and tell the user to add an
     `Imputer`.
   - Count null labels for every algorithm, and raise `ValidationError` if there are any.
   - Store the missing count of each feature in the metadata. A boosting model learns a
     missing-value branch only for features that had missing values in training.
5. Run the same check once in `optimize.run`, before the first trial. Today a trial error
   is caught (`optimize.py:345`), so bad data fails all `max_evals` trials first.
6. In `predict`, do not add a check. Spark raises its own error when the user runs an
   action.
7. Fix the `Imputer`: compute fill values ignoring NaN. Today `F.mean` returns NaN on a
   column with NaN (`imputer.py:156`), so the imputer fills nulls with NaN, and the advice
   in step 4 fails.
8. Do not add a `skip` option. Users who want it can call `df.dropna(subset=...)`.

**Breaking.** Yes. Random Forest with nulls now fails in `fit`. Boosting models now return
predictions for rows they dropped before. Update the test in `test_mixins.py`. Until
`Goals.md` step 3 gives the Encoder an unknown index, label encoding turns an unseen
category into null, and a Random Forest `predict` then fails on that row instead of
dropping it.

**Done when.** The null-row probe in `examples/databricks_all_algorithms_validation.py`
passes. `predict` returns one row for each input row for the three boosting algorithms.
Random Forest with nulls raises `ValidationError` with the column names. A null label
raises `ValidationError`. `Imputer` output contains no NaN.

**Open.** Check on Databricks what LightGBM and CatBoost do with NaN at predict time in a
feature that had no NaN in training.

Refines `Goals.md` step 3 ("stop dropping rows at assembly").

## 4. `validation="train_test"` keeps the model trained on 80% of the rows

**Implemented.** `fit(refit=True)`; metadata `refit`, `kept_model`, `n_training_rows`; search trials run with `refit=False`.

**Today.** `_fit_train_test` trains on the train split and scores on the test split.
Then it keeps that model (`base.py:386-392`). `kfold` trains on all rows after the folds
(`base.py:430`). Thus the two modes give you different models. The README's main example
fits with `train_test`, then saves and predicts, so users ship the 80% model. No algorithm
uses the validation split for early stopping, so a refit loses nothing.

**Do.**

1. Add `refit: bool = True` to `BaseModel.fit`. After `train_test` scores the model,
   train again on all rows. This is the same as `kfold` today, and the same as
   scikit-learn's `GridSearchCV(refit=True)`. The scores then estimate the model, which saw
   more data, so they are slightly pessimistic.
2. With `refit=False`, keep the model from the last training split: the train split for
   `train_test`, and the last fold for `kfold`. The `kfold` mean score does not describe
   the last-fold model. Document `refit=False` as a way to save training in search, not as
   a holdout feature.
3. In `optimize.run`, call each trial with `refit=False`. This is required, not optional:
   without it, each `train_test` trial trains twice. `best_model` already gets one final
   training on all rows.
4. Store in the metadata `refit`, `kept_model` (`"full"`, `"train_split"`, or
   `"last_fold"`), and `n_training_rows` (rows the kept model saw). Keep `n_samples` as the
   number of input rows.
5. Users who want a test set that no model trains on use the public split utilities and
   `evaluate()` (`Goals.md` steps 6 and 9): split, `fit(train, validation="none")`, then
   `evaluate(test)`.

**Breaking.** Yes. Models trained with `train_test` now use 100% of the rows, so their
predictions change. The scores do not change.

**Done when.** `train_test` and `kfold` both keep a model trained on all rows by default.
`refit=False` keeps the model from the last split and does not train on all rows. A search
trial trains once for `train_test` and `n_folds` times for `kfold`. The metadata names the
kept model.

## 5. `fit()` uncaches your own DataFrame, and k-fold reads the source many times

**Implemented** except the Databricks rerun. Detection uses `df.storageLevel`; note that a derived plan (`df.select("*")`) reports `NONE` even over cached data, which is harmless: smallaxe then caches and uncaches only its own plan. `cache_strategy=None` is the default; `optimize.run` persists once around the search.

**Today.** `persist()` returns the same DataFrame object. So with
`cache_strategy="memory"` or `"disk"`, `fit` unpersists the user's own DataFrame
(`base.py:193`, `:211-213`). This happens also if the user cached it before `fit`. By
default `cache_strategy="none"`, so `kfold` reads the source again for each fold, for the
final fit, and for the metadata count. `optimize.run` does not pass `cache_strategy`, so
trials cannot cache at all.

`df.is_cached` does not detect the user's cache. A second DataFrame object over cached data
reports `is_cached=False`. After smallaxe persisted and unpersisted that second object, the
user's data was no longer cached, and the user's object still reported `is_cached=True`
(measured).

**Do.**

1. Detect an existing cache with `df.storageLevel != StorageLevel.NONE`. Unpersist only
   what smallaxe persisted.
2. Make `fit(cache_strategy=None)` the default, meaning: if `fit` reads the data more than
   once (`kfold`, or `train_test` with `refit=True`) and the user has not cached it,
   persist with `MEMORY_AND_DISK` for the duration of `fit`, then unpersist in `finally`.
   `"none"`, `"memory"`, and `"disk"` keep their meanings.
3. If `persist` fails, continue without a cache and warn once. Databricks serverless
   supports `persist` only in environment version 5 and later, and admins can disable it.
4. In `optimize.run`, persist once around the whole search, and call the trials with
   `cache_strategy="none"`.
5. Use item 4's `refit=False` in search trials.

Caching is a performance step only. Item 1 makes the folds correct without it.

**Breaking.** No. Results stay the same, and fit becomes faster. Memory use during `fit`
goes up by the size of the cached data.

**Done when.** A DataFrame that the user cached, through any object over the same data, is
still cached after `fit`. A search reads the source once. A search trial with `kfold`
trains `n_folds` times, not `n_folds + 1`.

Covers the caller-cache bug in `Goals.md` item 6 and step 5.

## 6. Most global settings are stored but not used

**Implemented.** Seed order is the model's `seed` parameter, then the global seed, then a random draw for splits (recorded as `metadata["split_seed"]`); the estimator gets the global seed when its own is None; `optimize.run(seed=None, verbose=None)` read the globals; `set_cache_strategy` and `set_spark_session` warn.

**Today.** Of the four global settings, only `set_seed` has an effect, and it does less
than it says:

- Its docstring says it controls hyperopt sampling (`__init__.py:110`). It does not.
  `optimize.run` uses only its own `seed` argument (`optimize.py:358-362`).
- It never reaches the estimator. The algorithm `seed` defaults to `None`
  (`random_forest.py:44`, `xgboost.py:27`), so `set_seed(42)` still trains a different
  forest each run. Metadata records the global seed (`metadata_mixin.py:73`), not the seed
  the estimator used.
- Without a seed, `kfold` falls back to 42 (`validation_mixin.py:161`), and `train_test` is
  random (`:58`).

`set_verbosity`, `set_cache_strategy`, and `set_spark_session` store a value, but no
library code reads it. The global cache values (`auto` / `always` / `never`) are different
from the `fit(cache_strategy=)` values (`none` / `memory` / `disk`). The validation notebook
calls `set_verbosity("quiet")`, and `AGENTS.md` lists the `fit` cache values only.

**Do.**

1. **Seed: wire it up fully.** The order is the call's `seed=` argument, then the global
   seed, then unseeded. Use it for splits and folds, for the estimator seed when the user did
   not set one, and in `optimize.run(seed=None)`. Use one unseeded behavior for `kfold` and
   `train_test`. Record the seeds actually used in the metadata. Fix the docstring, and
   document that a seed reproduces results for the same data. Item 1 makes this hold across
   partitionings.
2. **Verbosity: wire it up.** `optimize.run(verbose=None)` reads the global verbosity.
   With `quiet`, show no progress bar.
3. **Cache strategy: deprecate.** `set_cache_strategy` warns with `DeprecationWarning` and
   has no effect. Remove it at 1.0. Caching depends on the DataFrame, not the process, and
   item 5 gives `fit` a good default.
4. **Spark session: deprecate.** `set_spark_session` / `get_spark_session` warn with
   `DeprecationWarning`. Remove them at 1.0. Use `df.sparkSession` where a DataFrame is at
   hand, and the active session elsewhere. A stored session can go stale after a restart,
   and can differ from the one Databricks provides.

**Breaking.** Small. The deprecated settings warn. Users of `set_seed` get a seeded
estimator, so their trained models change, toward what they expected.

**Done when.** `set_seed(42)` and two fits give identical predictions. Each public setting
changes the behavior of the library, or gives a deprecation warning. One set of cache
values is in the code, `README.md`, and `AGENTS.md`.

## 7. Feature inference ignores string and boolean columns without a message

**Implemented.** `infer_feature_cols` / `warn_ignored_cols` in `base.py`; booleans are features; one `UserWarning` per `fit` or per search; `metadata["ignored_cols"]` and `metadata["feature_cols_inferred"]`. The "Open" decision on requiring explicit `feature_cols` is still open.

**Today.** If you do not give `feature_cols`, `_infer_feature_cols` keeps only
`NumericType` columns (`base.py:270-277`). String columns are not used, and boolean
columns are not used either, because `BooleanType` is not a `NumericType` in PySpark.
`VectorAssembler` turns booleans into `1.0` / `0.0` without a cast (measured). The Pipeline
always passes explicit `feature_cols`, so this affects direct `fit` and `optimize.run`,
which infers again in each trial (`optimize.py:334-338`).

Inference also includes every numeric column: numeric IDs, timestamps stored as numbers,
row indices, and a `predict_label` left over from an earlier `predict`. That is a larger
risk to the model than the columns it leaves out.

**Do.**

1. Include boolean columns as features.
2. Collect the columns that inference does not use, other than the label and
   `exclude_cols`. Warn once per `fit` with one `UserWarning` that names them and their
   types, and lists the inferred feature columns. Tell the user to encode them or to put
   them in `exclude_cols`. Set `stacklevel` to point at the user's call. This reads only the
   schema, so it costs no Spark action.
3. In `optimize.run`, infer once before the first trial, and pass explicit `feature_cols`
   to each trial.
4. Store `metadata["ignored_cols"]` as `{name: type}`, and `metadata["feature_cols_inferred"]`.
5. Do not encode the columns automatically.

**Breaking.** Small. Models trained on frames with boolean columns change.

**Done when.** A DataFrame with a string ID column and a boolean column gives one warning
that names the string column, and the boolean column is a feature. A DataFrame with only
numeric and boolean columns gives no warning. A search gives at most one warning.

**Open.** Decide whether v1 requires explicit `feature_cols` (or a strict mode) when the
DataFrame has non-numeric columns.

## Already tracked in `Goals.md`

Do not make separate fixes for these. They are recorded here so that the list of
gotchas is complete.

| Gotcha | Where it is tracked |
|---|---|
| `Pipeline.save` / `load` use local `os` paths, and pipelines with an `Encoder` or a model step cannot be saved and loaded again | Step 2 (pipeline persistence) |
| LightGBM and CatBoost need JVM packages, and CatBoost fails on autoscaling clusters | Step 8: warn when CatBoost trains on an autoscaling cluster |
| `smallaxe/auto/` and `smallaxe/viz/` are empty | Step 10. Keep both out of the README and `__all__` until they are implemented. |
| `_fs._spark()` calls `spark._jvm`, which Spark Connect and serverless sessions do not have | Step 8 (Databricks ergonomics) |
