# Changelog

## 1.0.0 (unreleased)

1.0 freezes the API. It removes every spelling that warned in 0.11 or earlier and states
the public surface explicitly: the names in each module's `__all__`, their documented
arguments, and the artifact formats. From 1.0 smallaxe follows semantic versioning; see
"Stability and versioning" in the README.

### Upgrading from 0.x

Run your code on 0.11 first: each item below warned there (unless marked "not
deprecated in 0.11"), naming its replacement. In 1.0 a removed name raises
`ImportError` / `AttributeError`, and a removed argument `TypeError`.

- `RandomForestRegressor()`, `XGBoostRegressor()`, `LightGBMRegressor()`,
  `CatBoostRegressor()` (from `smallaxe.training` or their algorithm modules): use
  `regressors.random_forest()`, `regressors.xgboost()`, `regressors.lightgbm()`,
  `regressors.catboost()` after `from smallaxe.training import regressors`.
- `RandomForestClassifier()`, `XGBoostClassifier()`, `LightGBMClassifier()`,
  `CatBoostClassifier()`: use `classifiers.random_forest(task=...)`,
  `classifiers.xgboost(task=...)`, `classifiers.lightgbm(task=...)`,
  `classifiers.catboost(task=...)`. Their `.load(path)`: use `regressors.load(path)`,
  `classifiers.load(path)` or `smallaxe.load_model(path)`. Models they saved in 0.8.1 and
  later still load.
- `Regressors.<algorithm>(...)`, `Classifiers.<algorithm>(...)`, `Regressors.load`,
  `Classifiers.load`, `Regressors.available_models()`, `Classifiers.available_models()`:
  use the same functions of the `regressors` / `classifiers` modules.
- `Regressors.list_models()` / `Classifiers.list_models()`: use
  `regressors.algorithms()` (every algorithm) and `regressors.available_models()` (which
  are installed). The `"class_name"` entry of `available_models()`: use the entry's key,
  the algorithm name.
- `smallaxe.set_cache_strategy()` / `get_cache_strategy()`: pass `cache_strategy=` to
  `fit`. The `smallaxe.VALID_CACHE_STRATEGIES` they checked against (not deprecated in
  0.11): the allowed values are in the `fit` docstring.
- `smallaxe.set_spark_session()` / `get_spark_session()`: use
  `dataframe.sparkSession` or `SparkSession.getActiveSession()`.
- `model.set_param({...})`: use `model.set_params(**params)`.
- `df=` on `Model.fit` / `predict` / `evaluate`, `Classifier.predict_proba`, every
  `Pipeline`, `Imputer`, `Scaler` and `Encoder` method, and every `smallaxe.metrics`
  function: use `dataframe=`, or pass the DataFrame positionally.
- `stratified=` in `Model.fit` and `optimize.run`: delete it; it had no effect.
- `dataset_info(name)` printing by default: it only returns the dict now; pass
  `verbose=True` to also print. `except ValueError` around it (not deprecated in 0.11):
  use `except ValidationError`.
- `metrics.log_loss(dataframe, label_col, probability_col, eps)`: pass `eps=` by keyword.
- `smallaxe.training.BaseModel` / `BaseRegressor` / `BaseClassifier`: use
  `smallaxe.training.Model` / `Regressor` / `Classifier`. The `Base*` classes remain in
  `smallaxe.training.base`.
- `smallaxe.training.base.infer_feature_cols`, `warn_ignored_cols`,
  `check_training_data` and `smallaxe.search.optimize.infer_feature_cols`,
  `warn_ignored_cols`, `check_training_data`: `Model.fit` does each of these.
- `smallaxe.training.mixins.validation_mixin.hashable_columns`, `split_hash`: use
  `fit(validation=..., fold_key=...)`. `resolve_split_seed` (in `validation_mixin`,
  `base` and `optimize`): use `smallaxe.set_seed()` or the model's `seed`.
- `smallaxe.training.algorithm.resolve`: use `get(name).estimator_class(task)` /
  `get(name).model_class(task)`. `estimator_kwargs`: build the model with a factory and
  `set_params()`. `legacy_class_name`: use the algorithm name and the model's task.
  `package_version`: read `metadata["algorithm_package_version"]` of a fitted model.
  `COLUMN_ROLES`: use `get(name).cols`. `LEGACY_CLASS_NAMES`: use
  `smallaxe.load_model()`, which reads artifacts saved by 0.8.1 and later.
- `HYPEROPT_AVAILABLE` (in `smallaxe.search` and `smallaxe.search.optimize`),
  `optimize.HYPEROPT_INSTALL_HINT`: hyperopt is a required dependency; drop the check.
- `optimize.MAXIMIZE_METRICS`, `MINIMIZE_METRICS`, `VALID_METRICS`: use
  `smallaxe.metrics.TASK_METRICS`.
- `XGBOOST_AVAILABLE`, `LIGHTGBM_AVAILABLE`, `CATBOOST_AVAILABLE`,
  `smallaxe.training.catboost.is_catboost_available()`: use
  `smallaxe.training.algorithm.get(name).available()` or
  `regressors.available_models()`. `catboost_install_hint()`: use
  `algorithm.get("catboost").dependency.install_hint`.
- `smallaxe.preprocessing.encoder.VALID_METHODS` / `VALID_HANDLE_RARE`,
  `smallaxe.preprocessing.scaler.VALID_METHODS`: the values are listed in the class
  docstrings; an invalid one raises `ValidationError`.
- `smallaxe.warnings` (the standard library module, re-exported; not deprecated in
  0.11): `import warnings`.
- Not deprecated in 0.11: `smallaxe.preprocessing.imputer.VALID_NUMERICAL_STRATEGIES` /
  `VALID_CATEGORICAL_STRATEGIES` (listed in the `Imputer` docstring);
  `MULTICLASS_AVERAGED_METRICS` in `smallaxe.metrics` and
  `smallaxe.metrics.classification` (use `TASK_METRICS["multiclass"]`); `smallaxe.preprocessing.encoder.OTHER` / `UNKNOWN` (the
  strings `"__OTHER__"` / `"__UNKNOWN__"`); `smallaxe.pipeline.pipeline.step_kind`; the
  `REGRESSION_TASKS`, `CLASSIFICATION_TASKS`, `VALID_TASKS`, `VALID_VALIDATION` and
  `VALID_CACHE_STRATEGIES` attributes of the model classes (use
  `smallaxe.training.algorithm.REGRESSION_TASKS`, `CLASSIFICATION_TASKS`,
  `ALL_TASKS`); and the artifact layout constants (`ARTIFACT_KIND`,
  `LEGACY_METADATA_FILE`, `SPARK_MODEL_DIR`, `STEPS_DIR`, `STEP_KINDS`, `IMPUTER`,
  `SCALER`, `ENCODER`, `MODEL`, `PREPROCESSING_KINDS`, `DEFAULT_PREDICTION_COL`,
  `DEFAULT_PROBABILITY_COL`) of the preprocessing, pipeline and persistence modules.
- `pipeline["name"]` for a missing step (not deprecated in 0.11): catch
  `ValidationError` instead of `KeyError`.
- Positional `fit` arguments after `n_folds` (not deprecated in 0.11): removing
  `stratified` moved
  `cache_strategy`, `fold_key` and `refit` one place left; pass them by keyword.

### Removed

- Everything in the list above, with the `smallaxe._deprecation` module (the `df=` and
  positional-`eps` wrappers and the module `__getattr__` shims) and the global
  cache-strategy and Spark-session state in `smallaxe._config`.
- `Model.load` no longer checks a class alias's algorithm (the aliases are gone); it
  checks the kind for `Regressor.load` / `Classifier.load`, as before. Artifacts saved
  by the eight 0.8.x classes (0.8.1 and later) still load: the map from their class
  names to algorithms is a private detail of loading, used nowhere else.
- The `DependencyError` path for a missing hyperopt: `smallaxe.search` imports hyperopt
  normally, so a broken install fails with hyperopt's `ImportError` on
  `import smallaxe.search`.
- Dead code the removals left behind, all private: `SparkModelMixin`'s
  `_create_spark_estimator`, `_uses_constructor_col_params` and its generic
  `_fit_spark_model` / `_get_feature_importance` (both are now abstract, and `Model`
  implements them), `Model._create_spark_estimator`, `BaseModel._infer_feature_cols`,
  `BaseModel._load_artifacts`, `BaseModel.clone` (`Model.clone` is the one clone), and
  the `stratified` / `label_col` parameters of `ValidationMixin._train_test_split` /
  `_kfold_split` with the `_stratified_train_test_split`, `_simple_kfold_split` and
  `_stratified_kfold_split` shims. A class built directly on `BaseModel` must implement
  `_fit_spark_model` and `_get_feature_importance`.

### Changed

- **The public surface is stated explicitly.** Every module defines `__all__`, and every
  module-level name outside it starts with `_`. `smallaxe.training` exports `algorithm`,
  `Model`, `Regressor`, `Classifier` and the factory modules `regressors` and
  `classifiers`. The task names are defined once, in `smallaxe.training.algorithm`
  (`REGRESSION_TASKS`, `CLASSIFICATION_TASKS`, `ALL_TASKS`).
  `tests/test_public_api.py` pins every public module's `__all__` and the parameters of
  every public function, method and class, including those implemented in a private
  module and exported by a public one (the `smallaxe.datasets` functions). The
  implementation classes `BaseModel`, `BaseRegressor`, `BaseClassifier` and the mixins
  of `smallaxe.training.mixins` stay in their modules' `__all__`, but the README's
  stability policy leaves them out of the public API: their members are public as
  members of `Model`, `Regressor` and `Classifier`.
- **Results no longer depend on Spark's ANSI SQL mode** (`spark.sql.ansi.enabled`, on
  by default from Spark 4). With ANSI mode on, an implicit cast that fails, an
  arithmetic overflow, a boolean compared with a number, or (on Spark 3.3) a missing
  map key raises where it gave null before, the string-to-integer cast no longer reads
  a fraction (`"7.0"`), and an integer meets a float in double instead of float.
  smallaxe never sets the option. Its results in either mode are the ones Spark 3's
  default mode (ANSI off) gives, with each implicit cast spelled out
  (`smallaxe._sql.integral_from_string` reads a string as an integer the way the
  default cast does; `try_cast`, `try_subtract` and `try_divide` give null where the
  default mode does). Without ANSI mode the results are unchanged, except where integer
  arithmetic overflowed (below). `tests/test_ansi_mode.py` runs the metric,
  preprocessing, visualization, split and dataset edge cases (zero denominators,
  unseen categories, constant and all-null columns, nulls, NaN, infinities, strings and
  booleans compared with numbers, float and decimal arithmetic, fills that round) with
  the option on and off and requires identical results, pins the values, and on Spark 3
  compares them with Spark's own implicit casts and operators; the whole suite also
  passes with it on (Spark 3.5). Fixed under ANSI mode:
  - `accuracy`, and `precision` / `recall` / `f1_score` with `average='binary'`, on a
    boolean column (an analysis error under ANSI mode), a string column holding a value
    that does not parse (a cast error), or an integer column compared with a float one
    (compared in double). As in Spark 3's default mode, a boolean compared with a
    number counts as 0 / 1; a string compared with an integer column is read as that
    integer type (`"7.9"` is 7), with a float or double column as that type, with a
    decimal column as a double, with a boolean column as a boolean (`"yes"`, `"1"`);
    the positive class 1 reads a string as an integer (`"1.5"` is positive, `"1e0"` is
    not); a string that does not parse never matches.
  - `mape` and `log_loss`: integer columns are computed in `decimal(20,0)`, so
    `label - prediction`, `abs(label)` and `1 - label` no longer overflow (ANSI mode
    raised; without it the value wrapped around, giving a wrong score for values near
    the integer limits; this is the one change without ANSI mode). Float and decimal
    columns keep Spark's own arithmetic (an integer beside a float is a float), and a
    decimal ratio that overflows is skipped as before instead of raising.
  - `Imputer` with a categorical fill on an integer column spelled with a fraction
    (`categorical_strategy="7.0"`): ANSI mode's string-to-integer cast rejects it, so
    `fit` raised `ValidationError` and `transform` a Spark error. The fill is now read
    the way Spark 3's default cast reads it, and checked with `try_cast` rather than a
    cast that raises; the accepted fills, their values and the rejected fills are the
    ones Spark 3's default mode gives (a fill that rounds, `"7.9999999999999999999"`,
    or is too wide for the check, is still rejected).
  - `Encoder.transform` with a category not seen at fit, on Spark 3.3 with ANSI mode on
    (a missing map key raised there): the lookup is guarded, so the category maps to the
    `__OTHER__` / `__UNKNOWN__` index in every mode.
- **A Spark Connect session raises `ConfigurationError`.** smallaxe needs a classic
  (JVM, py4j) Spark session: every `save` and `load` reads and writes through the
  Hadoop FileSystem on the JVM (`smallaxe._fs`). With a Spark Connect session active
  (`SparkSession.builder.remote(...)`, `spark.api.mode=connect`, or `pyspark-connect`
  installed on Spark 4), those calls now raise `ConfigurationError("smallaxe needs a
  classic Spark session; Spark Connect is not supported")` instead of an
  `AttributeError` from `spark._jvm`. The check reads the session's class module, so it
  never imports `pyspark.sql.connect`; a classic session is unaffected. Spark Connect
  is out of scope for 1.0.
- **`fit`, `set_params` and `clone` return the model's own type.** They were typed as
  the base class; now `regressors.xgboost().fit(...)` is typed as a `Regressor`, so
  chained calls type-check. Runtime behaviour is unchanged.
- **Public methods raise smallaxe exceptions for user errors.** Each of these raised a
  raw Python or Spark error before:
  - Any public method given something that is not a DataFrame where it expects one
    (`Model`, `Pipeline`, `Imputer`, `Scaler`, `Encoder`, every metric, `optimize.run`)
    raises `ValidationError` (was `AttributeError`). Spark Connect DataFrames are
    accepted.
  - A column list given as a string or with non-string entries (`feature_cols`,
    `exclude_cols`, `numerical_cols`, `categorical_cols`) raises `ValidationError`; a
    string used to be split into one-letter column names.
  - `fit` with an explicit feature column that is not numeric, boolean or an ML vector
    raises `ValidationError` naming it (was Spark's `IllegalArgumentException`).
    `test_size`, `n_folds`, `fold_key` and `refit` are checked before any Spark job:
    a non-numeric `test_size` or a non-integer `n_folds` raises `ValidationError` (was
    `TypeError`), and so do a non-string `fold_key` and a non-bool `refit`.
  - `predict`, `predict_proba` and `Pipeline.predict` on a frame missing a feature column
    raise `ColumnNotFoundError` (was Spark's `AnalysisException`).
  - `Regressor(...)` / `Classifier(...)` / `Model(...)` with an algorithm that is
    neither a name nor an `Algorithm` raise `ValidationError`.
  - Saving or loading with a path that is not a string (including a `pathlib.Path`:
    pass `str(path)`) raises `ValidationError`.
  - `optimize.run` raises `ValidationError` for a `model` that is not a smallaxe model,
    a `param_space` naming a parameter the model does not have (before any trial runs;
    it used to fail every trial), and a non-integer `max_evals`, `seed` or
    `early_stopping_rounds`.
  - `optimize.run` also checks before any trial: a `param_space` key that is not a
    string (was hyperopt's `TypeError`), a choice or constant that is not a number for a
    numeric parameter (`{"max_depth": ["a", "b"]}` or `{"max_depth": "ab"}`, was
    `TypeError` mid-search), a `validation` that is not one of its strings (`[]` was a
    `TypeError`), a `verbose`, `early_stopping` or `refit` that is not a bool (`refit="y"`
    used to be accepted; `verbose="y"` was a `TypeError` from hyperopt), and `test_size`,
    `n_folds` and `fold_key` as `fit` checks them (they used to fail every trial). All
    raise `ValidationError`.
  - A task that is not a string (`Classifier("random_forest", task=["binary"])`,
    `classifiers.random_forest(task=[...])`, `Model(..., task={})`) raises
    `ValidationError` (was `TypeError: unhashable type`).
  - A label column that is not numeric (a string or boolean) in `fit`, `evaluate` or
    `optimize.run` raises `ValidationError` naming it (was Spark's
    `IllegalArgumentException`, or an `AnalysisException` for XGBoost).
  - `predict` / `predict_proba` with an `output_col` that is not a non-empty string
    raises `ValidationError` (was a `Py4JError`), and so does a feature column of the
    wrong type in the frame to predict (was Spark's `IllegalArgumentException`).
  - A parameter value the Spark estimator rejects when `fit` runs (Random Forest
    `max_depth=-3`, `subsampling_rate=5.0`, `feature_subset_strategy="bogus"`) raises
    `ValidationError` with Spark's message, chained to Spark's
    `IllegalArgumentException`. `set_params` still checks only types, not ranges.
    XGBoost checks its parameters inside the training job, so a bad XGBoost value still
    fails with Spark's `Py4JJavaError`.
  - `metrics.mse`, `rmse`, `mae` and `r2` raise `ValidationError` for a non-numeric label
    or prediction column, and `auc_roc` / `auc_pr` for a non-numeric label or a
    probability column that is neither a number nor a vector (all were Spark's
    `IllegalArgumentException`). Any other argument Spark's evaluator rejects is a
    `ValidationError` too.
  - `Algorithm.estimator_class(task)` and `model_class(task)` for a task the algorithm
    does not have raise `ValidationError` (was `KeyError`). `algorithm.get()` with a
    name that is not a string raises `ValidationError` saying so (was a `TypeError` for
    an unhashable value, and "Unknown algorithm" for any other).
  - Metric functions raise `ValidationError` for a column name that is not a string.
  - `pipeline["name"]` for a missing step raises `ValidationError` (was `KeyError`).
- **`set_params` checks every value before setting any.** A bool is rejected for an int
  or float parameter (`n_estimators=True`), and `seed` must be an int or None. Earlier a
  failing value left the parameters before it set.
- **`set_seed` rejects a bool** with `ConfigurationError`; `set_seed(True)` used to be
  accepted, since a bool is an int to Python.
- **`ConfigurationError` builds its message for every value given, falsy ones
  included.** `set_verbosity("")`, `set_verbosity(0)` and `set_verbosity(None)` now say
  `Invalid value '' for setting 'verbosity'. Allowed values: [...]` instead of the
  generic default message; the value is shown with `repr`. The constructor's `value`
  argument defaults to a private sentinel instead of None, so `value=None` is reported
  as a value.
- **The verbosity levels are documented as they behave.** `'quiet'` hides
  `optimize.run`'s progress bar (and `AutomatedTraining`'s progress lines, see "Added");
  `'normal'` (default) and `'verbose'` show them and do nothing else differently today. The docstrings no longer promise detailed logging.
- Load errors no longer name the 0.8.x classes: `regressors.load` on a classifier says
  `Cannot load a regressor: '...' holds a 'xgboost' classifier (task 'binary')`, and
  `Regressor.load` / `Classifier.load` on the other kind say
  `Cannot load model: Regressor.load expects a regressor, but ...`.
- `dataset_info(name, *, verbose=False)`: `verbose` is a keyword-only bool and the
  default prints nothing. The printed text's usage lines no longer pass `spark`. An
  unknown name raises only `ValidationError`; in 0.11 the error was also a
  `ValueError`.
- Every factory function's docstring lists all of its algorithm's parameters with their
  descriptions and defaults, taken from the algorithm's parameter table (a test keeps
  them equal). Every public class, method and function has an Args / Returns / Raises
  docstring.
- The package classifier is `Development Status :: 5 - Production/Stable`.
- **The hard dependencies are what smallaxe imports.** `plotly` is no longer installed
  with smallaxe (only `smallaxe.viz` imports it, when a chart is drawn); install
  `smallaxe[viz]` or plotly itself if your code used it through smallaxe. `numpy>=1.17.3` (imported by `smallaxe.search` and
  by `pyspark.ml`) and `py4j>=0.10.9.5` (smallaxe catches its `Py4JJavaError`) are now
  declared; PySpark already installed both. `pandas` stays required: from Spark 3.4
  `pyspark.ml.functions`, which `smallaxe.preprocessing` (the Scaler) and binary
  `Classifier.evaluate` import, imports it. The
  `xgboost` extra also declares `pandas>=1.2`, which `xgboost.spark` imports.
- **`import smallaxe.search` works in a fresh Python 3.8 or 3.9 environment.** There the
  newest hyperopt is 0.2.7, which imports `pkg_resources`, and setuptools 82 removed it;
  smallaxe now requires `setuptools>=56.0.0,<82` on Python 3.8 and 3.9 (pip downgrades
  a newer setuptools there). On Python 3.10 and later it requires `hyperopt>=0.3.0`,
  which does not import `pkg_resources`.
- **The type annotations are checked strictly.** smallaxe ships `py.typed`, and
  `mypy smallaxe` now passes with untyped and partially typed definitions disallowed,
  `check_untyped_defs`, `no_implicit_optional`, `warn_return_any`, `warn_redundant_casts`
  and `warn_unused_ignores` (CI blocks on it, with mypy pinned to 2.4.0). Private
  annotations that were wrong are fixed (`Scaler`'s fitted model is Optional until
  `fit`; the internal JSON reader returns whatever the file holds); no public
  signature changed. mypy 2 checks Python 3.10 and later only, so the configured target
  is 3.10; the code stays Python 3.8-compatible and is tested there.

### Fixed

- **Fitting a model on a DataFrame with no rows raises `ValidationError`.** `Model.fit`,
  `optimize.run` and `AutomatedTraining.fit` used to pass the empty
  data to the estimator, which failed with its own error (XGBoost: `IndexError`).
- **`optimize.run(metric="")` raises `ValidationError`.** A falsy metric that is not
  `None` (`""`, `0`) was read as "use the default metric".

### Added

- **Spark 4.x support: PySpark 4.0, 4.1, 4.2 and 4.3.** smallaxe now requires
  `pyspark>=3.3,<4.4` (was `<4.0`); the bound is one past the newest minor CI tests.
  CI adds legs on Java 17 for PySpark 4.0 on Python 3.9, 3.10, 3.11 and 3.12, 4.1 and
  4.2 on Python 3.12, and 4.3 on Python 3.11 and 3.12, plus 4.3 / Python 3.12 on Java
  25, and runs XGBoost on 4.3 / Python 3.12; the Spark 4 legs install pandas below 3,
  as Spark 4 documents, except one 4.3 / Python 3.12 leg that installs pandas 3, which a
  plain `pip install` resolves on Python 3.11 and later. Spark 4 needs Java 17 or later and Python 3.9 or later (4.1 and
  4.2: 3.10; 4.3: 3.11). Limits:
  - **Classic sessions only.** Spark Connect is not supported; a Connect session raises
    `ConfigurationError` (see "Changed").
  - **XGBoost** needs xgboost 3.3 or later on Spark 4, the first release that declares
    PySpark 4 support (Python 3.12+). The `smallaxe[xgboost]` floor stays 1.7.0, for
    Spark 3.3: an extra cannot depend on the installed PySpark.
  - **LightGBM and CatBoost** are validated on Spark 4.0 only (Databricks Runtime 17.3
    LTS, Scala 2.13), with `synapseml-lightgbm_2.13:1.1.3-spark4.0` and
    `catboost-spark_4.0_2.13:1.2.10`. Their Spark 4.1 builds are not validated, and
    neither publishes one for Spark 4.2 or 4.3.
  - Python 3.13 and 3.14, which PySpark 4 supports, are not declared (smallaxe stays
    `<3.13`).

  The PySpark APIs smallaxe imports exist unchanged in PySpark 4.3, PySpark 4 no longer
  imports `distutils` (the `setuptools` requirement on Python 3.12 is for PySpark 3.5),
  and results no longer depend on ANSI SQL mode, which Spark 4 turns on (see
  "Changed"); `tests/test_ansi_mode.py` runs on every CI leg, the Spark 4 ones
  included.
- **`smallaxe.auto.AutomatedTraining`**: one call that trains every installed algorithm
  on the same data and the same validation rows, compares them on one metric and keeps
  the best fitted model. `AutomatedTraining(task, algorithms=None, metric=None,
  params=None, param_spaces=None, max_evals=20, seed=None)` supports
  `"simple_regression"` and `"binary"` (`"multiclass"` raises `ValidationError`); the
  default metric is `rmse` / `auc_roc`, as in `optimize.run`, and must be one of the
  task's `TASK_METRICS`. `fit(dataframe, label_col, feature_cols=None,
  exclude_cols=None, *, validation="train_test", test_size=0.2, n_folds=5,
  fold_key=None, cache_strategy=None)` resolves one split seed for the run (`seed`, else
  the global seed, else one random draw) and every model and search uses it, so all
  algorithms score on exactly the same rows; `validation="none"` is rejected. Each model
  comes from its factory with its fixed `params` and is refitted on every row; an
  algorithm with a `param_spaces` entry is tuned with `optimize.run` (`max_evals`
  trials). The arguments are checked once, with `Model.fit`'s checks, before anything
  trains; `params` / `param_spaces` naming an algorithm that is not requested, a
  parameter it does not have, or `seed` raise `ValidationError` in the constructor. An
  algorithm whose package is not installed is skipped with one `UserWarning` naming its
  install hint (all skipped: `DependencyError`); one that raises a smallaxe or JVM error
  while training is reported as failed with a `UserWarning` and the run continues (all
  failed: `ValidationError` carrying each error). After `fit`: `best_model`,
  `best_algorithm`, `best_score`, `models`, `comparison` (a pandas DataFrame, one row per
  algorithm, best first: `algorithm`, `status`, the metric, the task's other metrics,
  `fit_seconds`, `params`, `error`), and `predict`, `predict_proba` (binary) and
  `evaluate`, which use `best_model`. An undefined score (AUC on a one-class split) ranks
  last and never wins. There is no new artifact kind: save `best_model` and
  `comparison.to_csv(...)`. With the global verbosity above `'quiet'`, `fit` prints one
  progress line per algorithm.
- **`smallaxe.viz`: plotly charts for evaluating models on DataFrames of any size.**
  Every chart aggregates in Spark and collects only the aggregated rows, bounded by its
  bins or classes, never by the row count; each returns a `plotly.graph_objects.Figure`
  (it never calls `show()`) and takes `title=` to replace its default title. Rows with a
  null or NaN value in a column a chart reads are left out and not counted. Column names
  holding a dot or a backtick are read as that one column, not as struct access.
  - `confusion_matrix(dataframe, label_col, prediction_col="predict_label", *,
    normalize=False, title=None)`: one `groupBy(label, prediction).count()`, binary or
    multiclass. Actual classes on y, predicted on x, ascending; every class in either
    column is on both axes and an absent pair is 0. `normalize=True` divides each row by
    its actual-class total and keeps the counts in the hover text. More than 50 distinct
    classes raises `ValidationError` (a continuous label passed by mistake).
  - `roc_curve(dataframe, label_col, probability_col="probability", *, n_bins=200,
    title=None)` and `precision_recall_curve(...)` (same arguments): binary 0/1 labels;
    the score is element 1 of `predict_proba`'s vector (length 2) or a numeric column of
    any numeric type (float, int, decimal: cast to double for the exact area) in [0, 1]. One `groupBy` over `n_bins` equal-width score bins (an int in [10, 10000]),
    accumulated on the driver from the highest bin down; ROC runs from (0, 0) to (1, 1)
    with a chance diagonal, PR starts at recall 0 with the first point's precision (as
    Spark's curve does) and shows the positive-rate baseline. The legend and title carry
    the exact `smallaxe.metrics.auc_roc` / `auc_pr` to 4 decimals. A one-class input,
    labels other than 0/1, vectors of another length and scores outside [0, 1] raise
    `ValidationError`.
  - `residuals(dataframe, label_col, prediction_col="predict_label", *, n_bins=50,
    title=None)`: a histogram of `label - prediction` in `n_bins` (an int in [1, 10000])
    equal-width bins between the smallest and largest residual (one bar when they are
    equal), with a line at 0 and the mean and standard deviation in the subtitle.
    Rows whose residual is infinite or undefined (an infinite label or prediction) are
    left out and counted in the subtitle; a range too wide to bin in double precision
    raises `ValidationError`.
  - `actual_vs_predicted(dataframe, label_col, prediction_col="predict_label", *,
    n_bins=50, title=None)`: a 2-D density heatmap, actual on x and predicted on y, in
    `n_bins` x `n_bins` (an int in [5, 500]) equal-width cells over the joint range, with
    the y = x line. Rows with an infinite value are left out and counted in the subtitle.
  - `feature_importances(model, *, top_n=20, title=None)`: horizontal bars of a fitted
    model's `feature_importances` (or a fitted Pipeline's model step), largest at the
    top, the `top_n` largest. An unfitted model raises `ModelNotFittedError`; an
    algorithm that reports none, `ValidationError`.
  - `model_comparison(results, metric=None, *, title=None)`: one bar per model, best
    first by the metric's direction (the one `optimize.run` and `AutomatedTraining` use;
    a k-fold `mean_<metric>` key has its metric's), the best bar highlighted and value
    labels on the bars. `results` is a fitted `AutomatedTraining` (its `ok` comparison
    rows; `metric` defaults to its metric) or a mapping of model name to a mapping of
    metric name to score, such as `validation_scores` (`metric` required). Undefined
    scores are left out and named in the subtitle. An unfitted `AutomatedTraining`
    raises `ModelNotFittedError`; an empty mapping, or a metric missing from every
    entry, `ValidationError`.

  plotly stays optional: `import smallaxe.viz` does not import it, and a chart called
  without it raises `DependencyError` naming `pip install smallaxe[viz]`. Wrong-typed or
  missing arguments raise `ValidationError` / `ColumnNotFoundError` before any Spark job.
- **`smallaxe[viz]`**, an extra holding `plotly>=5.0.0` for `smallaxe.viz`;
  `smallaxe[all]` includes it. The `dev` extra and `requirements-dev.txt` also install
  it, so the test suite draws every chart.
- **`Pipeline.fit` and `Pipeline.fit_transform` take the model's validation options**
  as keyword-only arguments: `validation`, `test_size`, `n_folds`, `fold_key`, `refit`
  and `cache_strategy`, with `Model.fit`'s defaults. They reach the model step, which
  validates on the preprocessed frame. The preprocessing steps are fitted on all rows
  first, so their statistics (imputed values, scaling ranges, encoder categories) see
  the validation rows; the docstring says so. On a pipeline without a model, any of them
  given raises `ValidationError`; a value counts as given unless it equals its default
  and has the default's type, so `refit=1` and `n_folds=5.0` raise while an explicit
  `refit=True` does not (it cannot be told from leaving it out). A duck-typed model
  receives only the options given in that sense.
- **`Pipeline.validation_scores`**: the model step's `validation_scores`, or None for a
  pipeline without a model. It survives `save` / `load`.
- **`load_sample_regression()` / `load_sample_classification()` default to the active
  SparkSession**: `spark` is optional, and with no active session they raise
  `ValidationError`. A non-positive or non-integer `n_rows`, or a non-integer `seed`,
  raises `ValidationError`.
- A "Stability and versioning" section in the README: what is public, how long
  artifacts keep loading, how deprecations work, and the supported Python / Spark
  matrix.

## 0.11.0 (2026-11-25)

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
- **A duck-typed preprocessing step is fitted as `fit(dataframe, numerical_cols=...,
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
- `model.evaluate(dataframe, label_col=None)` scores a fitted model on any DataFrame with the
  metrics its validation computes for its task (the same keys as `validation_scores`,
  without `validation_type` / `test_size` / fold bookkeeping), undefined metrics as
  `None`. `label_col` defaults to the label the model was fitted with. It raises
  `ModelNotFittedError` before `fit` and `ColumnNotFoundError` when the label or a
  feature column is missing.
- `Encoder.output_cols`: the columns `transform` writes for the encoded features, in
  order (the encoded columns themselves for `'label'`, the generated columns for
  `'onehot'`).
- `Pipeline.predict_proba(dataframe, output_col="probability")` and
  `Pipeline.evaluate(dataframe, label_col=None)` apply the preprocessing steps and delegate to
  the model; `Pipeline.predict` gains `output_col` (default `"predict_label"`, as
  `model.predict`). They raise `ValidationError` on a pipeline without a model, and
  `predict_proba` does on a regressor. `pipeline.feature_cols` and
  `pipeline.skipped_steps` report what `fit` decided.
- `smallaxe.load_pipeline(path)` loads a saved pipeline, and `smallaxe.load(path)` loads
  any smallaxe artifact (model, pipeline, Imputer, Scaler, Encoder, including the
  unversioned model, pipeline, Imputer and Scaler layouts of 0.10 and earlier), choosing
  by the kind the artifact records. `load_model` and `load_pipeline` reject other kinds.
  `import smallaxe` still imports neither.
- `model.set_params(**params)`, `regressors.<algorithm>()` / `classifiers.<algorithm>()`
  and their `load`, `algorithms` and `available_models`, `dataset_info(...)` returning a
  dict and its `verbose` argument: see Deprecated below.

### Deprecated

0.11.0 is the release that warns: every spelling below still works exactly as in 0.10,
emits one `DeprecationWarning` that names its replacement and says it is removed in
smallaxe 1.0, attributed to the caller's line, and is removed in 1.0. smallaxe's own
code, docs and examples use only the replacements.

- `model.set_param({...})`: use `model.set_params(**params)`, which validates the same
  way and returns the model.
- The `df` argument: every public callable that takes a DataFrame now calls it
  `dataframe` (`optimize.run` already did). Positional calls are unaffected; `df=` by
  keyword still works and warns, and passing both `df=` and `dataframe` raises
  `ValidationError`. Renamed in `Model.fit` / `predict` / `evaluate`,
  `Classifier.predict_proba`, `Pipeline.fit` / `transform` / `fit_transform` /
  `predict` / `predict_proba` / `evaluate`, `Imputer` / `Scaler` / `Encoder` `fit` /
  `transform` / `fit_transform`, and every function in `smallaxe.metrics` (`mse`,
  `rmse`, `mae`, `r2`, `mape`, `accuracy`, `precision`, `recall`, `f1_score`,
  `multiclass_scores`, `auc_roc`, `auc_pr`, `log_loss`).
- `stratified=` in `Model.fit` and `optimize.run`: it has no effect (splits hash each
  row's content, which is label-independent, so every class splits in the same
  proportion in expectation) and is removed in 1.0. Passing any value other than None
  warns; results are unchanged.
- `datasets.dataset_info(name)` printing by default: it now also returns the
  description as a dict (the 1.0 contract). Pass `verbose=False` to only get the dict,
  or `verbose=True` to keep printing; neither warns. An unknown name raises
  `ValidationError` instead of `ValueError`; in 0.11 the error is also a `ValueError`,
  so existing `except ValueError` keeps working until 1.0.
- `Regressors` / `Classifiers`: use the module functions `regressors.random_forest()`,
  `regressors.xgboost()`, `regressors.lightgbm()`, `regressors.catboost()`,
  `regressors.load(path)` and `regressors.available_models()`, and the same in
  `classifiers` (constructors keep `task=`), after
  `from smallaxe.training import regressors, classifiers`. Every static method of the
  two classes warns and delegates; importing the classes does not warn.
  `Regressors.list_models()` / `Classifiers.list_models()` (the 0.8.x class names of
  the installed algorithms): use `regressors.algorithms()` (all algorithm names) and
  `regressors.available_models()` (which are installed). The module
  `available_models()` has no `class_name` entry; the deprecated class method still
  returns it.
- `metrics.log_loss(dataframe, label_col, probability_col, eps)` with `eps`
  positional: `eps` is keyword-only (`log_loss(..., eps=1e-6)`).
- Names that become private, with what to use instead. Each old name still imports
  (module `__getattr__`) and warns when read:
  - `smallaxe.training.BaseModel` / `BaseRegressor` / `BaseClassifier`: use
    `smallaxe.training.Model` / `Regressor` / `Classifier`. They are no longer in
    `smallaxe.training.__all__`; the classes stay in `smallaxe.training.base`.
  - `smallaxe.training.base.infer_feature_cols`, `warn_ignored_cols`,
    `check_training_data`: `Model.fit` does each of these.
  - `smallaxe.training.mixins.validation_mixin.hashable_columns`, `split_hash`: use
    `fit(validation=..., fold_key=...)`; `resolve_split_seed`: use `smallaxe.set_seed()`
    or the model's `seed`.
  - `smallaxe.training.algorithm.resolve`, `estimator_kwargs`, `legacy_class_name`,
    `package_version`, `COLUMN_ROLES`, `LEGACY_CLASS_NAMES`. `names`, `get`,
    `Algorithm`, `Param`, `Dependency` and the task constants stay public.
  - `smallaxe.search.optimize.HYPEROPT_INSTALL_HINT`, `MAXIMIZE_METRICS`,
    `MINIMIZE_METRICS`, `VALID_METRICS` (use `smallaxe.metrics.TASK_METRICS`), and
    `HYPEROPT_AVAILABLE` in `smallaxe.search` and `smallaxe.search.optimize` (hyperopt
    is a required dependency, so it is always True). `smallaxe.search.__all__` no longer
    lists it.
  - `smallaxe.training.catboost.is_catboost_available`, `catboost_install_hint`, and
    `XGBOOST_AVAILABLE`, `LIGHTGBM_AVAILABLE`, `CATBOOST_AVAILABLE` in the algorithm
    modules: use `smallaxe.training.algorithm.get(name).available()` or
    `regressors.available_models()`.
  - `VALID_METHODS` in `smallaxe.preprocessing.encoder` and `scaler`, and
    `VALID_HANDLE_RARE` in `encoder`.
- The earlier deprecations (the eight 0.8.x class aliases, `set_cache_strategy` /
  `get_cache_strategy`, `set_spark_session` / `get_spark_session`) are unchanged, and
  their messages now all say "removed in smallaxe 1.0"; the class aliases name the
  `regressors` / `classifiers` function that replaces them.

### Fixed

- **smallaxe's warnings point at the caller's line.** The feature-inference, caching,
  k-fold undefined-metric, legacy multiclass-score and deprecation warnings are
  attributed to the first frame outside smallaxe, so one raised inside a pipeline or a
  search points at the user's `pipeline.fit(...)` / `optimize.run(...)` line instead of
  a line in smallaxe.
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
- **`fit` and `optimize.run` work on Spark 3.3 again.** 0.10.0 compared the DataFrame's
  storage level with `StorageLevel.NONE`, which Spark only has from 3.4, so every `fit`
  on Spark 3.3 raised `AttributeError`. The check now reads the storage level's fields.
  Pull-request CI now runs on Spark 3.3 as well as 3.5.
  0.10.1 was never released; this fix first ships in 0.11.0.

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
