# Gotchas

Behaviour of smallaxe 1.0 that can surprise you, and what to do about it. Each item says
what happens and what to do.

## Data and features

### Feature inference uses every numeric and boolean column

Without `feature_cols`, `fit` and `optimize.run` use every numeric and boolean column
except the label and `exclude_cols`. That includes numeric IDs, timestamps stored as
numbers, row indices, and a `predict_label` left over from an earlier `predict`. Other
columns (strings, dates) are left out, named in one `UserWarning` per `fit` or search,
and recorded in `metadata["ignored_cols"]`.

Do: pass `feature_cols`, or list the unwanted columns in `exclude_cols`. Encode string
columns with an `Encoder` (or use a `Pipeline`) if they should be features.

### Random Forest rejects missing feature values; the boosting algorithms keep them

`fit` checks the training data once. With Random Forest, a null, NaN or infinite feature
value raises `ValidationError` naming each column and its count. XGBoost, LightGBM and
CatBoost treat null and NaN as missing and keep the row. A null label raises
`ValidationError` for every algorithm. `predict` never drops a row; with Random Forest a
missing value fails when Spark computes the prediction.

Do: put an `Imputer` before a Random Forest. Not checked: what LightGBM and CatBoost
predict for NaN in a feature that had no NaN in training.

### CatBoost needs an Encoder for categorical columns

Features are assembled into one vector, which takes numbers only. A CatBoost pipeline
with `categorical_cols` and no `Encoder` raises `PreprocessingError` in `fit`, as for
every other algorithm. Native CatBoost categoricals are not implemented.

## Validation and reproducibility

### The model you keep is not the one the scores describe

With `validation="train_test"` or `"kfold"` and the default `refit=True`, the scores
come from models trained on the split, and the model you keep is then trained on every
row. The scores therefore describe a model that saw less data. With `refit=False` the
kept model is the last split's: the train split for `train_test`, the last fold for
`kfold`, whose mean score does not describe it.

Do: for a test set that no model trained on, split the data yourself,
`fit(train, validation="none")`, then `evaluate(test)`.

### A pipeline validates on data its preprocessing has already seen

`Pipeline.fit(validation=...)` fits the preprocessing steps on every row before the model
step is validated, so imputed values, scaling ranges and encoder categories include the
validation rows.

Do: for a strict holdout, split first, fit the pipeline on the training rows, and
`pipeline.evaluate(test)`.

### A seed fixes the splits, not every tree

A row's split or fold is a hash of its content (or of `fold_key`) and the seed, so the
same rows and seed give the same folds whatever the partitioning. Exact duplicate rows
land in the same fold. Spark's Random Forest training depends on the partitioning too: on
this checkout, the same data and seed in 3 and in 4 partitions gave different
predictions, and the same partitioning gave identical ones.

Do: to reproduce a model exactly, keep the data, the seed and the partitioning. Use
`fold_key` to keep all rows of one entity (a customer, a session) in one fold.

### Parameter values are checked for type in `set_params`, for range in `fit`

`set_params` and the factories check each value's type. A value of the right type but
out of range (Random Forest `max_depth=-3`) raises `ValidationError` only when `fit`
builds the Spark estimator. XGBoost checks its parameters inside the training job, so a
bad XGBoost value fails with Spark's `Py4JJavaError`.

## Spark versions and sessions

### Classic Spark sessions only

smallaxe saves and loads through the Hadoop FileSystem on the JVM. With a Spark Connect
session (`SparkSession.builder.remote(...)`, `spark.api.mode=connect`, or the
`pyspark-connect` / `pyspark-client` packages) every `save` and `load` raises
`ConfigurationError("smallaxe needs a classic Spark session; Spark Connect is not
supported")`.

Do: use a classic session, the default of `pip install pyspark`.

### ANSI SQL mode does not change smallaxe's results

Spark 4 turns `spark.sql.ansi.enabled` on by default. smallaxe never changes the setting,
and gives Spark 3's default-mode results with it on or off (`tests/test_ansi_mode.py`).
Your own Spark code around smallaxe is still subject to it: a failing cast or an overflow
raises instead of giving null.

### pandas 3 on Spark 4

Spark 4.0 to 4.3 document pandas below 3, and 4.2 and 4.3 warn on pandas 3. A plain
`pip install smallaxe` resolves pandas 3 on Python 3.11 and later. CI installs pandas
below 3 on its Spark 4 legs, except one leg (PySpark 4.3, Python 3.12) that runs the
whole suite with pandas 3.

Do: pin `pandas<3` to stay inside the range Spark documents.

### XGBoost on Spark 4 needs xgboost 3.3 and Python 3.12

xgboost 3.3 is the first release that declares PySpark 4 support, and it needs Python
3.12 or later. The `smallaxe[xgboost]` extra keeps its floor (1.7.0) for Spark 3.3, so
it does not enforce this. CI runs XGBoost on Spark 4 with PySpark 4.3 and Python 3.12.

Do: on Spark 4, use Python 3.12 and `pip install "xgboost>=3.3"`.

### XGBoost models saved before 0.11 have 100 trees

Releases before 0.11 trained XGBoost's default 100 rounds whatever `n_estimators` said.
Those models keep their 100 rounds when loaded in 1.0.

Do: refit them to get the number of rounds `n_estimators` asks for.

## LightGBM and CatBoost on clusters

### They need JVM packages built for your Spark and Scala

LightGBM needs SynapseML's `synapseml-lightgbm` package and CatBoost needs
`catboost-spark`, each built for one Spark and Scala version, plus their native
libraries. They are validated on Databricks with these builds:

- Spark 3.5, Scala 2.12: `com.microsoft.azure:synapseml-lightgbm_2.12:1.1.3` and
  `ai.catboost:catboost-spark_3.5_2.12:1.2.10`.
- Spark 4.0, Scala 2.13: `com.microsoft.azure:synapseml-lightgbm_2.13:1.1.3-spark4.0`
  (repository `https://mmlspark.azureedge.net/maven`) and
  `ai.catboost:catboost-spark_4.0_2.13:1.2.10`.

SynapseML has no Scala 2.13 build for Spark 3, so LightGBM is not available on Spark 3
runtimes with Scala 2.13. The Spark 4.1 builds of both packages are not validated, and
neither project publishes a build for Spark 4.2 or 4.3. CI does not run either algorithm.

### Worker failures can be intermittent

CatBoost-Spark and SynapseML run native code on the executors. A fit can fail with a
worker error, such as CatBoost's "Error while executing workers", that does not recur
when the same fit runs again. In the validation runs, one CatBoost regression fit failed
once with that error and then passed 20 of 20 repeats. The cause is not established.

Do: retry a fit that fails with a worker error before treating it as a bug. Use a
fixed-size cluster: the validation runs used one, and autoscaling is not validated.
`AutomatedTraining` reports such a failure as `failed` in its comparison and goes on with
the other algorithms.

## AutomatedTraining

### What AutomatedTraining does not do

- It supports `simple_regression` and `binary`. `task="multiclass"` raises
  `ValidationError`.
- It rejects `validation="none"`: the comparison needs held-out rows.
- Like `fit`, it trains on numeric and boolean features only. Impute and encode other
  columns first.
- An algorithm tuned through `param_spaces` reports only the compared metric in
  `comparison`.
- It has no artifact of its own. Save `best_model` and `comparison.to_csv(...)`.
- Only smallaxe errors and JVM errors are caught and reported as `failed`. Any other
  exception stops the run.
- If every requested algorithm is missing its package, `fit` raises `DependencyError`;
  if every one fails, `ValidationError` carrying each error.

## Charts

### Charts need plotly

`pip install smallaxe` does not install plotly. `import smallaxe.viz` works without it,
and calling a chart raises `DependencyError` naming `pip install smallaxe[viz]`.

### What a chart leaves out

Rows with a null or NaN label, prediction or score are left out of every chart and not
counted. The regression charts also leave out rows with an infinite value and say how
many in the subtitle. `confusion_matrix` raises `ValidationError` for more than 50
classes. The ROC and precision-recall curves take binary 0/1 labels only.

## Persistence and search

### Paths are strings

`save` and `load` take a string path (local, `dbfs:/`, `s3://`, `hdfs://`). A
`pathlib.Path` raises `ValidationError`; pass `str(path)`.

### Pickles are not loaded

Pipelines are saved without pickle. The one pickle smallaxe reads is the `Encoder` inside
a pipeline saved by 0.10, through an unpickler that resolves only the `Encoder` class. Any
other pickled step raises `ValidationError`.

### `optimize.run` takes a model, not a Pipeline

A search over preprocessing and model together is not supported. Impute and encode the
frame first, then search over the model.
