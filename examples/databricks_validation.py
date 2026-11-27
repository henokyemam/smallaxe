# Databricks notebook source
# MAGIC %md
# MAGIC # smallaxe end-to-end check (Databricks)
# MAGIC
# MAGIC A short check of each algorithm on a Databricks cluster: fit, validate and predict for
# MAGIC regression and binary classification, and a hyperparameter search. Each optional algorithm
# MAGIC runs if its package is present and is marked SKIP otherwise (for example LightGBM on a Spark 3
# MAGIC runtime with Scala 2.13, for which SynapseML has no build). The Maven libraries each validated
# MAGIC runtime needs are listed in `databricks_all_algorithms_validation.py`; the `catboost_spark`
# MAGIC Python module comes from the catboost-spark Maven library, not from PyPI.

# COMMAND ----------

# MAGIC %pip install --quiet "git+https://github.com/henokyemam/smallaxe.git@main" xgboost catboost hyperopt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# Databricks source notebook: imports sit in later cells (E402), and the runtime
# provides spark, dbutils and display (F821).
# ruff: noqa: E402, F821
from hyperopt import hp

import smallaxe
from smallaxe.search import optimize
from smallaxe.training import classifiers, regressors

smallaxe.set_seed(42)
results = {}  # name -> "PASS" | "SKIP: reason" | "FAIL: reason"


def run_check(name, fn):
    from smallaxe.exceptions import DependencyError

    try:
        fn()
        results[name] = "PASS"
        print(f"PASS  {name}")
    except DependencyError as exc:
        results[name] = f"SKIP: {exc}"
        print(f"SKIP  {name}: {exc}")
    except Exception as exc:  # noqa: BLE001
        results[name] = f"FAIL: {type(exc).__name__}: {exc}"
        print(f"FAIL  {name}: {exc}")

# COMMAND ----------

# MAGIC %md ## Availability report

# COMMAND ----------

print("Spark:", spark.version)
print("Regressors :", regressors.available_models())
print("Classifiers:", classifiers.available_models())

# COMMAND ----------

# MAGIC %md ## Build real datasets (Kaggle Titanic + King County)

# COMMAND ----------

import pandas as pd
from pyspark.sql import functions as F

TITANIC = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
HOUSE = "https://raw.githubusercontent.com/Shreyas3108/house-price-prediction/master/kc_house_data.csv"

# Read on the driver via pandas, then distribute with createDataFrame. This avoids
# a shared-filesystem dependency: a local file:/tmp path is not visible to executors
# on a multi-node cluster.
tdf = spark.createDataFrame(pd.read_csv(TITANIC))
am = tdf.approxQuantile("Age", [0.5], 0.0)[0]
tdf = (
    tdf.withColumn("sex_code", F.when(F.col("Sex") == "male", 1).otherwise(0))
    .withColumn("Age", F.when(F.col("Age").isNull(), am).otherwise(F.col("Age")))
    .select("Survived", "Pclass", "sex_code", F.col("Age").cast("double").alias("Age"),
            "SibSp", "Parch", F.col("Fare").cast("double").alias("Fare"))
).cache()
TFEAT = ["Pclass", "sex_code", "Age", "SibSp", "Parch", "Fare"]

hfeat = ["bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "grade"]
hdf = spark.createDataFrame(pd.read_csv(HOUSE))
hdf = hdf.select(F.col("price").cast("double").alias("price"),
                 *[F.col(c).cast("double").alias(c) for c in hfeat]).cache()
print("titanic rows:", tdf.count(), "| house rows:", hdf.count())

# COMMAND ----------

# MAGIC %md ## Per-algorithm: fit -> validate -> predict (regression + binary)

# COMMAND ----------

def check_regressor(make):
    m = make()
    m.fit(hdf, label_col="price", feature_cols=hfeat, validation="train_test", test_size=0.2)
    assert m.validation_scores["r2"] is not None
    assert m.predict(hdf).count() == hdf.count()
    print("   r2:", round(m.validation_scores["r2"], 4))


def check_classifier(make):
    m = make()
    m.fit(tdf, label_col="Survived", feature_cols=TFEAT, validation="train_test", test_size=0.2)
    assert m.validation_scores["accuracy"] is not None
    assert m.predict(tdf).count() == tdf.count()
    print("   accuracy:", round(m.validation_scores["accuracy"], 4))


run_check("RF regressor", lambda: check_regressor(lambda: regressors.random_forest(n_estimators=50, max_depth=8, seed=42)))
run_check("RF classifier", lambda: check_classifier(lambda: classifiers.random_forest(task="binary", n_estimators=50, max_depth=6, seed=42)))
run_check("XGBoost regressor", lambda: check_regressor(lambda: regressors.xgboost(n_estimators=80, max_depth=5, seed=42)))
run_check("XGBoost classifier", lambda: check_classifier(lambda: classifiers.xgboost(task="binary", n_estimators=80, max_depth=5, seed=42)))
run_check("LightGBM regressor", lambda: check_regressor(lambda: regressors.lightgbm(n_estimators=80, max_depth=6, seed=42)))
run_check("LightGBM classifier", lambda: check_classifier(lambda: classifiers.lightgbm(task="binary", n_estimators=80, max_depth=6, seed=42)))
run_check("CatBoost regressor", lambda: check_regressor(lambda: regressors.catboost(n_estimators=80, max_depth=6, seed=42)))
run_check("CatBoost classifier", lambda: check_classifier(lambda: classifiers.catboost(task="binary", n_estimators=80, max_depth=6, seed=42)))

# COMMAND ----------

# MAGIC %md ## Hyperparameter search

# COMMAND ----------

def search_rf():
    r = optimize.run(regressors.random_forest(seed=42), hdf, label_col="price", feature_cols=hfeat,
                     param_space={"n_estimators": hp.quniform("n_estimators", 40, 120, 20),
                                  "max_depth": hp.quniform("max_depth", 5, 12, 1)},
                     metric="rmse", validation="train_test", max_evals=6, seed=42, verbose=False)
    assert r.best_model is not None and r.best_score > 0
    print("   RF best rmse:", round(r.best_score, 2), r.best_params)


def search_xgb():
    r = optimize.run(classifiers.xgboost(task="binary", seed=42), tdf, label_col="Survived", feature_cols=TFEAT,
                     param_space={"n_estimators": hp.quniform("n_estimators", 40, 120, 20),
                                  "learning_rate": hp.uniform("learning_rate", 0.03, 0.3)},
                     metric="auc_roc", validation="train_test", max_evals=6, seed=42, verbose=False)
    assert r.best_model is not None
    print("   XGBoost best auc_roc:", round(r.best_score, 4), r.best_params)


run_check("search: RF (rmse)", search_rf)
run_check("search: XGBoost (auc_roc)", search_xgb)

# COMMAND ----------

# MAGIC %md ## Summary

# COMMAND ----------

print("=" * 70)
for k, v in results.items():
    tag = v.split(":")[0]
    print(f"{tag:>5}  {k}" + ("" if v == "PASS" else f"   ({v})"))
print("=" * 70)
n_pass = sum(1 for v in results.values() if v == "PASS")
n_skip = sum(1 for v in results.values() if v.startswith("SKIP"))
n_fail = sum(1 for v in results.values() if v.startswith("FAIL"))
print(f"{n_pass} passed, {n_skip} skipped, {n_fail} failed (of {len(results)} checks)")

import json as _json

dbutils.notebook.exit(
    _json.dumps({"summary": f"{n_pass} passed, {n_skip} skipped, {n_fail} failed", "results": results})
)
