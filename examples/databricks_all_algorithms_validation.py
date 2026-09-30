# Databricks notebook source
# MAGIC %md
# MAGIC # smallaxe — all-algorithm validation on real mixed-type data
# MAGIC
# MAGIC Exercises every algorithm (Random Forest, XGBoost, LightGBM, CatBoost) on real datasets that contain **both
# MAGIC numeric and categorical** features:
# MAGIC
# MAGIC | Task | Dataset | Rows | Numeric | Categorical | Missing values |
# MAGIC |---|---|---|---|---|---|
# MAGIC | Regression | Diamonds (Kaggle `shivam2503/diamonds`) | 53,940 | 6 | 3 | none |
# MAGIC | Binary | Telco Customer Churn (Kaggle `blastchar`) | 7,043 | 4 | 15 | `TotalCharges` (numeric) |
# MAGIC | Binary (2nd) | Adult Census Income (Kaggle `uciml`) | 32,561 | 6 | 8 | `workclass`, `occupation`, `native_country` (categorical) |
# MAGIC | Multiclass (7) | Covertype, 60k sample, one-hot groups collapsed to categoricals | 60,000 | 10 | 2 (4 + 40 levels) | none |
# MAGIC
# MAGIC Checks per algorithm: full `Pipeline` (Imputer → Scaler → Encoder → model), row preservation,
# MAGIC smallaxe metrics cross-checked against scikit-learn, `train_test` / stratified `kfold` validation,
# MAGIC `predict_proba`, `feature_importances`, save/load round-trip, `search.optimize`, label encoding,
# MAGIC plus probes for known gaps (pipeline persistence, native categoricals, null-row dropping, unseen categories).
# MAGIC
# MAGIC **Runtime requirement:** DBR 16.4 LTS **Scala 2.12** (Spark 3.5.2) with Maven libraries
# MAGIC `com.microsoft.azure:synapseml-lightgbm_2.12:1.1.3` and `ai.catboost:catboost-spark_3.5_2.12:1.2.10`.
# MAGIC SynapseML has no Scala 2.13 build, and smallaxe pins `pyspark<4.0`, so this is the only LTS
# MAGIC runtime that can host all four algorithms at once.
# MAGIC
# MAGIC **Install:** the next cell installs smallaxe from `main` (it contains the SynapseML/CatBoost
# MAGIC compatibility fix); switch to `smallaxe` from PyPI once a release after 0.8.0 is published. The
# MAGIC `synapseml` pip version must match the `synapseml-lightgbm` jar version.
# MAGIC
# MAGIC **Cluster must be fixed-size** (no autoscaling): CatBoost-Spark training fails when executors join or leave mid-fit.
# MAGIC
# MAGIC Result statuses: PASS, FAIL, SKIP (dependency missing), BLOCKED (a prerequisite check failed), and GAP
# MAGIC (a known library gap listed in `Goals.md`; the probe passes once the gap is fixed). Validated on
# MAGIC 2026-09-30 (16.4 LTS Scala 2.12, 2 fixed m5d.8xlarge workers): 62 PASS / 21 GAP / 1 FAIL.

# COMMAND ----------

# MAGIC %pip install --quiet "git+https://github.com/henokyemam/smallaxe.git@main" xgboost hyperopt synapseml==1.1.3

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import gzip
import importlib
import io
import json
import math
import sys
import time
import traceback
import urllib.request
import uuid
import zipfile
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd
import pyspark
from hyperopt import hp
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F
from sklearn import metrics as SK

import smallaxe
from smallaxe import metrics as M
from smallaxe.exceptions import DependencyError
from smallaxe.pipeline import Pipeline
from smallaxe.preprocessing import Encoder, Imputer, Scaler
from smallaxe.search import optimize
from smallaxe.training import Classifiers, Regressors

SEED = 42
smallaxe.set_seed(SEED)
smallaxe.set_verbosity("quiet")
RUN_ID = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
ART = f"dbfs:/tmp/smallaxe_validation/{RUN_ID}"  # Hadoop-FS path used for model artifacts
ALGOS = ["random_forest", "xgboost", "lightgbm", "catboost"]
results = []


class Blocked(Exception):
    """A prerequisite check failed, so this one cannot run."""


def step(section, algo, task, check, fn, probe=False):
    """Run one check and record PASS / FAIL / SKIP (or GAP for known-gap probes)."""
    t0 = time.time()
    row = {"section": section, "algo": algo, "task": task, "check": check}
    try:
        out = fn()
        row["status"] = "PASS"
        row["detail"] = out if isinstance(out, (dict, str)) or out is None else str(out)
    except DependencyError as exc:
        row["status"], row["detail"] = "SKIP", str(exc)[:300]
    except Blocked as exc:
        row["status"], row["detail"] = "BLOCKED", str(exc)[:300]
    except Exception as exc:  # noqa: BLE001
        row["status"] = "GAP" if probe else "FAIL"
        row["detail"] = f"{type(exc).__name__}: {str(exc)[:500]}"
        row["trace"] = traceback.format_exc()[-2000:]
    row["secs"] = round(time.time() - t0, 1)
    results.append(row)
    print(
        f"{row['status']:>4} [{row['secs']:>6}s] {section} | {algo} | {task} | {check} :: {str(row['detail'])[:260]}"
    )
    return row


def ver(mod):
    try:
        return getattr(importlib.import_module(mod), "__version__", "present")
    except Exception as exc:  # noqa: BLE001
        return f"missing ({type(exc).__name__})"


ENV = {
    "run_id": RUN_ID,
    "dbr": spark.conf.get("spark.databricks.clusterUsageTags.sparkVersion", "?"),
    "spark": spark.version,
    "scala": spark.sparkContext._jvm.scala.util.Properties.versionNumberString(),
    "python": sys.version.split()[0],
    "pyspark_file": pyspark.__file__,
    "smallaxe": smallaxe.__version__,
    "xgboost": ver("xgboost"),
    "synapseml": ver("synapse.ml"),
    "catboost": ver("catboost"),
    "catboost_spark": ver("catboost_spark"),
    "hyperopt": ver("hyperopt"),
    "sklearn": ver("sklearn"),
    "executors": spark.sparkContext._jsc.sc().getExecutorMemoryStatus().size() - 1,
    "default_parallelism": spark.sparkContext.defaultParallelism,
    "regressors_available": {k: v["available"] for k, v in Regressors.available_models().items()},
    "classifiers_available": {k: v["available"] for k, v in Classifiers.available_models().items()},
}
print(json.dumps(ENV, indent=1))

# COMMAND ----------

# MAGIC %md ## Load real datasets (driver-side pandas → distributed DataFrames)

# COMMAND ----------


def to_spark(pdf, num_cols):
    """pandas -> Spark, turning numeric NaN into real nulls (as they would be in a table)."""
    sdf = spark.createDataFrame(pdf)
    for c in num_cols:
        sdf = sdf.withColumn(
            c, F.when(F.isnan(F.col(c).cast("double")), None).otherwise(F.col(c).cast("double"))
        )
    return sdf


# Regression: Diamonds
dia = pd.read_csv("https://raw.githubusercontent.com/tidyverse/ggplot2/main/data-raw/diamonds.csv")
REG = NS(
    key="regression",
    name="diamonds",
    task="simple_regression",
    label="price",
    num=["carat", "depth", "table", "x", "y", "z"],
    cat=["cut", "color", "clarity"],
)
dia["price"] = dia["price"].astype(float)
REG.sdf = to_spark(dia[REG.num + REG.cat + ["price"]], REG.num)

# Binary: Telco Customer Churn (TotalCharges has blank strings -> numeric missing values)
tel = pd.read_csv(
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
)
tel["TotalCharges"] = pd.to_numeric(tel["TotalCharges"], errors="coerce")
tel["label"] = (tel["Churn"].str.strip() == "Yes").astype(int)
BIN = NS(
    key="binary",
    name="telco_churn",
    task="binary",
    label="label",
    num=["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"],
    cat=[
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
    ],
)
BIN.sdf = to_spark(tel[BIN.num + BIN.cat + ["label"]], BIN.num)
BIN.raw_with_id = to_spark(tel[["customerID"] + BIN.num + BIN.cat + ["label"]], BIN.num)

# Binary #2: Adult Census Income ('?' -> categorical missing values)
ADULT_COLS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]
adult = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
    header=None,
    names=ADULT_COLS,
    na_values="?",
    skipinitialspace=True,
)
adult["label"] = (adult["income"].astype(str).str.strip() == ">50K").astype(int)
BIN2 = NS(
    key="binary_adult",
    name="adult_income",
    task="binary",
    label="label",
    num=["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"],
    cat=[
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country",
    ],
)
BIN2.sdf = to_spark(adult[BIN2.num + BIN2.cat + ["label"]], BIN2.num)

# Multiclass: Covertype (7 classes). The 4 wilderness + 40 soil one-hot blocks are collapsed back
# into two categorical string columns so the Encoder has real work (incl. rare levels).
raw = urllib.request.urlopen("https://archive.ics.uci.edu/static/public/31/covertype.zip").read()
with zipfile.ZipFile(io.BytesIO(raw)) as z:
    cov = pd.read_csv(
        io.BytesIO(gzip.decompress(z.read([n for n in z.namelist() if n.endswith(".gz")][0]))),
        header=None,
    )
COV_NUM = [
    "elevation",
    "aspect",
    "slope",
    "h_dist_hydro",
    "v_dist_hydro",
    "h_dist_road",
    "hillshade_9am",
    "hillshade_noon",
    "hillshade_3pm",
    "h_dist_fire",
]
covdf = pd.DataFrame(cov.iloc[:, :10].values.astype(float), columns=COV_NUM)
covdf["wilderness_area"] = [f"W{i + 1}" for i in cov.iloc[:, 10:14].values.argmax(1)]
covdf["soil_type"] = [f"S{i + 1:02d}" for i in cov.iloc[:, 14:54].values.argmax(1)]
covdf["label"] = cov.iloc[:, 54].astype(int) - 1  # Spark needs labels in [0, n_classes)
covdf = covdf.sample(n=60000, random_state=SEED).reset_index(drop=True)
MULTI = NS(
    key="multiclass",
    name="covertype_60k",
    task="multiclass",
    label="label",
    num=COV_NUM,
    cat=["wilderness_area", "soil_type"],
)
MULTI.sdf = to_spark(covdf, COV_NUM)

DATASETS = [REG, BIN, BIN2, MULTI]
for ds in DATASETS:
    ds.train, ds.test = ds.sdf.randomSplit([0.8, 0.2], seed=SEED)
    # Unseen-category behaviour is probed explicitly in section G; keep the algorithm matrix free of it
    # for columns the Encoder will NOT route to __OTHER__ (<= 30 levels).
    ds.dropped_unseen = 0
    for c in ds.cat:
        seen = [r[0] for r in ds.train.select(c).distinct().collect() if r[0] is not None]
        if len(seen) <= 30:
            before = ds.test.count()
            ds.test = ds.test.filter(
                F.col(c).isNull() | F.col(c).isin(seen)
            )  # keep real missing values
            ds.dropped_unseen += before - ds.test.count()
    ds.train, ds.test = ds.train.cache(), ds.test.cache()
    ds.n_train, ds.n_test = ds.train.count(), ds.test.count()
    nulls = (
        ds.sdf.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in ds.num + ds.cat])
        .first()
        .asDict()
    )
    ds.nulls = {k: v for k, v in nulls.items() if v}
    if ds.task != "simple_regression":
        dist = ds.test.groupBy(ds.label).count().toPandas()
        ds.n_classes = ds.sdf.select(ds.label).distinct().count()
        ds.baseline_acc = float(
            dist["count"].max() / dist["count"].sum()
        )  # majority-class accuracy
    print(
        f"{ds.name:>14}: train={ds.n_train} test={ds.n_test} (dropped unseen-category rows: {ds.dropped_unseen}) "
        f"num={len(ds.num)} cat={len(ds.cat)} nulls={ds.nulls}"
        + (
            f" classes={ds.n_classes} majority_acc={ds.baseline_acc:.3f}"
            if ds.task != "simple_regression"
            else ""
        )
    )

# COMMAND ----------

# MAGIC %md ## Shared helpers: model factory, preprocessing, metric cross-check vs scikit-learn

# COMMAND ----------

PARAMS = {
    "random_forest": dict(n_estimators=60, max_depth=8, seed=SEED),
    "xgboost": dict(n_estimators=80, max_depth=6, learning_rate=0.1, seed=SEED),
    "lightgbm": dict(n_estimators=80, max_depth=6, learning_rate=0.1, seed=SEED),
    "catboost": dict(n_estimators=80, max_depth=6, learning_rate=0.1, seed=SEED),
}


def make(algo, ds, **overrides):
    params = {**PARAMS[algo], **overrides}
    if ds.task == "simple_regression":
        return getattr(Regressors, algo)(**params)
    return getattr(Classifiers, algo)(task=ds.task, **params)


def prep_steps(encoder_method="onehot"):
    return [
        ("imputer", Imputer(numerical_strategy="median", categorical_strategy="most_frequent")),
        ("scaler", Scaler(method="standard")),
        # max_categories=30 routes rare/unseen levels to __OTHER__ on Adult (41 countries) and Covertype (40 soils)
        ("encoder", Encoder(method=encoder_method, max_categories=30)),
    ]


def close(a, b, tol):
    return a is not None and b is not None and abs(a - b) <= tol * max(1.0, abs(b))


def crosscheck(pred, ds):
    """Score predictions with smallaxe.metrics and scikit-learn; return both + any mismatches."""
    y, yhat = ds.label, "predict_label"
    if ds.task == "simple_regression":
        pdf = pred.select(y, yhat).toPandas()
        sm = {
            "rmse": M.rmse(pred, y, yhat),
            "mae": M.mae(pred, y, yhat),
            "mse": M.mse(pred, y, yhat),
            "r2": M.r2(pred, y, yhat),
            "mape": M.mape(pred, y, yhat),
        }
        sk = {
            "rmse": math.sqrt(SK.mean_squared_error(pdf[y], pdf[yhat])),
            "mae": SK.mean_absolute_error(pdf[y], pdf[yhat]),
            "mse": SK.mean_squared_error(pdf[y], pdf[yhat]),
            "r2": SK.r2_score(pdf[y], pdf[yhat]),
            "mape": 100 * SK.mean_absolute_percentage_error(pdf[y], pdf[yhat]),
        }
        tol = {k: 1e-6 for k in sm}
    elif ds.task == "binary":
        p = pred.withColumn("p1", vector_to_array("probability")[1])
        pdf = p.select(y, yhat, "p1").toPandas()
        sm = {
            "accuracy": M.accuracy(p, y, yhat),
            "precision": M.precision(p, y, yhat),
            "recall": M.recall(p, y, yhat),
            "f1_score": M.f1_score(p, y, yhat),
            "auc_roc": M.auc_roc(p, y, "p1"),
            "auc_pr": M.auc_pr(p, y, "p1"),
            "log_loss": M.log_loss(p, y, "p1"),
        }
        prec, rec, _ = SK.precision_recall_curve(pdf[y], pdf["p1"])
        sk = {
            "accuracy": SK.accuracy_score(pdf[y], pdf[yhat]),
            "precision": SK.precision_score(pdf[y], pdf[yhat]),
            "recall": SK.recall_score(pdf[y], pdf[yhat]),
            "f1_score": SK.f1_score(pdf[y], pdf[yhat]),
            "auc_roc": SK.roc_auc_score(pdf[y], pdf["p1"]),
            "auc_pr": SK.auc(rec, prec),
            "log_loss": SK.log_loss(pdf[y], np.clip(pdf["p1"], 1e-15, 1 - 1e-15)),
        }
        # Spark's binary evaluator bins the curve (numBins=1000), so AUCs are approximate.
        tol = {
            "accuracy": 1e-9,
            "precision": 1e-9,
            "recall": 1e-9,
            "f1_score": 1e-9,
            "auc_roc": 5e-3,
            "auc_pr": 2e-2,
            "log_loss": 1e-6,
        }
    else:
        pdf = pred.select(y, yhat).toPandas()
        sm = {
            "accuracy": M.accuracy(pred, y, yhat),
            "precision": M.precision(pred, y, yhat),
            "recall": M.recall(pred, y, yhat),
            "f1_score": M.f1_score(pred, y, yhat),
        }
        sk = {
            "accuracy": SK.accuracy_score(pdf[y], pdf[yhat]),
            "precision": SK.precision_score(pdf[y], pdf[yhat], average="macro", zero_division=0),
            "recall": SK.recall_score(pdf[y], pdf[yhat], average="macro", zero_division=0),
            "f1_score": SK.f1_score(pdf[y], pdf[yhat], average="macro", zero_division=0),
        }
        # What smallaxe's binary formulas actually measure on multiclass data: class 1 one-vs-rest.
        pos = (pdf[y] == 1).astype(int), (pdf[yhat] == 1).astype(int)
        sk_class1 = {
            "precision": SK.precision_score(*pos, zero_division=0),
            "recall": SK.recall_score(*pos, zero_division=0),
            "f1_score": SK.f1_score(*pos, zero_division=0),
        }
        tol = {k: 1e-9 for k in sm}
    mismatches = {
        k: {"smallaxe": sm[k], "sklearn": sk[k]} for k in sm if not close(sm[k], sk[k], tol[k])
    }
    out = {
        "smallaxe": {k: round(v, 5) for k, v in sm.items() if v is not None},
        "mismatches": mismatches,
    }
    if ds.task == "multiclass":
        out["smallaxe_equals_class1_only"] = all(
            close(sm[k], sk_class1[k], 1e-9) for k in sk_class1
        )
    return out


def quality_gate(ds, sm):
    """Fail if a model does not clearly learn (guards against silently broken features)."""
    if ds.task == "simple_regression":
        assert sm["r2"] > 0.85, f"r2 {sm['r2']:.3f} <= 0.85"
    elif ds.task == "binary":
        assert sm["auc_roc"] > 0.75, f"auc_roc {sm['auc_roc']:.3f} <= 0.75"
    else:
        assert (
            sm["accuracy"] > ds.baseline_acc + 0.05
        ), f"accuracy {sm['accuracy']:.3f} vs majority {ds.baseline_acc:.3f}"


# COMMAND ----------

# MAGIC %md ## A. Full Pipeline per algorithm × dataset (Imputer → Scaler → Encoder(onehot) → model)

# COMMAND ----------

PIPES = {}


def pipeline_check(algo, ds):
    pipe = Pipeline(prep_steps() + [("model", make(algo, ds))])
    pipe.fit(ds.train, label_col=ds.label, numerical_cols=ds.num, categorical_cols=ds.cat)
    PIPES[(algo, ds.key)] = pipe
    pred = pipe.predict(ds.test).cache()
    n_out = pred.count()
    assert n_out == ds.n_test, f"row count changed: {ds.n_test} -> {n_out}"
    cc = crosscheck(pred, ds)
    pred.unpersist()
    n_feat = len(pipe["model"]._feature_cols)
    if cc["mismatches"] and not (ds.task == "multiclass" and cc.get("smallaxe_equals_class1_only")):
        raise AssertionError(f"smallaxe vs sklearn mismatch: {cc['mismatches']}")
    quality_gate(ds, cc["smallaxe"])
    cc["n_features_after_encoding"] = n_feat
    return cc


for ds in DATASETS:
    for algo in ALGOS:
        step(
            "A_pipeline",
            algo,
            ds.key,
            "fit+predict+metrics",
            lambda algo=algo, ds=ds: pipeline_check(algo, ds),
        )

# COMMAND ----------

# MAGIC %md ## Multiclass metric correctness (precision / recall / F1)

# COMMAND ----------

for algo in ALGOS:
    a_row = next(
        (
            r
            for r in results
            if r["section"] == "A_pipeline" and r["algo"] == algo and r["task"] == "multiclass"
        ),
        None,
    )

    def mc_metric_check(a_row=a_row):
        assert a_row and a_row["status"] == "PASS", "pipeline run did not pass"
        d = a_row["detail"]
        assert not d["mismatches"], (
            "smallaxe precision/recall/f1 disagree with sklearn macro averages"
            + (
                " and equal class-1 one-vs-rest values (binary formula reused for multiclass)"
                if d.get("smallaxe_equals_class1_only")
                else ""
            )
            + f": {d['mismatches']}"
        )
        return "matches sklearn macro averages"

    step(
        "A_metrics",
        algo,
        "multiclass",
        "precision/recall/f1 vs sklearn macro",
        mc_metric_check,
        probe=True,
    )

# COMMAND ----------

# MAGIC %md ## B. Validation strategies on preprocessed data (train_test, stratified kfold, feature inference)

# COMMAND ----------

for ds in [REG, BIN, MULTI]:
    ds.prep = Pipeline(prep_steps()).fit(ds.train, numerical_cols=ds.num, categorical_cols=ds.cat)
    ds.train_p = ds.prep.transform(ds.train).cache()
    ds.test_p = ds.prep.transform(ds.test).cache()
    ds.feats = [c for c in ds.train_p.columns if c != ds.label]
    ds.train_p.count(), ds.test_p.count()
    print(ds.name, "encoded features:", len(ds.feats))

MODELS = {}


def train_test_check(algo, ds):
    m = make(algo, ds)
    # feature_cols omitted on purpose: exercises numeric feature inference from the encoded frame
    m.fit(ds.train_p, label_col=ds.label, validation="train_test", test_size=0.2)
    s = m.validation_scores
    assert s["validation_type"] == "train_test" and s["test_size"] == 0.2
    assert sorted(m._feature_cols) == sorted(
        ds.feats
    ), "inferred features differ from encoded columns"
    MODELS[(algo, ds.key)] = m
    keys = (
        ["rmse", "r2", "mape"]
        if ds.task == "simple_regression"
        else ["accuracy", "f1_score", "auc_roc"]
    )
    return {k: round(s[k], 4) for k in keys if s.get(k) is not None}


def kfold_check(algo, ds, n_folds=3):
    m = make(algo, ds)
    m.fit(
        ds.train_p,
        label_col=ds.label,
        feature_cols=ds.feats,
        validation="kfold",
        n_folds=n_folds,
        stratified=True,
        cache_strategy="disk",
    )
    s = m.validation_scores
    assert (
        s["validation_type"] == "kfold"
        and s["n_folds"] == n_folds
        and len(s["fold_scores"]) == n_folds
    )
    for k in ["accuracy", "precision", "recall", "f1_score", "auc_roc", "auc_pr", "log_loss"]:
        assert s.get(f"mean_{k}") is not None, f"mean_{k} missing"
        assert s.get(f"std_{k}") is not None, f"std_{k} missing"
    MODELS[(algo, ds.key)] = m  # final model is refit on all training data
    return {k: round(s[f"mean_{k}"], 4) for k in ["accuracy", "auc_roc", "log_loss"]} | {
        "std_auc_roc": round(s["std_auc_roc"], 4)
    }


for algo in ALGOS:
    step(
        "B_validation",
        algo,
        "regression",
        "train_test + feature inference",
        lambda algo=algo: train_test_check(algo, REG),
    )
    step(
        "B_validation",
        algo,
        "binary",
        "stratified kfold(3) + cache_strategy=disk",
        lambda algo=algo: kfold_check(algo, BIN),
    )
    step(
        "B_validation",
        algo,
        "multiclass",
        "train_test (stratified)",
        lambda algo=algo: train_test_check(algo, MULTI),
    )

# fit(cache_strategy=...) may have unpersisted the caller's cache (probed in section G); re-cache for speed.
if not BIN.train_p.storageLevel.useMemory:
    BIN.train_p = BIN.train_p.cache()
    BIN.train_p.count()

# COMMAND ----------

# MAGIC %md ## C. predict_proba and feature_importances

# COMMAND ----------


def fitted(algo, ds):
    if (algo, ds.key) not in MODELS:
        raise Blocked(f"section B fit failed for {algo}/{ds.key}")
    return MODELS[(algo, ds.key)]


def proba_check(algo, ds):
    m = fitted(algo, ds)
    out = m.predict_proba(ds.test_p).withColumn("pa", vector_to_array("probability"))
    stats = out.select(
        F.count("*").alias("n"),
        F.min(F.size("pa")).alias("kmin"),
        F.max(F.size("pa")).alias("kmax"),
        F.max(F.abs(F.aggregate("pa", F.lit(0.0), lambda a, b: a + b) - 1)).alias("max_sum_dev"),
        F.min(F.array_min("pa")).alias("pmin"),
    ).first()
    assert stats["n"] == ds.test_p.count(), "row count changed"
    assert (
        stats["kmin"] == stats["kmax"] == ds.n_classes
    ), f"prob vector size {stats['kmin']}..{stats['kmax']} != {ds.n_classes}"
    assert (
        stats["max_sum_dev"] < 1e-6
    ), f"probabilities do not sum to 1 (dev {stats['max_sum_dev']})"
    assert stats["pmin"] >= 0
    return {"n_classes": ds.n_classes, "max_sum_dev": float(stats["max_sum_dev"])}


def importance_check(algo, ds):
    fi = fitted(algo, ds).feature_importances
    assert fi is not None, "feature_importances is None"
    assert len(fi) == len(ds.feats) and set(fi) == set(
        ds.feats
    ), f"{len(fi)} importances for {len(ds.feats)} features"
    top = sorted(fi.items(), key=lambda kv: -kv[1])[:3]
    return {"top3": [(k, round(float(v), 4)) for k, v in top]}


for algo in ALGOS:
    for ds in [BIN, MULTI]:
        step(
            "C_proba",
            algo,
            ds.key,
            "predict_proba shape/sum",
            lambda algo=algo, ds=ds: proba_check(algo, ds),
        )
    for ds in [REG, BIN, MULTI]:
        step(
            "C_importance",
            algo,
            ds.key,
            "feature_importances",
            lambda algo=algo, ds=ds: importance_check(algo, ds),
            probe=True,
        )

# COMMAND ----------

# MAGIC %md ## D. Model persistence: save → load (via factory) → identical predictions

# COMMAND ----------


def persist_check(algo, ds):
    m = fitted(algo, ds)
    path = f"{ART}/models/{algo}_{ds.key}"
    m.save(path)
    factory = Regressors if ds.task == "simple_regression" else Classifiers
    loaded = factory.load(path)
    assert type(loaded) is type(m), f"{type(loaded).__name__} != {type(m).__name__}"
    assert loaded._feature_cols == m._feature_cols and loaded.get_params() == m.get_params()
    sample_pdf = ds.test_p.limit(1000).toPandas()
    sample_pdf["rid"] = range(len(sample_pdf))
    sample = spark.createDataFrame(sample_pdf)
    cols = ["rid", "predict_label"] + (["probability"] if ds.task != "simple_regression" else [])
    a = m.predict(sample).select(*cols).toPandas().sort_values("rid").reset_index(drop=True)
    b = loaded.predict(sample).select(*cols).toPandas().sort_values("rid").reset_index(drop=True)
    max_diff = float(np.max(np.abs(a["predict_label"].values - b["predict_label"].values)))
    assert max_diff <= 1e-9, f"predict_label differs after load (max diff {max_diff})"
    if "probability" in cols:
        pd_diff = float(
            np.max(
                np.abs(
                    np.stack(a["probability"].map(lambda v: v.toArray()))
                    - np.stack(b["probability"].map(lambda v: v.toArray()))
                )
            )
        )
        assert pd_diff <= 1e-9, f"probability differs after load (max diff {pd_diff})"
    assert loaded.validation_scores == json.loads(
        json.dumps(m.validation_scores, default=str)
    ), "validation_scores not preserved"
    return {"path": path, "rows_compared": len(a), "max_pred_diff": max_diff}


for algo in ALGOS:
    for ds in [REG, BIN, MULTI]:
        step(
            "D_persistence",
            algo,
            ds.key,
            "save/load identical predictions",
            lambda algo=algo, ds=ds: persist_check(algo, ds),
        )

# COMMAND ----------

# MAGIC %md ## E. Hyperparameter search (`search.optimize`) for every algorithm

# COMMAND ----------

SPACES = {
    "random_forest": {
        "n_estimators": hp.quniform("n_estimators", 20, 80, 20),
        "max_depth": hp.quniform("max_depth", 3, 8, 1),
    },
    "boost": {
        "n_estimators": hp.quniform("n_estimators", 30, 90, 30),
        "max_depth": hp.quniform("max_depth", 3, 7, 1),
        "learning_rate": hp.uniform("learning_rate", 0.05, 0.3),
    },
}
REG_SEARCH = REG.train_p.sample(fraction=0.35, seed=SEED).cache()


def search_check(algo, ds, df, metric, max_evals):
    space = SPACES["random_forest" if algo == "random_forest" else "boost"]
    r = optimize.run(
        make(algo, ds),
        df,
        label_col=ds.label,
        feature_cols=ds.feats,
        param_space=space,
        metric=metric,
        validation="train_test",
        max_evals=max_evals,
        seed=SEED,
        verbose=False,
    )
    fails = [t["error"] for t in r.trials_history if t["status"] == "fail"]
    assert r.n_trials == max_evals and not fails, f"{len(fails)} failed trials: {fails[:2]}"
    assert r.best_model is not None and math.isfinite(r.best_score)
    assert set(r.best_params) == set(space)
    assert r.best_model.predict(ds.test_p).count() == ds.test_p.count()
    return {"best_" + metric: round(r.best_score, 4), "best_params": r.best_params}


for algo in ALGOS:
    step(
        "E_search",
        algo,
        "binary",
        "optimize auc_roc (4 evals)",
        lambda algo=algo: search_check(algo, BIN, BIN.train_p, "auc_roc", 4),
    )
    step(
        "E_search",
        algo,
        "regression",
        "optimize rmse (3 evals)",
        lambda algo=algo: search_check(algo, REG, REG_SEARCH, "rmse", 3),
    )

# COMMAND ----------

# MAGIC %md ## F. Label encoding (Encoder(method="label")) + every algorithm

# COMMAND ----------


def label_encoding_check(algo, ds):
    pipe = Pipeline(prep_steps("label") + [("model", make(algo, ds))])
    pipe.fit(ds.train, label_col=ds.label, numerical_cols=ds.num, categorical_cols=ds.cat)
    pred = pipe.predict(ds.test)
    n = pred.count()
    assert n == ds.n_test, f"row count changed: {ds.n_test} -> {n}"
    p = pred.withColumn("p1", vector_to_array("probability")[1])
    auc = M.auc_roc(p, ds.label, "p1")
    assert auc > 0.75, f"auc_roc {auc:.3f}"
    return {"auc_roc": round(auc, 4), "n_features": len(pipe["model"]._feature_cols)}


for algo in ALGOS:
    step(
        "F_label_encoding",
        algo,
        "binary",
        "Imputer+Scaler+Encoder(label)+model",
        lambda algo=algo: label_encoding_check(algo, BIN),
    )

# COMMAND ----------

# MAGIC %md ## G. Known-gap probes (from code review; GAP = gap confirmed, PASS = works)

# COMMAND ----------

LOCAL = f"/local_disk0/tmp/smallaxe_validation/{RUN_ID}"


def pipeline_persist_probe(path):
    if ("random_forest", "binary") not in PIPES:
        raise Blocked("section A RF/binary pipeline failed")
    pipe = PIPES[("random_forest", "binary")]
    pipe.save(path)
    loaded = Pipeline.load(path)
    n = loaded.predict(BIN.test).count()
    assert n == BIN.n_test
    return f"saved+loaded, {n} rows predicted"


step(
    "G_gaps",
    "random_forest",
    "binary",
    "Pipeline(save/load) with Encoder+model, local path",
    lambda: pipeline_persist_probe(f"{LOCAL}/pipe_local"),
    probe=True,
)
step(
    "G_gaps",
    "random_forest",
    "binary",
    "Pipeline(save/load) with Encoder+model, /dbfs path",
    lambda: pipeline_persist_probe(f"/dbfs/tmp/smallaxe_validation/{RUN_ID}/pipe_fuse"),
    probe=True,
)


def numeric_pipeline_persist_probe():
    pipe = Pipeline(
        [
            ("imputer", Imputer(numerical_strategy="median")),
            ("scaler", Scaler()),
            ("model", make("random_forest", REG)),
        ]
    )
    # numeric-only frame so the model sees no string columns
    pipe.fit(REG.train.select(*REG.num, "price"), label_col="price", numerical_cols=REG.num)
    pipe_path = f"{LOCAL}/pipe_numeric"
    pipe.save(pipe_path)
    loaded = Pipeline.load(pipe_path)
    return f"loaded; predicted {loaded.predict(REG.test.select(*REG.num, 'price')).count()} rows"


step(
    "G_gaps",
    "random_forest",
    "regression",
    "Pipeline(save/load) numeric-only + model",
    numeric_pipeline_persist_probe,
    probe=True,
)


def catboost_native_categorical_probe():
    pipe = Pipeline(
        [
            ("imputer", Imputer(numerical_strategy="median", categorical_strategy="most_frequent")),
            ("model", make("catboost", BIN)),
        ]
    )
    pipe.fit(BIN.train, label_col=BIN.label, numerical_cols=BIN.num, categorical_cols=BIN.cat)
    return f"predicted {pipe.predict(BIN.test).count()} rows with raw string categoricals"


step(
    "G_gaps",
    "catboost",
    "binary",
    "native categoricals (Pipeline without Encoder)",
    catboost_native_categorical_probe,
    probe=True,
)


def null_rows_probe():
    m = make("random_forest", REG).fit(REG.train, label_col="price", feature_cols=REG.num)
    holey = REG.test.withColumn(
        "carat", F.when(F.rand(SEED) < 0.05, None).otherwise(F.col("carat"))
    )
    n_in, n_out = holey.count(), m.predict(holey).count()
    assert n_out == n_in, f"predict silently dropped {n_in - n_out} of {n_in} rows containing nulls"
    return "rows preserved"


step(
    "G_gaps",
    "random_forest",
    "regression",
    "predict on rows with nulls (no Imputer)",
    null_rows_probe,
    probe=True,
)


def extra_column_probe():
    pipe = Pipeline(prep_steps() + [("model", make("random_forest", BIN))])
    pipe.fit(
        BIN.raw_with_id.randomSplit([0.8, 0.2], seed=SEED)[0],
        label_col="label",
        numerical_cols=BIN.num,
        categorical_cols=BIN.cat,
    )
    feats = pipe["model"]._feature_cols
    assert "customerID" not in feats, "unlisted ID column was passed to the model as a feature"
    return "ID column ignored"


step(
    "G_gaps",
    "random_forest",
    "binary",
    "unlisted string ID column in input frame",
    extra_column_probe,
    probe=True,
)


def unseen_category_probe():
    if ("random_forest", "binary") not in PIPES:
        raise Blocked("section A RF/binary pipeline failed")
    pipe = PIPES[("random_forest", "binary")]
    shifted = BIN.test.withColumn(
        "PaymentMethod",
        F.when(F.rand(SEED) < 0.1, F.lit("Crypto (new)")).otherwise(F.col("PaymentMethod")),
    )
    n = pipe.predict(shifted).count()
    assert n == BIN.n_test, f"{BIN.n_test - n} rows with an unseen category were dropped"
    return f"{n} rows predicted incl. unseen category"


step(
    "G_gaps",
    "random_forest",
    "binary",
    "unseen category at predict time",
    unseen_category_probe,
    probe=True,
)


def cache_side_effect_probe():
    df = BIN.train_p.limit(2000).cache()
    df.count()
    make("random_forest", BIN).fit(
        df, label_col=BIN.label, feature_cols=BIN.feats, cache_strategy="memory"
    )
    lvl = df.storageLevel
    assert (
        lvl.useMemory or lvl.useDisk
    ), "fit(cache_strategy='memory') unpersisted the caller's own cached DataFrame"
    return "caller cache intact"


step(
    "G_gaps",
    "random_forest",
    "binary",
    "fit(cache_strategy) leaves caller's cache intact",
    cache_side_effect_probe,
    probe=True,
)

# COMMAND ----------

# MAGIC %md ## Summary

# COMMAND ----------

res = pd.DataFrame(results)
counts = res["status"].value_counts().to_dict()
matrix = (
    res[
        res["section"].isin(
            [
                "A_pipeline",
                "B_validation",
                "C_proba",
                "D_persistence",
                "E_search",
                "F_label_encoding",
            ]
        )
    ]
    .assign(ok=lambda d: d["status"].eq("PASS"))
    .groupby(["algo", "task"])["ok"]
    .agg(lambda s: f"{int(s.sum())}/{len(s)}")
    .unstack(fill_value="-")
)
print("=" * 100)
print(json.dumps(ENV, indent=1))
print(matrix.to_string())
print("=" * 100)
for _, r in res.iterrows():
    if r["status"] != "PASS":
        print(
            f"{r['status']:>4}  {r['section']} | {r['algo']} | {r['task']} | {r['check']}\n       -> {str(r['detail'])[:400]}"
        )
print("=" * 100)
print(counts)

key_metrics = {
    f"{r['algo']}|{r['task']}": r["detail"]["smallaxe"]
    for r in results
    if r["section"] == "A_pipeline" and r["status"] == "PASS"
}
payload = {
    "env": ENV,
    "counts": counts,
    "matrix": matrix.to_dict(),
    "key_metrics": key_metrics,
    "rows": [{k: v for k, v in r.items() if k != "trace"} for r in results],
}
dbutils.fs.put(
    f"dbfs:/tmp/smallaxe_validation/results/{RUN_ID}.json", json.dumps(payload, default=str), True
)
dbutils.fs.put(
    f"dbfs:/tmp/smallaxe_validation/results/{RUN_ID}_traces.json",
    json.dumps([r for r in results if "trace" in r], default=str),
    True,
)
display(spark.createDataFrame(res.drop(columns=[c for c in ["trace"] if c in res]).astype(str)))

# COMMAND ----------

# Clean up scratch model artifacts (results JSON is kept under dbfs:/tmp/smallaxe_validation/results/)
dbutils.fs.rm(ART, True)
dbutils.fs.rm(f"dbfs:/tmp/smallaxe_validation/{RUN_ID}", True)
dbutils.notebook.exit(
    json.dumps(
        {
            "run_id": RUN_ID,
            "counts": counts,
            "matrix": matrix.to_dict(),
            "results_json": f"dbfs:/tmp/smallaxe_validation/results/{RUN_ID}.json",
        },
        default=str,
    )
)
