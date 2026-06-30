"""End-to-end regression on the Kaggle "House Sales in King County" dataset.

Demonstrates the full smallaxe regression workflow on a real-world dataset:
load -> train -> evaluate -> save/load -> optimize -> compare algorithms.

The data is the Kaggle "House Sales in King County, USA" dataset (~21k rows),
pulled from a public no-auth mirror. The target is the home sale ``price``.

Run:
    python examples/house_prices_regression.py
"""

import os
import sys
import tempfile
import urllib.request

os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

from pyspark.sql import SparkSession  # noqa: E402

import smallaxe  # noqa: E402
from smallaxe.search import optimize  # noqa: E402
from smallaxe.training import Regressors  # noqa: E402
from smallaxe.training.xgboost import XGBOOST_AVAILABLE  # noqa: E402

HOUSE_URL = (
    "https://raw.githubusercontent.com/Shreyas3108/house-price-prediction/master/kc_house_data.csv"
)

FEATURE_COLS = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "view",
    "condition",
    "grade",
    "sqft_above",
    "sqft_basement",
    "yr_built",
    "lat",
    "long",
    "sqft_living15",
    "sqft_lot15",
]


def load_house(spark: SparkSession):
    cache = os.path.join(tempfile.gettempdir(), "smallaxe_kc_house.csv")
    if not os.path.exists(cache):
        urllib.request.urlretrieve(HOUSE_URL, cache)
    df = spark.read.csv(cache, header=True, inferSchema=True)
    # Cast features + target to double so feature inference and metrics are clean.
    from pyspark.sql import functions as F

    cols = [F.col(c).cast("double").alias(c) for c in FEATURE_COLS]
    return df.select(F.col("price").cast("double").alias("price"), *cols)


def report(scores: dict, title: str) -> None:
    print(f"\n[{title}] held-out validation scores:")
    for key in ("rmse", "mae", "mse", "r2", "mape"):
        if scores.get(key) is not None:
            print(f"  {key:>5}: {scores[key]:,.4f}")


def main() -> None:
    spark = (
        SparkSession.builder.master("local[2]")
        .appName("smallaxe-house")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    smallaxe.set_seed(42)

    df = load_house(spark)
    n = df.count()
    print(f"Loaded King County house sales: {n} rows, {len(FEATURE_COLS)} features")

    # --- Train + evaluate a Random Forest regressor. ---
    rf = Regressors.random_forest(n_estimators=100, max_depth=10, seed=42)
    rf.fit(df, label_col="price", feature_cols=FEATURE_COLS, validation="train_test", test_size=0.2)
    report(rf.validation_scores, "Random Forest")

    # --- Save and reload; confirm identical predictions. ---
    from pyspark.sql import functions as F

    model_dir = os.path.join(tempfile.mkdtemp(), "house_rf")
    rf.save(model_dir)
    loaded = Regressors.load(model_dir)
    diff = (
        rf.predict(df)
        .select("price", F.col("predict_label").alias("p1"))
        .withColumn("rn", F.monotonically_increasing_id())
        .join(
            loaded.predict(df)
            .select(F.col("predict_label").alias("p2"))
            .withColumn("rn", F.monotonically_increasing_id()),
            "rn",
        )
        .filter(F.abs(F.col("p1") - F.col("p2")) > 1e-9)
        .count()
    )
    print(f"\n[Persistence] save/load prediction mismatches: {diff} (expected 0)")

    # --- Optimize the Random Forest for RMSE. ---
    from hyperopt import hp

    result = optimize.run(
        Regressors.random_forest(seed=42),
        df,
        label_col="price",
        feature_cols=FEATURE_COLS,
        param_space={
            "n_estimators": hp.quniform("n_estimators", 40, 160, 20),
            "max_depth": hp.quniform("max_depth", 5, 15, 1),
        },
        metric="rmse",
        validation="train_test",
        max_evals=12,
        seed=42,
        verbose=False,
    )
    print(
        f"\n[Optimization] best RMSE: {result.best_score:,.2f} "
        f"over {result.n_successful_trials} trials"
    )
    print(f"[Optimization] best params: {result.best_params}")

    # --- Compare against XGBoost when available. ---
    if XGBOOST_AVAILABLE:
        xgb = Regressors.xgboost(n_estimators=150, max_depth=6, learning_rate=0.1, seed=42)
        xgb.fit(
            df, label_col="price", feature_cols=FEATURE_COLS, validation="train_test", test_size=0.2
        )
        report(xgb.validation_scores, "XGBoost")
    else:
        print("\n[XGBoost] not installed; skipping cross-algorithm comparison.")

    spark.stop()
    print("\nHouse-prices regression end-to-end workflow: SUCCESS")


if __name__ == "__main__":
    main()
