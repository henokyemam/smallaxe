"""End-to-end binary classification on the Kaggle Titanic dataset.

Demonstrates the full smallaxe workflow on a real-world dataset:
load -> preprocess -> train -> evaluate -> save/load -> optimize -> predict.

The data is the Kaggle "Titanic - Machine Learning from Disaster" training set,
pulled from a public no-auth mirror.

Run:
    python examples/titanic_classification.py
"""

import os
import sys
import tempfile
import urllib.request

# Make Spark workers use the same interpreter as the driver (avoids
# PYTHON_VERSION_MISMATCH on machines with multiple Pythons on PATH).
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

from pyspark.sql import SparkSession  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

import smallaxe  # noqa: E402
from smallaxe.search import optimize  # noqa: E402
from smallaxe.training import Classifiers  # noqa: E402

TITANIC_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"


def load_titanic(spark: SparkSession):
    """Download the Titanic training set and return engineered features."""
    cache = os.path.join(tempfile.gettempdir(), "smallaxe_titanic.csv")
    if not os.path.exists(cache):
        urllib.request.urlretrieve(TITANIC_URL, cache)

    df = spark.read.csv(cache, header=True, inferSchema=True)

    # Engineer numeric features. Encode the two key categoricals (Sex, Embarked)
    # to numeric codes and impute the missing Age/Fare/Embarked values.
    age_median = df.approxQuantile("Age", [0.5], 0.0)[0]
    fare_median = df.approxQuantile("Fare", [0.5], 0.0)[0]
    df = (
        df.withColumn("sex_code", F.when(F.col("Sex") == "male", 1).otherwise(0))
        .withColumn(
            "embarked_code",
            F.when(F.col("Embarked") == "S", 0).when(F.col("Embarked") == "C", 1).otherwise(2),
        )
        .withColumn("Age", F.when(F.col("Age").isNull(), age_median).otherwise(F.col("Age")))
        .withColumn("Fare", F.when(F.col("Fare").isNull(), fare_median).otherwise(F.col("Fare")))
        .select(
            "PassengerId",
            "Survived",
            "Pclass",
            "sex_code",
            F.col("Age").cast("double").alias("Age"),
            "SibSp",
            "Parch",
            F.col("Fare").cast("double").alias("Fare"),
            "embarked_code",
        )
    )
    return df


def main() -> None:
    spark = (
        SparkSession.builder.master("local[2]")
        .appName("smallaxe-titanic")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    smallaxe.set_seed(42)

    df = load_titanic(spark)
    n = df.count()
    print(f"Loaded Titanic: {n} rows, columns: {df.columns}")

    feature_cols = ["Pclass", "sex_code", "Age", "SibSp", "Parch", "Fare", "embarked_code"]

    # --- Train + evaluate a Random Forest classifier (train/test split). ---
    model = Classifiers.random_forest(task="binary", n_estimators=100, max_depth=5, seed=42)
    model.fit(
        df,
        label_col="Survived",
        feature_cols=feature_cols,
        validation="train_test",
        test_size=0.2,
    )
    scores = model.validation_scores
    print("\n[Random Forest] held-out validation scores:")
    for key in ("accuracy", "precision", "recall", "f1_score", "auc_roc", "auc_pr", "log_loss"):
        if scores.get(key) is not None:
            print(f"  {key:>10}: {scores[key]:.4f}")

    # --- Save and reload; confirm identical predictions. ---
    model_dir = os.path.join(tempfile.mkdtemp(), "titanic_rf")
    model.save(model_dir)
    loaded = Classifiers.load(model_dir)
    before = model.predict(df).select("PassengerId", "predict_label")
    after = loaded.predict(df).select("PassengerId", "predict_label")
    mismatches = (
        before.join(after.withColumnRenamed("predict_label", "predict_label2"), "PassengerId")
        .filter(F.col("predict_label") != F.col("predict_label2"))
        .count()
    )
    print(f"\n[Persistence] save/load prediction mismatches: {mismatches} (expected 0)")

    # --- Hyperparameter optimization. ---
    from hyperopt import hp

    result = optimize.run(
        Classifiers.random_forest(task="binary", seed=42),
        df,
        label_col="Survived",
        feature_cols=feature_cols,
        param_space={
            "n_estimators": hp.quniform("n_estimators", 40, 160, 20),
            "max_depth": hp.quniform("max_depth", 3, 10, 1),
            "feature_subset_strategy": ["sqrt", "log2", "onethird"],
        },
        metric="auc_roc",
        validation="kfold",
        n_folds=4,
        max_evals=15,
        seed=42,
        verbose=False,
    )
    print(
        "\n[Optimization] best AUC-ROC: "
        f"{result.best_score:.4f} over {result.n_successful_trials} trials"
    )
    print(f"[Optimization] best params: {result.best_params}")

    # --- Predict with the tuned model. ---
    preds = result.best_model.predict(df)
    survivors = preds.filter(F.col("predict_label") == 1).count()
    print(f"\n[Prediction] tuned model predicts {survivors}/{n} survivors")

    spark.stop()
    print("\nTitanic end-to-end workflow: SUCCESS")


if __name__ == "__main__":
    main()
