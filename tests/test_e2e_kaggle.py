"""End-to-end tests on real Kaggle datasets (network-gated).

These tests download small public Kaggle datasets and exercise the full
smallaxe workflow: load -> train -> evaluate -> save/load -> optimize -> predict.
They are skipped automatically when there is no network access, so they never
break offline or air-gapped CI runs.
"""

import os
import tempfile
import urllib.request

import pytest

from smallaxe.search import optimize
from smallaxe.training import Classifiers, Regressors

hp = pytest.importorskip("hyperopt").hp

TITANIC_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
HOUSE_URL = (
    "https://raw.githubusercontent.com/Shreyas3108/house-price-prediction/master/kc_house_data.csv"
)


def _download(url: str, name: str) -> str:
    """Download ``url`` to a temp cache, skipping the test if the network fails."""
    cache = os.path.join(tempfile.gettempdir(), name)
    if not os.path.exists(cache):
        try:
            urllib.request.urlretrieve(url, cache)
        except Exception as exc:  # noqa: BLE001 - any network failure -> skip
            pytest.skip(f"network unavailable for {url}: {exc}")
    return cache


@pytest.fixture(scope="module")
def titanic_df(spark_session):
    from pyspark.sql import functions as F

    path = _download(TITANIC_URL, "smallaxe_titanic_test.csv")
    df = spark_session.read.csv(path, header=True, inferSchema=True)
    age_median = df.approxQuantile("Age", [0.5], 0.0)[0]
    fare_median = df.approxQuantile("Fare", [0.5], 0.0)[0]
    return (
        df.withColumn("sex_code", F.when(F.col("Sex") == "male", 1).otherwise(0))
        .withColumn("Age", F.when(F.col("Age").isNull(), age_median).otherwise(F.col("Age")))
        .withColumn("Fare", F.when(F.col("Fare").isNull(), fare_median).otherwise(F.col("Fare")))
        .select(
            "Survived",
            "Pclass",
            "sex_code",
            F.col("Age").cast("double").alias("Age"),
            "SibSp",
            "Parch",
            F.col("Fare").cast("double").alias("Fare"),
        )
    )


@pytest.fixture(scope="module")
def house_df(spark_session):
    from pyspark.sql import functions as F

    path = _download(HOUSE_URL, "smallaxe_house_test.csv")
    features = ["bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "grade"]
    df = spark_session.read.csv(path, header=True, inferSchema=True)
    cols = [F.col(c).cast("double").alias(c) for c in features]
    return df.select(F.col("price").cast("double").alias("price"), *cols)


TITANIC_FEATURES = ["Pclass", "sex_code", "Age", "SibSp", "Parch", "Fare"]
HOUSE_FEATURES = ["bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "grade"]


class TestTitanicEndToEnd:
    """Binary classification on the Kaggle Titanic dataset."""

    def test_train_evaluate_persist_predict(self, titanic_df):
        import smallaxe

        smallaxe.set_seed(42)
        model = Classifiers.random_forest(task="binary", n_estimators=80, max_depth=5, seed=42)
        model.fit(
            titanic_df,
            label_col="Survived",
            feature_cols=TITANIC_FEATURES,
            validation="train_test",
            test_size=0.2,
        )
        scores = model.validation_scores
        # Sex + class are strong signals; a reasonable model clears these floors.
        assert scores["accuracy"] > 0.70
        assert scores["auc_roc"] > 0.75

        # Save/load must reproduce predictions exactly.
        model_dir = os.path.join(tempfile.mkdtemp(), "titanic_rf")
        model.save(model_dir)
        loaded = Classifiers.load(model_dir)
        before = model.predict(titanic_df).select("predict_label").collect()
        after = loaded.predict(titanic_df).select("predict_label").collect()
        assert [r.predict_label for r in before] == [r.predict_label for r in after]

    def test_optimization_improves_or_matches(self, titanic_df):
        import smallaxe

        smallaxe.set_seed(42)
        result = optimize.run(
            Classifiers.random_forest(task="binary", seed=42),
            titanic_df,
            label_col="Survived",
            feature_cols=TITANIC_FEATURES,
            param_space={
                "n_estimators": hp.quniform("n_estimators", 40, 120, 20),
                "max_depth": hp.quniform("max_depth", 3, 9, 1),
            },
            metric="auc_roc",
            validation="kfold",
            n_folds=3,
            max_evals=6,
            seed=42,
            verbose=False,
        )
        assert 0.5 < result.best_score <= 1.0
        assert result.best_model is not None
        preds = result.best_model.predict(titanic_df)
        assert "predict_label" in preds.columns


class TestHouseEndToEnd:
    """Regression on the Kaggle King County house-sales dataset."""

    def test_train_evaluate_optimize(self, house_df):
        import smallaxe

        smallaxe.set_seed(42)
        model = Regressors.random_forest(n_estimators=80, max_depth=10, seed=42)
        model.fit(
            house_df,
            label_col="price",
            feature_cols=HOUSE_FEATURES,
            validation="train_test",
            test_size=0.2,
        )
        scores = model.validation_scores
        # These features explain a good share of price variance.
        assert scores["r2"] > 0.5
        assert scores["rmse"] > 0

        result = optimize.run(
            Regressors.random_forest(seed=42),
            house_df,
            label_col="price",
            feature_cols=HOUSE_FEATURES,
            param_space={"max_depth": hp.quniform("max_depth", 5, 12, 1)},
            metric="rmse",
            validation="train_test",
            max_evals=5,
            seed=42,
            verbose=False,
        )
        assert result.best_score > 0
        assert isinstance(result.best_params["max_depth"], int)
