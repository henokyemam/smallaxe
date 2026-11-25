"""Write the frozen ``format_version`` 1 pipeline artifacts in this directory.

Run it from the repository root with the checkout on the path::

    PYTHONPATH=. python tests/fixtures/pipeline_format_1/make_fixture.py

It writes two pipelines saved by ``Pipeline.save`` and checks they are format_version 1:

- ``pipeline``: Imputer, Scaler, one-hot Encoder and a Random Forest regressor, fitted on
  ``numerical_cols`` and ``categorical_cols`` with an unlisted string ID column present.
- ``pipeline_skipped``: the same steps fitted with ``numerical_cols`` only, so the
  Encoder is saved as a skipped step.

``expected.json`` records each pipeline's feature columns, skipped steps, and its
predictions for the probe frame (which has an unseen category and an extra column). It
removes the ``.crc`` and ``_SUCCESS`` files Spark leaves.
"""

import json
import shutil
from pathlib import Path

from pyspark.sql import SparkSession

from smallaxe.pipeline import Pipeline
from smallaxe.preprocessing import Encoder, Imputer, Scaler
from smallaxe.training import regressors

HERE = Path(__file__).parent
SCHEMA = "id int, code string, age double, income double, city string, target double"
CITIES = ["paris", "rome", "oslo"]

FIT = [
    [
        i,
        f"c{i:03d}",
        float(20 + i % 30) if i % 7 else None,
        float(40000 + i * 850),
        CITIES[i % 3] if i % 8 else None,
        50.0 + i * 3 + (i % 3) * 20,
    ]
    for i in range(1, 61)
]
PROBE = FIT[:8] + [
    [101, "c101", 33.0, 64000.0, "lima", 0.0],
    [102, "c102", None, 70000.0, None, 0.0],
]


def _pipeline():
    return Pipeline(
        [
            ("imputer", Imputer(numerical_strategy="median", categorical_strategy="most_frequent")),
            ("scaler", Scaler()),
            ("encoder", Encoder(method="onehot")),
            ("model", regressors.random_forest(n_estimators=5, max_depth=3, seed=11)),
        ]
    )


def _predictions(pipeline, df):
    rows = pipeline.predict(df).select("id", "predict_label").orderBy("id").collect()
    return [[row["id"], row["predict_label"]] for row in rows]


def main():
    spark = SparkSession.builder.master("local[1]").getOrCreate()
    fit = spark.createDataFrame(FIT, SCHEMA)
    probe = spark.createDataFrame(PROBE, SCHEMA).drop("target")

    cases = {
        "pipeline": {"numerical_cols": ["age", "income"], "categorical_cols": ["city"]},
        "pipeline_skipped": {"numerical_cols": ["age", "income"], "categorical_cols": None},
    }
    expected = {"schema": SCHEMA, "probe": PROBE, "pipelines": {}}
    for name, columns in cases.items():
        pipeline = _pipeline().fit(fit, label_col="target", **columns)
        out = HERE / name
        shutil.rmtree(out, ignore_errors=True)
        pipeline.save(str(out))
        written = json.loads((out / "metadata.json").read_text())["format_version"]
        if written != 1:
            raise SystemExit(f"{out} has format_version {written!r}, not 1")
        expected["pipelines"][name] = {
            **columns,
            "feature_cols": pipeline.feature_cols,
            "skipped_steps": pipeline.skipped_steps,
            "predictions": _predictions(pipeline, probe),
        }

    (HERE / "expected.json").write_text(json.dumps(expected, indent=2) + "\n")
    for junk in list(HERE.rglob("*.crc")) + list(HERE.rglob("_SUCCESS")):
        junk.unlink()


if __name__ == "__main__":
    main()
