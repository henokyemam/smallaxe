"""Write the 0.10-layout pipeline fixture that ``tests/test_pipeline_persistence.py`` loads.

Run it with the released smallaxe 0.10.0 wheel installed from PyPI, never with a checkout,
in a fresh virtualenv (Python 3.11, PySpark 3.5.9). ``-I`` keeps the repository off
``sys.path``, so the installed wheel is the smallaxe that is imported::

    python -m venv .scratch/venv_0.10.0
    # install smallaxe==0.10.0 pyspark==3.5.9 into it
    export JAVA_HOME=... PYSPARK_PYTHON=.scratch/venv_0.10.0/bin/python
    .scratch/venv_0.10.0/bin/python -I tests/fixtures/pipeline_0_10/make_fixture.py

It fits ``Pipeline([Imputer, Scaler, Encoder(method="label")])`` with the 0.10 code and
saves it with the 0.10 ``Pipeline.save`` into ``pipeline/``: an unversioned
``metadata.json`` with ``step_names`` / ``step_types``, the Imputer and Scaler in their
0.10 layout (``imputer_metadata.json``, ``scaler_metadata.json`` plus ``spark_model/``)
and the Encoder pickled as ``step_2_encoder.pkl`` (0.10's Encoder had no ``save``; only
a label Encoder pickles, a one-hot one holds Spark models). ``expected.json`` records the
0.10 ``transform`` output of the probe frame and the smallaxe and PySpark versions that
wrote it. It removes the ``.crc`` and ``_SUCCESS`` files Spark leaves.
"""

import json
import math
import shutil
from pathlib import Path

import pyspark
from pyspark.sql import SparkSession

import smallaxe
from smallaxe.pipeline import Pipeline
from smallaxe.preprocessing import Encoder, Imputer, Scaler

HERE = Path(__file__).parent
SCHEMA = "id int, age double, income double, city string, note string"
COLUMNS = ["id", "age", "income", "city", "note"]

FIT = [
    [1, 25.0, 50000.0, "paris", "a"],
    [2, 31.0, None, "paris", "b"],
    [3, None, 72000.0, "rome", "c"],
    [4, 44.0, 81000.0, None, "d"],
    [5, 52.0, 93000.0, "rome", "e"],
    [6, 38.0, 61000.0, "oslo", "f"],
    [7, 29.0, 55000.0, "paris", "g"],
]
# The fit rows, then an unseen city, a null city, and nulls in both numeric columns.
PROBE = FIT + [
    [8, 33.0, 64000.0, "lima", "h"],
    [9, 47.0, 70000.0, None, "i"],
    [10, None, None, "rome", None],
]


def _rows(df):
    out = []
    for row in df.orderBy("id").collect():
        values = []
        for column in COLUMNS:
            value = row[column]
            if isinstance(value, float) and math.isnan(value):
                value = "NaN"
            values.append(value)
        out.append(values)
    return out


def main():
    if smallaxe.__version__ != "0.10.0":
        raise SystemExit(
            f"pipeline_0_10 must be written by smallaxe 0.10.0; this is smallaxe "
            f"{smallaxe.__version__} from {smallaxe.__file__}"
        )
    spark = SparkSession.builder.master("local[1]").getOrCreate()

    pipeline = Pipeline(
        [
            ("imputer", Imputer(numerical_strategy="median", categorical_strategy="most_frequent")),
            ("scaler", Scaler(method="standard")),
            ("encoder", Encoder(method="label")),
        ]
    )
    pipeline.fit(
        spark.createDataFrame(FIT, SCHEMA),
        numerical_cols=["age", "income"],
        categorical_cols=["city"],
    )
    out = HERE / "pipeline"
    shutil.rmtree(out, ignore_errors=True)
    pipeline.save(str(out))

    expected = {
        "smallaxe_version": smallaxe.__version__,
        "pyspark_version": pyspark.__version__,
        "schema": SCHEMA,
        "columns": COLUMNS,
        "numerical_cols": ["age", "income"],
        "categorical_cols": ["city"],
        "fit": FIT,
        "probe": PROBE,
        "output": _rows(pipeline.transform(spark.createDataFrame(PROBE, SCHEMA))),
    }
    (HERE / "expected.json").write_text(json.dumps(expected, indent=2) + "\n")
    for junk in list(HERE.rglob("*.crc")) + list(HERE.rglob("_SUCCESS")):
        junk.unlink()


if __name__ == "__main__":
    main()
