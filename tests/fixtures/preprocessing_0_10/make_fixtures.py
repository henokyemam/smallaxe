"""Regenerate the 0.10-layout preprocessing fixtures in this directory.

Run it with smallaxe 0.10 (for example a checkout of commit b47fec4) on the path, from
the repository root::

    python tests/fixtures/preprocessing_0_10/make_fixtures.py

It fits an ``Imputer`` and a ``Scaler`` with the 0.10 code, saves them with the 0.10
``save`` (``imputer_metadata.json`` / ``scaler_metadata.json`` written with ``open``,
plus the Scaler's ``spark_model``), and records the 0.10 ``transform`` output of the
probe frame in ``expected.json``. Delete the ``.crc`` and ``_SUCCESS`` files Spark leaves
before committing.
"""

import json
import math
import shutil
from pathlib import Path

from pyspark.sql import SparkSession

from smallaxe.preprocessing import Imputer, Scaler

HERE = Path(__file__).parent


def _rows(df, columns):
    out = []
    for row in df.orderBy("id").collect():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float) and math.isnan(value):
                value = "NaN"
            values.append(value)
        out.append(values)
    return out


def main():
    spark = SparkSession.builder.master("local[1]").getOrCreate()
    expected = json.loads((HERE / "expected.json").read_text())

    spec = expected["imputer"]
    schema = "id int, age double, income double, city string"
    imputer = Imputer(numerical_strategy="median", categorical_strategy="most_frequent")
    imputer.fit(spark.createDataFrame(spec["fit"], schema), ["age", "income"], ["city"])
    shutil.rmtree(HERE / "imputer", ignore_errors=True)
    imputer.save(str(HERE / "imputer"))
    spec["output"] = _rows(
        imputer.transform(spark.createDataFrame(spec["probe"], schema)), spec["columns"]
    )

    schema = "id int, x double, y double, note string"
    fit = [[i, float(i), float(i * i) - 3.0, "n"] for i in range(1, 7)]
    probe = fit + [[7, None, 2.5, "n"], [8, 100.0, None, None]]
    scaler = Scaler(method="standard")
    scaler.fit(spark.createDataFrame(fit, schema), ["x", "y"])
    shutil.rmtree(HERE / "scaler", ignore_errors=True)
    scaler.save(str(HERE / "scaler"))
    columns = ["id", "x", "y", "note"]
    expected["scaler"] = {
        "columns": columns,
        "fit": fit,
        "probe": probe,
        "output": _rows(scaler.transform(spark.createDataFrame(probe, schema)), columns),
    }

    (HERE / "expected.json").write_text(json.dumps(expected, indent=2) + "\n")
    for junk in list(HERE.rglob("*.crc")) + list(HERE.rglob("_SUCCESS")):
        junk.unlink()


if __name__ == "__main__":
    main()
