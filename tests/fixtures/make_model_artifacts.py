"""Write the frozen model artifacts that ``tests/test_legacy_artifacts.py`` loads.

Each fixture directory holds four tiny models saved by one smallaxe release (a Random
Forest regressor, a Random Forest multiclass classifier, an XGBoost regressor and an
XGBoost binary classifier) and ``expected_predictions.json``: what that release
predicted for :func:`frame`. The directory name says which release wrote it:
``artifacts_0_9_0`` must be written by smallaxe 0.9.0, ``artifacts_format_1`` by a
smallaxe that writes ``format_version`` 1.

For a release, run it with that release installed from PyPI, never with a checkout of
a later commit, in a fresh virtualenv (Python 3.11, PySpark 3.5.9, xgboost 3.2.0, scikit-learn and
pyarrow). ``-I`` keeps the repository off ``sys.path``, so the installed wheel is the
smallaxe that is imported::

    python -m venv .scratch/venv_0.9.0
    # install smallaxe==0.9.0 pyspark==3.5.9 xgboost==3.2.0 scikit-learn pyarrow into it
    export JAVA_HOME=... PYSPARK_PYTHON=.scratch/venv_0.9.0/bin/python
    .scratch/venv_0.9.0/bin/python -I tests/fixtures/make_model_artifacts.py \\
        tests/fixtures/artifacts_0_9_0

For ``artifacts_format_<n>``, run it from the repository root with the checkout on the
path (``PYTHONPATH=. python tests/fixtures/make_model_artifacts.py
tests/fixtures/artifacts_format_1``); it checks the saved ``format_version``.

The 0.8.1 fixtures (``artifacts_0_8_1``) predate this script and are left as they are.
It removes the ``.crc`` and ``_SUCCESS`` files Spark leaves.
"""

import json
import shutil
import sys
from pathlib import Path

from pyspark.sql import SparkSession

import smallaxe
from smallaxe.training import Classifiers, Regressors

# (fixture name, factory, task, label column, constructor params)
MODELS = [
    ("random_forest_regressor", "regressor", None, "target", {"max_depth": 3}),
    ("random_forest_classifier", "classifier", "multiclass", "multiclass", {"max_depth": 3}),
    ("xgboost_regressor", "regressor", None, "target", {"max_depth": 2}),
    ("xgboost_classifier", "classifier", "binary", "binary", {"max_depth": 2}),
]


def frame(spark):
    """The frame every fixture was fitted on and predicted for (as for 0.8.1)."""
    rows = [
        (i, 20.0 + (i % 30), 40000.0 + i * 1000.0, 50.0 + i * 5 + (i % 7), i % 2, i % 3)
        for i in range(1, 121)
    ]
    return spark.createDataFrame(rows, ["id", "age", "income", "target", "binary", "multiclass"])


def _check_writer(out):
    """Refuse to write a fixture directory with a smallaxe that does not match its name."""
    name = out.name
    if name.startswith("artifacts_format_"):
        return
    if not name.startswith("artifacts_"):
        raise SystemExit(f"{out} must be named artifacts_<version> or artifacts_format_<n>")
    expected = name[len("artifacts_") :].replace("_", ".")
    if smallaxe.__version__ != expected:
        raise SystemExit(
            f"{out} must be written by smallaxe {expected}; this is smallaxe "
            f"{smallaxe.__version__} from {smallaxe.__file__}"
        )


def _check_format(out, metadata_path):
    """An ``artifacts_format_<n>`` directory must hold ``format_version`` n artifacts."""
    if not out.name.startswith("artifacts_format_"):
        return
    expected = int(out.name[len("artifacts_format_") :])
    written = json.loads(metadata_path.read_text()).get("format_version")
    if written != expected:
        raise SystemExit(f"{metadata_path} has format_version {written!r}, not {expected}")


def main(out):
    _check_writer(out)
    spark = SparkSession.builder.master("local[1]").getOrCreate()
    df = frame(spark)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)

    expected = {}
    for name, kind, task, label_col, params in MODELS:
        algorithm_name = name.rsplit("_", 1)[0]
        params = dict(params, n_estimators=5, seed=42)
        if kind == "regressor":
            model = getattr(Regressors, algorithm_name)(**params)
        else:
            model = getattr(Classifiers, algorithm_name)(task=task, **params)
        model.fit(df, label_col=label_col, feature_cols=["age", "income"], validation="none")
        model.save(str(out / name))
        _check_format(out, out / name / "metadata.json")
        rows = model.predict(df).select("id", "predict_label").orderBy("id").collect()
        expected[name] = {
            "label_col": label_col,
            "predictions": [[row["id"], float(row["predict_label"])] for row in rows],
        }

    (out / "expected_predictions.json").write_text(json.dumps(expected, indent=1) + "\n")
    for junk in list(out.rglob("*.crc")) + list(out.rglob("_SUCCESS")):
        junk.unlink()
    print(f"smallaxe {smallaxe.__version__} ({smallaxe.__file__}) wrote {out}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: make_model_artifacts.py <fixture directory>")
    main(Path(sys.argv[1]).resolve())
