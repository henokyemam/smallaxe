"""The same results with Spark's ANSI SQL mode on and off.

Spark 4 turns ``spark.sql.ansi.enabled`` on by default: an invalid cast, a division by
zero, an arithmetic overflow, a boolean compared with a number, or (on Spark 3.3) a
missing map key then raise instead of giving null. smallaxe never sets the option; its
expressions are written so that the result does not depend on it. Each case below runs
on the test session with the option off and then on (restored afterwards) and must give
the same value, or the same smallaxe exception, both times. Spark 3.3 to 3.5 implement
ANSI mode too, so this is local evidence for the Spark 4 default.

The expressions are built inside each case, after the option is set: Spark fixes a
cast's or a division's ANSI behaviour when the expression is created.
"""

import math
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Dict

import pytest
from pyspark.ml.linalg import Vectors
from pyspark.sql import Row, SparkSession

from smallaxe.datasets import load_sample_classification, load_sample_regression
from smallaxe.exceptions import SmallaxeError
from smallaxe.metrics import (
    accuracy,
    auc_pr,
    auc_roc,
    f1_score,
    log_loss,
    mae,
    mape,
    multiclass_scores,
    precision,
    r2,
    recall,
    rmse,
)
from smallaxe.preprocessing import Encoder, Imputer, Scaler
from smallaxe.training.mixins.validation_mixin import ValidationMixin
from smallaxe.viz import _classification, _regression

ANSI = "spark.sql.ansi.enabled"
ROOT = Path(__file__).resolve().parent.parent


def _normalise(value: Any) -> Any:
    """``value`` with NaN made comparable, Rows as dicts and floats exact."""
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    if isinstance(value, Row):
        return _normalise(value.asDict(recursive=True))
    if isinstance(value, dict):
        return {key: _normalise(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalise(item) for item in value]
    return value


def _rows(dataframe: Any, order: str = "id") -> Any:
    return [row.asDict(recursive=True) for row in dataframe.orderBy(order).collect()]


def _sorted_rows(dataframe: Any) -> Any:
    return sorted((tuple(sorted(row.asDict().items())) for row in dataframe.collect()), key=str)


# --- Metrics ----------------------------------------------------------------------------


def _mape_zero_labels(spark: SparkSession) -> Any:
    df = spark.createDataFrame(
        [(0.0, 1.0), (2.0, 1.0), (4.0, 5.0), (None, 1.0)], "label double, predict_label double"
    )
    return mape(df)


def _mape_all_zero(spark: SparkSession) -> Any:
    df = spark.createDataFrame([(0, 1.0), (0, 2.0)], "label int, predict_label double")
    return mape(df)


def _mape_integer_columns(spark: SparkSession) -> Any:
    df = spark.createDataFrame(
        [(0, 3), (2, 1), (-4, 5), (2147483647, -2147483648)], "label int, predict_label int"
    )
    return mape(df)


def _mape_decimal_label(spark: SparkSession) -> Any:
    df = spark.createDataFrame(
        [(Decimal("0.00"), 1.0), (Decimal("2.50"), 2.0), (Decimal("-1.25"), -1.0)],
        "label decimal(10,2), predict_label double",
    )
    return mape(df)


def _regression_scores_constant_label(spark: SparkSession) -> Any:
    df = spark.createDataFrame([(3, 3.0), (3, 2.0), (3, 4.0)], "label int, predict_label double")
    return [r2(df), rmse(df), mae(df)]


def _log_loss_integer_labels(spark: SparkSession) -> Any:
    df = spark.createDataFrame(
        [(0, 0.0), (1, 1.0), (1, 0.25), (0, None)], "label int, probability double"
    )
    return log_loss(df)


def _multiclass_zero_denominators(spark: SparkSession) -> Any:
    # Class 3 is only predicted (support 0) and class 2 never predicted (precision 0/0).
    df = spark.createDataFrame(
        [(0, 0.0), (1, 3.0), (2, 1.0), (1, 1.0), (None, 1.0), (2, None)],
        "label int, predict_label double",
    )
    return multiclass_scores(df)


def _multiclass_all_null(spark: SparkSession) -> Any:
    df = spark.createDataFrame([(None, 1.0), (1, None)], "label int, predict_label double")
    return multiclass_scores(df)


def _binary_scores_boolean_label(spark: SparkSession) -> Any:
    df = spark.createDataFrame(
        [(True, 1.0), (False, 1.0), (True, 0.0), (False, 0.0), (None, 1.0)],
        "label boolean, predict_label double",
    )
    return [accuracy(df), precision(df), recall(df), f1_score(df)]


def _binary_scores_boolean_columns(spark: SparkSession) -> Any:
    df = spark.createDataFrame(
        [(True, True), (False, True), (True, False)], "label boolean, predict_label boolean"
    )
    return [accuracy(df), precision(df), recall(df), f1_score(df)]


def _binary_scores_string_label(spark: SparkSession) -> Any:
    # A string that is not a number never matches; "1" and " 1 " are the number 1.
    df = spark.createDataFrame(
        [("1", 1.0), (" 1 ", 1.0), ("yes", 1.0), ("0", 0.0), ("0", 1.0), (None, 0.0)],
        "label string, predict_label double",
    )
    return [accuracy(df), precision(df), recall(df), f1_score(df)]


def _binary_scores_string_prediction(spark: SparkSession) -> Any:
    df = spark.createDataFrame(
        [(1, "1"), (1, "no"), (0, "0"), (0, "1")], "label int, predict_label string"
    )
    return [accuracy(df), precision(df), recall(df), f1_score(df)]


def _accuracy_string_columns(spark: SparkSession) -> Any:
    df = spark.createDataFrame(
        [("a", "a"), ("b", "a"), ("c", "c")], "label string, predict_label string"
    )
    return [accuracy(df), precision(df), recall(df)]


def _binary_scores_no_positives(spark: SparkSession) -> Any:
    df = spark.createDataFrame([(0, 0.0), (0, 0.0)], "label int, predict_label double")
    return [accuracy(df), precision(df), recall(df), f1_score(df)]


def _auc_one_class(spark: SparkSession) -> Any:
    df = spark.createDataFrame([(0, 0.2), (0, 0.4)], "label int, probability double")
    return [auc_roc(df), auc_pr(df)]


def _auc_integer_scores(spark: SparkSession) -> Any:
    df = spark.createDataFrame(
        [(0, 0), (1, 1), (1, 0), (0, 1), (1, 1)], "label int, probability int"
    )
    return [auc_roc(df), auc_pr(df)]


# --- Preprocessing ----------------------------------------------------------------------


def _encoder_unseen_and_null(spark: SparkSession) -> Any:
    train = spark.createDataFrame([(1, "a"), (2, "b"), (3, "a")], "id int, c string")
    test = spark.createDataFrame([(1, "a"), (2, "z"), (3, None)], "id int, c string")
    results = []
    for method in ("onehot", "label"):
        encoder = Encoder(method=method).fit(train, categorical_cols=["c"])
        results.append(_rows(encoder.transform(test)))
    return results


def _encoder_rare_and_typed_columns(spark: SparkSession) -> Any:
    train = spark.createDataFrame(
        [(1, True, 1.5), (2, False, 2.5), (3, True, 1.5), (4, True, 3.5)],
        "id int, flag boolean, x double",
    )
    test = spark.createDataFrame(
        [(1, False, 9.5), (2, None, 1.5), (3, True, float("nan"))],
        "id int, flag boolean, x double",
    )
    encoder = Encoder(max_categories=1, handle_rare="other").fit(
        train, categorical_cols=["flag", "x"]
    )
    return _rows(encoder.transform(test))


def _imputer_categorical_fill_spellings(spark: SparkSession) -> Any:
    df = spark.createDataFrame([(1, 3), (2, None)], "id int, code int")
    results = []
    for fill in ("7.0", "7", " 7 ", "+7", "7.", "1e3", "7.5", "UNKNOWN", "99999999999"):
        try:
            imputed = Imputer(categorical_strategy=fill).fit_transform(
                df, categorical_cols=["code"]
            )
            results.append((fill, _rows(imputed)))
        except SmallaxeError as exc:
            results.append((fill, type(exc).__name__))
    return results


def _imputer_categorical_typed_columns(spark: SparkSession) -> Any:
    df = spark.createDataFrame(
        [(1, Decimal("1.25"), True, 2.0), (2, None, None, None)],
        "id int, price decimal(5,2), flag boolean, d double",
    )
    results = []
    for column, fill in (
        ("price", "1.5"),
        ("price", "1.255"),
        ("price", "12345.6"),
        ("price", "abc"),
        ("flag", "true"),
        ("flag", "maybe"),
        ("d", "2.5"),
        ("d", "x"),
    ):
        try:
            imputed = Imputer(categorical_strategy=fill).fit_transform(
                df, categorical_cols=[column]
            )
            results.append((column, fill, _rows(imputed)))
        except SmallaxeError as exc:
            results.append((column, fill, type(exc).__name__))
    return results


def _imputer_most_frequent_typed_columns(spark: SparkSession) -> Any:
    df = spark.createDataFrame(
        [(1, 2, 1.5), (2, 2, float("nan")), (3, None, None), (4, 5, 1.5)],
        "id int, code bigint, d double",
    )
    imputer = Imputer(categorical_strategy="most_frequent")
    return [_rows(imputer.fit_transform(df, categorical_cols=["code", "d"]))]


def _imputer_numerical_edge_columns(spark: SparkSession) -> Any:
    df = spark.createDataFrame(
        [(1, 4, None, 2.0, float("nan")), (2, 4, None, None, float("nan")), (3, 4, None, 6.0, 1.0)],
        "id int, constant int, empty double, d double, mostly_nan double",
    )
    results = []
    for strategy in ("mean", "median", "mode", 0, 2.5):
        imputer = Imputer(numerical_strategy=strategy)
        imputed = imputer.fit_transform(df, numerical_cols=["constant", "empty", "d", "mostly_nan"])
        results.append((strategy, imputer.numerical_fill_values, _rows(imputed)))
    return results


def _scaler_constant_and_null_columns(spark: SparkSession) -> Any:
    df = spark.createDataFrame(
        [(1, 5, 1.0), (2, 5, None), (3, 5, 3.0)], "id int, constant int, x double"
    )
    results = []
    for method in ("standard", "minmax", "maxabs"):
        scaled = Scaler(method=method).fit_transform(df, numerical_cols=["constant", "x"])
        results.append((method, _rows(scaled)))
    return results


# --- Visualization aggregates -----------------------------------------------------------


def _viz_residual_and_density_tables(spark: SparkSession) -> Any:
    df = spark.createDataFrame(
        [(1.0, 1.0), (2.0, 2.0), (3.0, 3.5), (float("inf"), 1.0), (None, 2.0)],
        "label double, predict_label double",
    )
    constant = spark.createDataFrame([(2, 2.0), (2, 2.0)], "label int, predict_label double")
    results = []
    for frame in (df, constant):
        histogram, stats = _regression._residual_table(frame, "label", "predict_label", 5)
        cells, density = _regression._density_table(frame, "label", "predict_label", 5)
        results.append((_sorted_rows(histogram), stats, _sorted_rows(cells), density))
    return results


def _viz_score_bins(spark: SparkSession) -> Any:
    numeric = spark.createDataFrame(
        [(0, 0.0), (1, 1.0), (1, 0.55), (2, 0.5), (0, 1.5), (1, float("nan")), (0, None)],
        "label int, probability double",
    )
    vectors = spark.createDataFrame(
        [
            (0, Vectors.dense([0.9, 0.1])),
            (1, Vectors.dense([0.2, 0.8])),
            (1, Vectors.dense([1.0])),
            (0, Vectors.dense([0.1, 0.2, 0.7])),
        ],
        ["label", "probability"],
    )
    return [
        _sorted_rows(_classification._score_bins(frame, "label", "probability", 10, "roc_curve"))
        for frame in (numeric, vectors)
    ]


def _viz_confusion_table(spark: SparkSession) -> Any:
    df = spark.createDataFrame(
        [(0, 0.0), (1, 2.0), (2, 2.0), (None, 1.0), (1, float("nan"))],
        "label bigint, predict_label double",
    )
    return _sorted_rows(_classification._confusion_table(df, "label", "predict_label"))


# --- Splits and datasets ----------------------------------------------------------------


def _validation_split_assignment(spark: SparkSession) -> Any:
    df = spark.createDataFrame(
        [(i, float(i) / 3, None if i % 4 == 0 else str(i), i % 2 == 0) for i in range(40)],
        "id int, x double, s string, flag boolean",
    )
    mixin = ValidationMixin()
    train, test = mixin._train_test_split(df, test_size=0.3, seed=11)
    folds = [
        sorted(row["id"] for row in validation.collect())
        for _, validation in mixin._kfold_split(df, n_folds=3, seed=11, fold_key="s")
    ]
    return [sorted(row["id"] for row in test.collect()), train.count(), folds]


def _sample_datasets(spark: SparkSession) -> Any:
    regression = load_sample_regression(spark, n_rows=30, seed=3)
    classification = load_sample_classification(spark, n_rows=30, seed=3)
    return [regression.collect(), classification.collect()]


CASES: Dict[str, Callable[[SparkSession], Any]] = {
    name[1:]: case
    for name, case in sorted(globals().items())
    if name.startswith("_")
    and callable(case)
    and name not in {"_normalise", "_rows", "_sorted_rows"}
    and getattr(case, "__module__", None) == __name__
}


def _outcome(case: Callable[[SparkSession], Any], spark: SparkSession, ansi: str) -> Any:
    """The case's normalised result, or the smallaxe exception it raised, with ANSI set."""
    previous = spark.conf.get(ANSI)
    spark.conf.set(ANSI, ansi)
    try:
        return ("ok", _normalise(case(spark)))
    except SmallaxeError as exc:
        return ("raises", type(exc).__name__, str(exc))
    finally:
        spark.conf.set(ANSI, previous)


@pytest.mark.parametrize("name", sorted(CASES))
def test_same_result_with_ansi_mode_on_and_off(spark_session, name):
    case = CASES[name]
    off = _outcome(case, spark_session, "false")
    on = _outcome(case, spark_session, "true")
    assert on == off


def test_the_option_is_restored(spark_session):
    before = spark_session.conf.get(ANSI)
    _outcome(_mape_all_zero, spark_session, "true" if before == "false" else "false")
    assert spark_session.conf.get(ANSI) == before


class TestExpectedValues:
    """The ANSI-independent results are the ones smallaxe documents."""

    def test_mape_skips_zero_and_null_labels(self, spark_session):
        assert _mape_zero_labels(spark_session) == pytest.approx((0.5 + 0.25) / 2 * 100)

    def test_mape_of_only_zero_labels_is_nan(self, spark_session):
        assert math.isnan(_mape_all_zero(spark_session))

    def test_mape_on_integer_columns_does_not_overflow(self, spark_session):
        expected = abs(2 - 1) / 2 + abs(-4 - 5) / 4 + abs(2147483647 + 2147483648) / 2147483647
        assert _mape_integer_columns(spark_session) == pytest.approx(expected / 3 * 100)

    def test_boolean_label_is_compared_as_0_and_1(self, spark_session):
        # TP 1, FP 1 (+1 with a null label), FN 1, TN 1; a null label is never correct.
        acc, prec, rec, f1 = _binary_scores_boolean_label(spark_session)
        assert (acc, rec) == (0.4, 0.5)
        assert prec == pytest.approx(1 / 3)
        assert f1 == pytest.approx(0.4)

    def test_a_string_label_is_compared_as_the_number_it_spells(self, spark_session):
        # Labels 1, 1, (not a number), 0, 0, null against predictions 1, 1, 1, 0, 1, 0.
        acc, prec, rec, f1 = _binary_scores_string_label(spark_session)
        assert acc == pytest.approx(3 / 6)
        assert prec == pytest.approx(2 / 4)
        assert rec == 1.0
        assert f1 == pytest.approx(2 * 0.5 / 1.5)

    def test_string_columns_compare_as_strings(self, spark_session):
        assert _accuracy_string_columns(spark_session) == [pytest.approx(2 / 3), 0.0, 0.0]

    def test_imputer_fill_spellings(self, spark_session):
        outcomes = dict(_imputer_categorical_fill_spellings(spark_session))
        for spelling in ("7.0", "7", " 7 ", "+7", "7."):
            assert outcomes[spelling][1]["code"] == 7, spelling
        for spelling in ("1e3", "7.5", "UNKNOWN", "99999999999"):
            assert outcomes[spelling] == "ValidationError", spelling

    def test_encoder_maps_unseen_to_the_fallback_and_keeps_null(self, spark_session):
        onehot, label = _encoder_unseen_and_null(spark_session)
        assert [row["c"] for row in label] == [0, 2, None]
        assert [row["c_UNKNOWN"] for row in onehot] == [0.0, 1.0, 0.0]


def test_smallaxe_never_sets_the_ansi_option():
    for path in sorted((ROOT / "smallaxe").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert "ansi.enabled" not in text, path
        assert "SPARK_ANSI_SQL_MODE" not in text, path
