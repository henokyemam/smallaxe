"""smallaxe.viz: plotly charts that aggregate in Spark and collect only bins or classes.

The assertions are on the figures' data, not on pixels: cells equal counts computed
independently in Python, curves are monotone and end at (1, 1), histogram and density
bins sum to the rows counted. The private ``_*_table`` / ``_score_bins`` helpers return
the aggregated Spark DataFrame each chart collects, so its row count can be checked
against the bin count.
"""

import dataclasses
import json
import math
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

import plotly.graph_objects as go
import pytest
from pyspark.ml.linalg import Vectors
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StructField, StructType

import smallaxe
from smallaxe import metrics, viz
from smallaxe.auto import AutomatedTraining
from smallaxe.exceptions import (
    ColumnNotFoundError,
    DependencyError,
    ModelNotFittedError,
    ValidationError,
)
from smallaxe.pipeline import Pipeline
from smallaxe.preprocessing import Scaler
from smallaxe.training import algorithm, classifiers, regressors
from smallaxe.viz import _classification, _regression, _style

ROOT = Path(__file__).resolve().parent.parent
CHARTS = [
    "confusion_matrix",
    "roc_curve",
    "precision_recall_curve",
    "residuals",
    "actual_vs_predicted",
    "feature_importances",
    "model_comparison",
]

needs_xgboost = pytest.mark.skipif(
    not algorithm.get("xgboost").available(), reason="xgboost is not installed"
)


@pytest.fixture(autouse=True)
def quiet():
    smallaxe.set_verbosity("quiet")
    yield
    smallaxe.set_verbosity("normal")


# --- Fixtures ---------------------------------------------------------------------------

# Multiclass labels and predictions; class 3 is predicted but never actual, and some rows
# have a null label or prediction.
MULTICLASS_ROWS = [
    (0.0, 0.0),
    (0.0, 0.0),
    (0.0, 1.0),
    (1.0, 1.0),
    (1.0, 1.0),
    (1.0, 1.0),
    (1.0, 2.0),
    (2.0, 2.0),
    (2.0, 0.0),
    (2.0, 3.0),
    (0.0, 3.0),
    (None, 1.0),
    (2.0, None),
]


def _rows_2000():
    """2000 (label, score, value, prediction) rows: scores that separate the classes."""
    rnd = random.Random(7)
    rows = []
    for _ in range(2000):
        positive = rnd.random() < 0.4
        score = min(1.0, max(0.0, rnd.gauss(0.62 if positive else 0.38, 0.18)))
        value = rnd.uniform(0, 100)
        rows.append((float(positive), score, value, value + rnd.gauss(0, 8)))
    return rows


@pytest.fixture(scope="module")
def multiclass_df(spark_session):
    schema = StructType(
        [StructField("label", DoubleType()), StructField("predict_label", DoubleType())]
    )
    return spark_session.createDataFrame(MULTICLASS_ROWS, schema)


@pytest.fixture(scope="module")
def scored_df(spark_session):
    return spark_session.createDataFrame(_rows_2000(), ["label", "score", "y", "pred"])


@pytest.fixture(scope="module")
def fitted_classifier(scored_df):
    return classifiers.random_forest(n_estimators=10, max_depth=4, seed=3).fit(
        scored_df, "label", feature_cols=["score", "y"]
    )


@pytest.fixture(scope="module")
def fitted_regressor(scored_df):
    return regressors.random_forest(n_estimators=5, max_depth=4, seed=3).fit(
        scored_df, "y", feature_cols=["score", "pred", "label"]
    )


def _figures(multiclass_df, scored_df, fitted_regressor):
    """Every chart once, on valid input."""
    return {
        "confusion_matrix": lambda: viz.confusion_matrix(multiclass_df, "label"),
        "roc_curve": lambda: viz.roc_curve(scored_df, "label", "score"),
        "precision_recall_curve": lambda: viz.precision_recall_curve(scored_df, "label", "score"),
        "residuals": lambda: viz.residuals(scored_df, "y", "pred"),
        "actual_vs_predicted": lambda: viz.actual_vs_predicted(scored_df, "y", "pred"),
        "feature_importances": lambda: viz.feature_importances(fitted_regressor),
        "model_comparison": lambda: viz.model_comparison(
            {"a": {"rmse": 1.0}, "b": {"rmse": 2.0}}, "rmse"
        ),
    }


# --- plotly is optional -------------------------------------------------------------------


def test_public_surface_is_the_seven_charts():
    assert sorted(viz.__all__) == sorted(CHARTS)


def test_importing_smallaxe_viz_does_not_import_plotly():
    code = (
        "import json, sys\n"
        "import smallaxe, smallaxe.viz\n"
        "print(json.dumps(sorted(m for m in sys.modules if m.split('.')[0] == 'plotly')))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=str(ROOT), capture_output=True, text=True, check=True
    )
    assert json.loads(result.stdout.strip().splitlines()[-1]) == []


@pytest.mark.parametrize("chart", CHARTS)
def test_every_chart_raises_dependency_error_without_plotly(monkeypatch, chart):
    monkeypatch.setitem(sys.modules, "plotly", None)
    monkeypatch.setitem(sys.modules, "plotly.graph_objects", None)

    with pytest.raises(DependencyError, match=r"plotly is not installed.*smallaxe\[viz\]") as info:
        if chart == "feature_importances":
            viz.feature_importances(None)
        else:
            getattr(viz, chart)(None, "label")

    assert info.value.package == "plotly"
    assert info.value.install_command == "pip install smallaxe[viz]"


def test_a_chart_without_plotly_raises_dependency_error_in_a_fresh_process():
    code = (
        "import sys\n"
        "sys.modules['plotly'] = None\n"
        "from smallaxe import viz\n"
        "from smallaxe.exceptions import DependencyError\n"
        "try:\n"
        "    viz.residuals(None, 'y')\n"
        "except DependencyError as exc:\n"
        "    print(exc.install_command)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=str(ROOT), capture_output=True, text=True, check=True
    )
    assert result.stdout.strip().splitlines()[-1] == "pip install smallaxe[viz]"


# --- Every chart: a Figure that serialises, never shown -------------------------------------


def test_every_chart_returns_a_figure_that_serialises_and_is_not_shown(
    monkeypatch, multiclass_df, scored_df, fitted_regressor
):
    def no_show(self, *args, **kwargs):
        raise AssertionError("a chart called show()")

    monkeypatch.setattr(go.Figure, "show", no_show)
    for name, build in _figures(multiclass_df, scored_df, fitted_regressor).items():
        fig = build()
        assert isinstance(fig, go.Figure), name
        assert fig.layout.template.layout.paper_bgcolor is not None, name
        json.loads(fig.to_json())
        fig.to_dict()


def test_no_chart_collects_more_rows_than_its_bins(
    monkeypatch, multiclass_df, scored_df, fitted_regressor
):
    """Every collect a chart runs returns a bounded number of rows; none uses toPandas."""
    collected = []
    original = DataFrame.collect

    def recording_collect(self):
        rows = original(self)
        collected.append(len(rows))
        return rows

    def no_to_pandas(self):
        raise AssertionError("a chart called toPandas()")

    monkeypatch.setattr(DataFrame, "collect", recording_collect)
    monkeypatch.setattr(DataFrame, "toPandas", no_to_pandas)

    viz.roc_curve(scored_df, "label", "score", n_bins=10)
    viz.precision_recall_curve(scored_df, "label", "score", n_bins=10)
    viz.residuals(scored_df, "y", "pred", n_bins=10)
    assert collected and max(collected) <= 11  # n_bins, plus the bin of invalid rows

    collected.clear()
    viz.actual_vs_predicted(scored_df, "y", "pred", n_bins=5)
    assert collected and max(collected) <= 25

    collected.clear()
    viz.confusion_matrix(multiclass_df, "label")
    assert collected and max(collected) <= 16  # four classes


def test_title_overrides_the_default_and_is_checked(scored_df):
    fig = viz.residuals(scored_df, "y", "pred", title="My residuals")

    assert fig.layout.title.text.split("<br>")[0] == "My residuals"
    with pytest.raises(ValidationError, match="title"):
        viz.residuals(scored_df, "y", "pred", title=3)


def test_layout_uses_the_shared_template_and_column_names(scored_df):
    fig = viz.actual_vs_predicted(scored_df, "y", "pred")

    assert fig.layout.xaxis.title.text == "Actual (y)"
    assert fig.layout.yaxis.title.text == "Predicted (pred)"
    assert fig.data[0].colorscale == go.Figure(go.Heatmap(colorscale="Blues")).data[0].colorscale
    assert fig.layout.template == go.Figure(layout={"template": _style.TEMPLATE}).layout.template


# --- confusion_matrix ---------------------------------------------------------------------


def _expected_matrix(rows):
    complete = [(a, p) for a, p in rows if a is not None and p is not None]
    classes = sorted({a for a, _ in complete} | {p for _, p in complete})
    counts = Counter(complete)
    return classes, [[counts[(a, p)] for p in classes] for a in classes]


def test_confusion_matrix_cells_equal_independent_counts(multiclass_df):
    fig = viz.confusion_matrix(multiclass_df, "label", "predict_label")

    classes, expected = _expected_matrix(MULTICLASS_ROWS)
    heatmap = fig.data[0]
    names = [str(int(value)) for value in classes]
    assert list(heatmap.x) == names == ["0", "1", "2", "3"]
    assert list(heatmap.y) == names
    assert [list(row) for row in heatmap.z] == expected
    # Class 3 is predicted but never actual: its row is all zeros, its column is not.
    assert list(heatmap.z[3]) == [0, 0, 0, 0]
    assert sum(row[3] for row in heatmap.z) == 2
    assert sum(map(sum, heatmap.z)) == 11  # the two rows with a null are left out
    assert fig.layout.yaxis.autorange == "reversed"
    assert [a.text for a in fig.layout.annotations] == [str(c) for row in expected for c in row]


def test_confusion_matrix_normalize_divides_each_row_by_its_actual_total(multiclass_df):
    fig = viz.confusion_matrix(multiclass_df, "label", normalize=True)

    _, expected = _expected_matrix(MULTICLASS_ROWS)
    heatmap = fig.data[0]
    for row, counts in zip(heatmap.z, expected):
        total = sum(counts)
        if total:
            assert sum(row) == pytest.approx(1.0)
            assert list(row) == pytest.approx([count / total for count in counts])
        else:
            assert list(row) == [0.0, 0.0, 0.0, 0.0]
    assert [list(row) for row in heatmap.customdata] == expected  # counts stay in the hover
    assert "customdata" in heatmap.hovertemplate


def test_confusion_matrix_on_a_real_binary_prediction(fitted_classifier, scored_df):
    predictions = fitted_classifier.predict(scored_df)
    fig = viz.confusion_matrix(predictions, "label")

    pairs = Counter(
        (row["label"], row["predict_label"])
        for row in predictions.select("label", "predict_label").collect()
    )
    classes = sorted({a for a, _ in pairs} | {p for _, p in pairs})
    assert [list(row) for row in fig.data[0].z] == [
        [pairs[(a, p)] for p in classes] for a in classes
    ]


def test_confusion_matrix_table_has_one_row_per_pair(multiclass_df):
    table = _classification._confusion_table(multiclass_df, "label", "predict_label")

    assert table.count() == len(
        {(a, p) for a, p in MULTICLASS_ROWS if a is not None and p is not None}
    )


def test_confusion_matrix_refuses_more_than_50_classes(spark_session):
    continuous = spark_session.range(300).select(
        F.col("id").cast("double").alias("label"), (F.col("id") * 1.5).alias("predict_label")
    )
    with pytest.raises(ValidationError, match="at most 50 classes"):
        viz.confusion_matrix(continuous, "label")

    # 51 classes on the diagonal: few pairs, still too many classes.
    diagonal = spark_session.range(51).select(
        F.col("id").cast("double").alias("label"), F.col("id").cast("double").alias("p")
    )
    with pytest.raises(ValidationError, match="at most 50 classes"):
        viz.confusion_matrix(diagonal, "label", "p")


# --- roc_curve and precision_recall_curve ----------------------------------------------------


def _area(xs, ys):
    return sum((xs[i] - xs[i - 1]) * (ys[i] + ys[i - 1]) / 2 for i in range(1, len(xs)))


def test_roc_curve_is_monotone_ends_at_one_and_matches_the_exact_area(scored_df):
    fig = viz.roc_curve(scored_df, "label", "score")

    curve, chance = fig.data
    xs, ys = list(curve.x), list(curve.y)
    assert (xs[0], ys[0]) == (0.0, 0.0)
    assert (xs[-1], ys[-1]) == (1.0, 1.0)
    assert all(b >= a for a, b in zip(xs, xs[1:]))
    assert all(b >= a for a, b in zip(ys, ys[1:]))
    exact = metrics.auc_roc(scored_df, "label", "score")
    assert abs(_area(xs, ys) - exact) < 0.01
    assert curve.name == f"ROC (AUC = {exact:.4f})"
    assert fig.layout.title.text.startswith(f"ROC curve (AUC = {exact:.4f})")
    assert (list(chance.x), list(chance.y)) == ([0.0, 1.0], [0.0, 1.0])
    assert len(xs) <= 200 + 1


def test_precision_recall_curve_recall_rises_to_one_with_the_exact_area(scored_df):
    fig = viz.precision_recall_curve(scored_df, "label", "score")

    curve, baseline = fig.data
    recall, precision = list(curve.x), list(curve.y)
    assert recall[0] == 0.0 and recall[-1] == 1.0
    assert all(b >= a for a, b in zip(recall, recall[1:]))
    assert all(0.0 <= p <= 1.0 for p in precision)
    assert precision[0] == precision[1]  # the start repeats the first point's precision
    exact = metrics.auc_pr(scored_df, "label", "score")
    assert curve.name == f"PR (AUC-PR = {exact:.4f})"
    assert abs(_area(recall, precision) - exact) < 0.02
    rows = scored_df.collect()
    rate = sum(row["label"] for row in rows) / len(rows)
    assert list(baseline.y) == pytest.approx([rate, rate])
    assert precision[-1] == pytest.approx(rate)


@pytest.mark.parametrize("chart", ["roc_curve", "precision_recall_curve"])
def test_curves_accept_the_predict_proba_vector_of_a_real_classifier(
    chart, fitted_classifier, scored_df
):
    probabilities = fitted_classifier.predict_proba(scored_df)
    fig = getattr(viz, chart)(probabilities, "label")

    metric = metrics.auc_roc if chart == "roc_curve" else metrics.auc_pr
    exact = metric(probabilities, "label", "probability")
    assert f"{exact:.4f}" in fig.data[0].name
    assert fig.data[0].x[-1] == 1.0


def test_vector_and_numeric_scores_give_the_same_curve(spark_session):
    rows = [(float(i % 2), (i % 10) / 10.0) for i in range(40)]
    numeric = spark_session.createDataFrame(rows, ["label", "score"])
    vectors = spark_session.createDataFrame(
        [(label, Vectors.dense([1 - score, score])) for label, score in rows],
        ["label", "probability"],
    )

    by_number = viz.roc_curve(numeric, "label", "score", n_bins=10)
    by_vector = viz.roc_curve(vectors, "label", n_bins=10)

    assert list(by_number.data[0].x) == list(by_vector.data[0].x)
    assert list(by_number.data[0].y) == list(by_vector.data[0].y)
    assert by_number.data[0].name == by_vector.data[0].name


@pytest.mark.parametrize("chart", ["roc_curve", "precision_recall_curve"])
@pytest.mark.parametrize("score_type", ["float", "int", "decimal(5,4)", "long"])
def test_curves_accept_a_score_column_of_any_numeric_type(spark_session, chart, score_type):
    # Spark's BinaryClassificationEvaluator, behind metrics.auc_roc / auc_pr, takes only
    # a double or a vector; the chart casts the score so a float32 column works too.
    rows = [(float(i % 2), ((i * 7) % 10) / 10.0 if i % 3 else float(i % 2)) for i in range(60)]
    if score_type in ("int", "long"):
        rows = [(label, float(round(score))) for label, score in rows]
    typed = spark_session.createDataFrame(rows, "label double, score double").withColumn(
        "score", F.col("score").cast(score_type)
    )
    # The same values as doubles (float32 rounding moves 0.3 across a bin edge).
    doubles = typed.withColumn("score", F.col("score").cast("double"))

    fig = getattr(viz, chart)(typed, "label", "score", n_bins=10)

    metric = metrics.auc_roc if chart == "roc_curve" else metrics.auc_pr
    assert f"{metric(doubles, 'label', 'score'):.4f}" in fig.data[0].name
    reference = getattr(viz, chart)(doubles, "label", "score", n_bins=10)
    assert list(fig.data[0].x) == list(reference.data[0].x)
    assert list(fig.data[0].y) == list(reference.data[0].y)


def test_score_bins_are_bounded_by_n_bins_and_leave_out_nulls(spark_session, scored_df):
    table = _classification._score_bins(scored_df, "label", "score", 10, "ROC curve")
    assert table.count() <= 10
    assert table.agg(F.sum("positives") + F.sum("negatives")).first()[0] == 2000

    with_nulls = scored_df.union(
        spark_session.createDataFrame(
            [(None, 0.5, 1.0, 1.0), (1.0, None, 1.0, 1.0), (0.0, float("nan"), 1.0, 1.0)],
            scored_df.schema,
        )
    )
    fig = viz.roc_curve(with_nulls, "label", "score")
    assert "2000 rows" in fig.layout.title.text


@pytest.mark.parametrize("chart", ["roc_curve", "precision_recall_curve"])
@pytest.mark.parametrize("only", [0.0, 1.0])
def test_curves_need_both_classes(spark_session, chart, only):
    one_class = spark_session.createDataFrame([(only, 0.2), (only, 0.8)], ["label", "score"])

    with pytest.raises(ValidationError, match="needs both classes"):
        getattr(viz, chart)(one_class, "label", "score")


@pytest.mark.parametrize("chart", ["roc_curve", "precision_recall_curve"])
def test_curves_reject_bad_scores_and_labels(spark_session, chart):
    draw = getattr(viz, chart)
    labels = spark_session.createDataFrame([(0.0, 0.2), (2.0, 0.8)], ["label", "score"])
    with pytest.raises(ValidationError, match="binary labels 0 and 1"):
        draw(labels, "label", "score")

    out_of_range = spark_session.createDataFrame([(0.0, 0.2), (1.0, 1.5)], ["label", "score"])
    with pytest.raises(ValidationError, match="between 0 and 1"):
        draw(out_of_range, "label", "score")

    three = spark_session.createDataFrame(
        [(0.0, Vectors.dense([0.2, 0.3, 0.5])), (1.0, Vectors.dense([0.1, 0.1, 0.8]))],
        ["label", "probability"],
    )
    with pytest.raises(ValidationError, match="length 2"):
        draw(three, "label")

    strings = spark_session.createDataFrame([(0.0, "a")], ["label", "score"])
    with pytest.raises(ValidationError, match="must be a probability"):
        draw(strings, "label", "score")

    with pytest.raises(ValidationError, match="no row"):
        draw(
            spark_session.createDataFrame([(None, 0.2)], "label double, score double"),
            "label",
            "score",
        )


@pytest.mark.parametrize("n_bins", [9, 10001, 20.0, True, "50", None])
def test_curve_n_bins_must_be_an_int_in_range(scored_df, n_bins):
    with pytest.raises(ValidationError, match="n_bins must be an integer between 10 and 10000"):
        viz.roc_curve(scored_df, "label", "score", n_bins=n_bins)


# --- residuals ------------------------------------------------------------------------------


def test_residual_histogram_counts_sum_to_the_non_null_rows(spark_session, scored_df):
    extra = spark_session.createDataFrame(
        [(1.0, 0.5, None, 3.0), (1.0, 0.5, 4.0, None), (1.0, 0.5, float("nan"), 2.0)],
        scored_df.schema,
    )
    fig = viz.residuals(scored_df.union(extra), "y", "pred", n_bins=30)

    bars = fig.data[0]
    assert sum(bars.y) == 2000
    assert len(bars.x) == 30
    residuals = [row["y"] - row["pred"] for row in scored_df.collect()]
    low, high = min(residuals), max(residuals)
    width = (high - low) / 30
    assert bars.x[0] == pytest.approx(low + width / 2)
    assert bars.x[-1] == pytest.approx(high - width / 2)
    assert list(bars.y) == [
        sum(1 for r in residuals if min(int((r - low) / width), 29) == index) for index in range(30)
    ]
    mean = sum(residuals) / len(residuals)
    std = math.sqrt(sum((r - mean) ** 2 for r in residuals) / (len(residuals) - 1))
    assert f"mean {mean:.4g}" in fig.layout.title.text
    assert f"standard deviation {std:.4g}" in fig.layout.title.text
    assert [shape.x0 for shape in fig.layout.shapes] == [0]


def test_residuals_of_a_constant_column_are_one_bar(spark_session):
    constant = spark_session.createDataFrame([(3.0, 1.0)] * 7, ["label", "predict_label"])
    fig = viz.residuals(constant, "label")

    assert list(fig.data[0].x) == [2.0]
    assert list(fig.data[0].y) == [7]


INF = float("inf")


def test_residuals_leave_out_infinite_values_and_say_how_many(spark_session):
    rows = [(float(i), i + (i % 3) - 1.0) for i in range(30)]
    infinite = [(INF, INF), (INF, 1.0), (0.0, -INF), (-INF, -INF)]
    frame = spark_session.createDataFrame(rows + infinite, "y double, p double")

    fig = viz.residuals(frame, "y", "p", n_bins=3)

    bars = fig.data[0]
    assert sum(bars.y) == 30
    assert all(math.isfinite(value) for value in list(bars.x) + list(bars.width))
    assert list(bars.x) == pytest.approx([-2 / 3, 0.0, 2 / 3])
    assert "30 rows" in fig.layout.title.text
    assert "4 rows with an infinite value left out" in fig.layout.title.text
    assert "nan" not in fig.layout.title.text and "mean inf" not in fig.layout.title.text
    fig.to_json()


def test_residuals_of_only_infinite_values_have_no_rows(spark_session):
    frame = spark_session.createDataFrame([(INF, INF), (INF, 1.0)], "y double, p double")

    with pytest.raises(ValidationError, match="no row"):
        viz.residuals(frame, "y", "p")


def test_residuals_whose_difference_overflows_are_left_out(spark_session):
    # Both values are finite, the residual is not.
    frame = spark_session.createDataFrame(
        [(1e308, -1e308), (1.0, 0.5), (2.0, 2.5)], "y double, p double"
    )
    fig = viz.residuals(frame, "y", "p", n_bins=2)

    assert sum(fig.data[0].y) == 2
    assert "1 rows with an infinite value left out" in fig.layout.title.text


def test_regression_charts_refuse_a_range_too_wide_to_bin(spark_session):
    frame = spark_session.createDataFrame([(1e308, 1e308), (-1e308, -1e308)], "y double, p double")
    with pytest.raises(ValidationError, match="too wide"):
        viz.actual_vs_predicted(frame, "y", "p")
    residual_frame = spark_session.createDataFrame(
        [(1e308, -1e307), (-1e308, 1e307)], "y double, p double"
    )
    with pytest.raises(ValidationError, match="too wide"):
        viz.residuals(residual_frame, "y", "p")


def test_density_leaves_out_infinite_values_and_says_how_many(spark_session):
    rows = [(float(i), float(i)) for i in range(10)]
    frame = spark_session.createDataFrame(
        rows + [(INF, 1.0), (1.0, -INF)], "label double, predict_label double"
    )

    fig = viz.actual_vs_predicted(frame, "label", n_bins=5)

    heatmap = fig.data[0]
    assert sum(map(sum, heatmap.z)) == 10
    assert [row[index] for index, row in enumerate(heatmap.z)] == [2] * 5
    assert list(heatmap.x) == pytest.approx([0.9, 2.7, 4.5, 6.3, 8.1])
    assert "2 rows with an infinite value left out" in fig.layout.title.text
    fig.to_json()


def test_residual_table_is_bounded_by_n_bins(scored_df):
    histogram, stats = _regression._residual_table(scored_df, "y", "pred", 7)

    assert histogram.count() <= 7
    assert stats["count"] == 2000


def test_residuals_on_a_real_regressor(fitted_regressor, scored_df):
    fig = viz.residuals(fitted_regressor.predict(scored_df), "y")

    assert sum(fig.data[0].y) == 2000


# --- actual_vs_predicted ----------------------------------------------------------------------


def test_density_bins_sum_to_the_row_count(spark_session, scored_df):
    with_null = scored_df.union(
        spark_session.createDataFrame([(1.0, 0.5, None, 3.0)], scored_df.schema)
    )
    fig = viz.actual_vs_predicted(with_null, "y", "pred", n_bins=20)

    heatmap, line = fig.data
    assert len(heatmap.z) == 20 and all(len(row) == 20 for row in heatmap.z)
    assert sum(map(sum, heatmap.z)) == 2000
    assert len(heatmap.x) == len(heatmap.y) == 20
    assert line.name == "y = x"
    assert list(line.x) == list(line.y)


def test_density_cells_match_independent_binning(spark_session):
    rows = [(0.0, 0.0), (1.0, 2.0), (2.0, 1.0), (4.0, 4.0), (4.0, 3.9)]
    frame = spark_session.createDataFrame(rows, ["label", "predict_label"])
    fig = viz.actual_vs_predicted(frame, "label", n_bins=5)

    expected = [[0] * 5 for _ in range(5)]
    for actual, predicted in rows:  # range 0..4, width 0.8
        expected[min(int(predicted / 0.8), 4)][min(int(actual / 0.8), 4)] += 1
    assert [list(row) for row in fig.data[0].z] == expected


def test_density_of_a_constant_column_is_one_cell(spark_session):
    constant = spark_session.createDataFrame([(2.0, 2.0)] * 4, ["label", "predict_label"])
    fig = viz.actual_vs_predicted(constant, "label")

    assert [list(row) for row in fig.data[0].z] == [[4]]


def test_density_table_is_bounded_by_n_bins_squared(scored_df):
    cells, stats = _regression._density_table(scored_df, "y", "pred", 5)

    assert cells.count() <= 25
    assert stats["count"] == 2000


@pytest.mark.parametrize("n_bins", [4, 501, 10.0, False])
def test_density_n_bins_must_be_an_int_in_range(scored_df, n_bins):
    with pytest.raises(ValidationError, match="between 5 and 500"):
        viz.actual_vs_predicted(scored_df, "y", "pred", n_bins=n_bins)


@pytest.mark.parametrize("chart", ["residuals", "actual_vs_predicted"])
def test_regression_charts_reject_bad_input(spark_session, scored_df, chart):
    draw = getattr(viz, chart)
    with pytest.raises(ValidationError, match="must be numeric"):
        draw(scored_df.withColumn("s", F.lit("a")), "s", "pred")
    with pytest.raises(ColumnNotFoundError):
        draw(scored_df, "y", "missing")
    with pytest.raises(ValidationError, match="no row"):
        draw(scored_df.limit(0), "y", "pred")
    with pytest.raises(ValidationError, match="no row"):
        draw(spark_session.createDataFrame([(None, 1.0)], "y double, pred double"), "y", "pred")
    with pytest.raises(ValidationError, match="n_bins"):
        draw(scored_df, "y", "pred", n_bins=0)


# --- feature_importances ------------------------------------------------------------------


def _check_importances(fig, importances, top_n):
    bars = fig.data[0]
    # Largest first; equal importances (often 0.0 for unused features) by name.
    expected = sorted(importances.items(), key=lambda item: (-item[1], item[0]))[:top_n]
    assert list(bars.x) == [value for _, value in expected]
    assert list(bars.y) == [name for name, _ in expected]
    assert all(a >= b for a, b in zip(bars.x, bars.x[1:]))
    assert bars.orientation == "h"
    assert fig.layout.yaxis.autorange == "reversed"  # the largest at the top


def test_feature_importances_of_a_random_forest_are_sorted_and_cut(fitted_regressor):
    importances = fitted_regressor.feature_importances

    _check_importances(viz.feature_importances(fitted_regressor), importances, 20)
    fig = viz.feature_importances(fitted_regressor, top_n=2)
    _check_importances(fig, importances, 2)
    assert "top 2 of 3 features" in fig.layout.title.text
    assert fig.layout.title.text.startswith("Feature importances (random_forest)")


@needs_xgboost
def test_feature_importances_of_xgboost(scored_df):
    model = regressors.xgboost(n_estimators=5, max_depth=3, seed=1).fit(
        scored_df, "y", feature_cols=["score", "pred", "label"]
    )
    _check_importances(viz.feature_importances(model, top_n=2), model.feature_importances, 2)


def test_feature_importances_of_a_fitted_pipeline(scored_df):
    model = regressors.random_forest(n_estimators=5, max_depth=3, seed=1)
    pipeline = Pipeline([("scaler", Scaler()), ("model", model)])
    pipeline.fit(scored_df, "y", numerical_cols=["score", "pred"])

    fitted = pipeline.named_steps["model"]
    _check_importances(viz.feature_importances(pipeline), fitted.feature_importances, 20)


def test_feature_importances_errors(fitted_regressor, monkeypatch):
    with pytest.raises(ModelNotFittedError):
        viz.feature_importances(regressors.random_forest())
    with pytest.raises(ModelNotFittedError):
        viz.feature_importances(Pipeline([("model", regressors.random_forest())]))
    with pytest.raises(ValidationError, match="has none"):
        viz.feature_importances(Pipeline([("scaler", Scaler())]))
    with pytest.raises(ValidationError, match="smallaxe Model or Pipeline"):
        viz.feature_importances({"x": 1.0})
    for top_n in (0, True, "3", 2.0):
        with pytest.raises(ValidationError, match="top_n"):
            viz.feature_importances(fitted_regressor, top_n=top_n)

    record = dataclasses.replace(fitted_regressor.algorithm, importances=None)
    monkeypatch.setattr(fitted_regressor, "_algorithm", record)
    with pytest.raises(ValidationError, match="reports no feature importances"):
        viz.feature_importances(fitted_regressor)


# --- model_comparison ----------------------------------------------------------------------


def test_model_comparison_sorts_a_minimised_metric_ascending():
    fig = viz.model_comparison(
        {"a": {"rmse": 3.0}, "b": {"rmse": 1.0}, "c": {"rmse": 2.0}, "d": {"rmse": None}},
        "rmse",
    )

    bars = fig.data[0]
    assert list(bars.x) == ["b", "c", "a"]
    assert list(bars.y) == [1.0, 2.0, 3.0]
    assert list(bars.marker.color) == [_style.ACCENT, _style.NEUTRAL, _style.NEUTRAL]
    assert list(bars.text) == ["1", "2", "3"]
    assert "lower is better" in fig.layout.title.text
    assert "left out (score undefined): d" in fig.layout.title.text


def test_model_comparison_sorts_a_maximised_metric_descending():
    scores = {
        "rf": {"r2": 0.7, "rmse": 2.0},
        "xgb": {"r2": 0.9},
        "nan": {"r2": float("nan")},
        "none": {"rmse": 1.0},
    }
    fig = viz.model_comparison(scores, "r2")

    assert list(fig.data[0].x) == ["xgb", "rf"]
    assert "higher is better" in fig.layout.title.text
    assert "nan, none" in fig.layout.title.text
    # A k-fold key has its metric's orientation.
    kfold = viz.model_comparison({"a": {"mean_r2": 0.1}, "b": {"mean_r2": 0.5}}, "mean_r2")
    assert list(kfold.data[0].x) == ["b", "a"]


def test_model_comparison_of_validation_scores(fitted_regressor, scored_df):
    other = regressors.random_forest(n_estimators=2, max_depth=2, seed=1).fit(
        scored_df, "y", feature_cols=["score"], validation="train_test"
    )
    best = regressors.random_forest(n_estimators=5, max_depth=4, seed=1).fit(
        scored_df, "y", feature_cols=["pred"], validation="train_test"
    )
    results = {"weak": other.validation_scores, "strong": best.validation_scores}
    fig = viz.model_comparison(results, "rmse")

    expected = sorted(results, key=lambda name: results[name]["rmse"])
    assert list(fig.data[0].x) == expected == ["strong", "weak"]


@pytest.fixture(scope="module")
def auto_runs(scored_df):
    algorithms = ["random_forest"] + (["xgboost"] if algorithm.get("xgboost").available() else [])
    small = {"random_forest": {"n_estimators": 5, "max_depth": 3}}
    if "xgboost" in algorithms:
        small["xgboost"] = {"n_estimators": 5}
    smallaxe.set_verbosity("quiet")
    regression = AutomatedTraining(
        task="simple_regression", algorithms=algorithms, params=small, seed=5
    ).fit(scored_df, "y", feature_cols=["score", "pred"])
    binary = AutomatedTraining(task="binary", algorithms=algorithms, params=small, seed=5).fit(
        scored_df, "label", feature_cols=["score", "y"]
    )
    smallaxe.set_verbosity("normal")
    return regression, binary


@pytest.mark.parametrize("index, metric", [(0, None), (0, "r2"), (1, None), (1, "log_loss")])
def test_model_comparison_of_an_automated_training(auto_runs, index, metric):
    auto = auto_runs[index]
    fig = viz.model_comparison(auto, metric)

    metric = metric or auto.metric
    table = auto.comparison
    ok = table[table["status"] == "ok"]
    maximize = metric in ("r2", "auc_roc")
    expected = ok.sort_values(metric, ascending=not maximize)
    assert list(fig.data[0].x) == list(expected["algorithm"])
    assert list(fig.data[0].y) == pytest.approx(list(expected[metric]))
    if metric == auto.metric:
        assert fig.data[0].x[0] == auto.best_algorithm
    json.loads(fig.to_json())


def test_model_comparison_errors(auto_runs):
    with pytest.raises(ModelNotFittedError):
        viz.model_comparison(AutomatedTraining(task="binary"))
    with pytest.raises(ValidationError, match="has no metric 'accuracy'"):
        viz.model_comparison(auto_runs[0], "accuracy")
    with pytest.raises(ValidationError, match="is empty"):
        viz.model_comparison({}, "rmse")
    with pytest.raises(ValidationError, match="needs metric="):
        viz.model_comparison({"a": {"rmse": 1.0}})
    with pytest.raises(ValidationError, match="No entry of results has a 'mae' score"):
        viz.model_comparison({"a": {"rmse": 1.0}}, "mae")
    with pytest.raises(ValidationError, match="knows the direction"):
        viz.model_comparison({"a": {"speed": 1.0}}, "speed")
    with pytest.raises(ValidationError, match="must be a mapping"):
        viz.model_comparison({"a": [1.0]}, "rmse")
    with pytest.raises(ValidationError, match="must be a number"):
        viz.model_comparison({"a": {"rmse": "1.0"}}, "rmse")
    with pytest.raises(ValidationError, match="Every 'rmse' score is undefined"):
        viz.model_comparison({"a": {"rmse": None}}, "rmse")
    with pytest.raises(ValidationError, match="fitted AutomatedTraining or a mapping"):
        viz.model_comparison(42, "rmse")
    with pytest.raises(ValidationError, match="metric must be"):
        viz.model_comparison({"a": {"rmse": 1.0}}, 3)


# --- Wrong-typed and missing inputs: smallaxe exceptions only --------------------------------


def _bad_calls(scored_df, multiclass_df):
    return {
        "confusion_matrix(not a frame)": (
            lambda: viz.confusion_matrix([1], "label"),
            ValidationError,
        ),
        "confusion_matrix(missing)": (
            lambda: viz.confusion_matrix(multiclass_df, "nope"),
            ColumnNotFoundError,
        ),
        "confusion_matrix(label_col=3)": (
            lambda: viz.confusion_matrix(multiclass_df, 3),
            ValidationError,
        ),
        "confusion_matrix(string)": (
            lambda: viz.confusion_matrix(multiclass_df.withColumn("s", F.lit("a")), "s"),
            ValidationError,
        ),
        "confusion_matrix(normalize='yes')": (
            lambda: viz.confusion_matrix(multiclass_df, "label", normalize="yes"),
            ValidationError,
        ),
        "confusion_matrix(empty)": (
            lambda: viz.confusion_matrix(multiclass_df.limit(0), "label"),
            ValidationError,
        ),
        "roc_curve(not a frame)": (lambda: viz.roc_curve(None, "label"), ValidationError),
        "roc_curve(missing)": (lambda: viz.roc_curve(scored_df, "label"), ColumnNotFoundError),
        "roc_curve(empty)": (
            lambda: viz.roc_curve(scored_df.limit(0), "label", "score"),
            ValidationError,
        ),
        "precision_recall_curve(label_col=None)": (
            lambda: viz.precision_recall_curve(scored_df, None, "score"),
            ValidationError,
        ),
        "residuals(not a frame)": (lambda: viz.residuals("df", "y"), ValidationError),
        "actual_vs_predicted(missing)": (
            lambda: viz.actual_vs_predicted(scored_df, "nope", "pred"),
            ColumnNotFoundError,
        ),
        "actual_vs_predicted(title=[])": (
            lambda: viz.actual_vs_predicted(scored_df, "y", "pred", title=[]),
            ValidationError,
        ),
        "feature_importances(None)": (lambda: viz.feature_importances(None), ValidationError),
        "model_comparison(None)": (lambda: viz.model_comparison(None, "rmse"), ValidationError),
    }


# Names Spark would otherwise read as struct access ("a.y") or must escape ("p`q").
ODD_NAMES = {"label": "a.label", "score": "s`core", "y": "a.y", "pred": "p`q`"}


def test_every_dataframe_chart_takes_column_names_with_dots_and_backticks(scored_df):
    odd = scored_df.toDF(*[ODD_NAMES[name] for name in scored_df.columns])

    positives = sum(1 for row in _rows_2000() if row[0] == 1.0)
    matrix = viz.confusion_matrix(odd, "a.label", "a.label").data[0].z
    assert [list(row) for row in matrix] == [[2000 - positives, 0], [0, positives]]
    for chart in (viz.roc_curve, viz.precision_recall_curve):
        assert (
            chart(odd, "a.label", "s`core").data[0].name
            == chart(scored_df, "label", "score").data[0].name
        )
    assert list(viz.residuals(odd, "a.y", "p`q`").data[0].y) == list(
        viz.residuals(scored_df, "y", "pred").data[0].y
    )
    assert viz.actual_vs_predicted(odd, "a.y", "p`q`").data[0].z == (
        viz.actual_vs_predicted(scored_df, "y", "pred").data[0].z
    )


def test_bad_inputs_raise_smallaxe_exceptions(scored_df, multiclass_df):
    failures = []
    for name, (call, expected) in _bad_calls(scored_df, multiclass_df).items():
        try:
            call()
        except expected:
            continue
        except Exception as exc:  # any other error is a failure to report
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
        else:
            failures.append(f"{name}: no error")
    assert failures == []
