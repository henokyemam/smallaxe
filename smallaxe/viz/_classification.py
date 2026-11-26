"""Classification charts: the confusion matrix, the ROC curve and the precision-recall curve.

Each one runs a single ``groupBy`` in Spark and collects only its result: one row per
(actual, predicted) class pair for the confusion matrix (at most ``50 * 50 + 1`` rows
are collected), one row per score bin for the curves (at most ``n_bins + 1``). The curves
are then accumulated on the driver; their area is the exact one from
:func:`smallaxe.metrics.auc_roc` / :func:`smallaxe.metrics.auc_pr`.
"""

import math
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from pyspark.ml.functions import vector_to_array
from pyspark.ml.linalg import VectorUDT
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from smallaxe.exceptions import ValidationError
from smallaxe.metrics import auc_pr, auc_roc
from smallaxe.metrics._checks import require_numeric, validate_columns
from smallaxe.metrics.classification import _check_auc_columns
from smallaxe.viz import _frames, _style

if TYPE_CHECKING:
    from plotly.graph_objects import Figure

__all__ = ["confusion_matrix", "roc_curve", "precision_recall_curve"]

# More distinct values than this in the label or prediction column is refused: it is
# usually a continuous column passed by mistake, and its matrix would be huge.
_MAX_CLASSES = 50

# The score bins of the ROC and precision-recall curves.
_MIN_SCORE_BINS, _MAX_SCORE_BINS = 10, 10000

# One collected score bin: (bin index, positives, negatives).
_Bin = Tuple[int, int, int]


# --- Confusion matrix ---------------------------------------------------------------


def _confusion_table(dataframe: DataFrame, label_col: str, prediction_col: str) -> DataFrame:
    """One row per (actual, predicted) pair present in the data, with its count.

    Rows where either column is null or NaN are left out.

    Raises:
        ValidationError: If ``dataframe`` is not a DataFrame or a column is not numeric.
        ColumnNotFoundError: If a column is missing.
    """
    validate_columns(dataframe, label_col, prediction_col)
    require_numeric(dataframe, "confusion_matrix", label_col, prediction_col)
    actual = _frames.column(label_col).cast("double")
    predicted = _frames.column(prediction_col).cast("double")
    return (
        dataframe.where(_frames.present(actual) & _frames.present(predicted))
        .groupBy(actual.alias("actual"), predicted.alias("predicted"))
        .count()
    )


def _class_name(value: float) -> str:
    """A class value as an axis label: ``1.0`` is ``"1"``, ``0.5`` stays ``"0.5"``."""
    return str(int(value)) if value.is_integer() else repr(value)


def confusion_matrix(
    dataframe: DataFrame,
    label_col: str,
    prediction_col: str = "predict_label",
    *,
    normalize: bool = False,
    title: Optional[str] = None,
) -> "Figure":
    """A heatmap of actual classes (rows) against predicted classes (columns).

    Works for binary and multiclass predictions. Every class that appears in either
    column is on both axes, in ascending order; a pair that never occurs is 0. Rows with
    a null or NaN label or prediction are left out. The counts come from one
    ``groupBy(label, prediction).count()`` in Spark.

    Args:
        dataframe: PySpark DataFrame with the label and prediction columns, such as the
            output of ``model.predict``.
        label_col: The actual class column (numeric).
        prediction_col: The predicted class column (numeric). Default 'predict_label'.
        normalize: Show each row divided by its actual class's total (rows sum to 1; a
            class that is never actual stays 0). The counts stay in the hover text.
            Default False: show the counts.
        title: The chart title. Defaults to "Confusion matrix".

    Returns:
        A ``plotly.graph_objects.Figure`` holding one heatmap; its ``z`` is the matrix
        (counts, or proportions with ``normalize=True``) and its ``customdata`` the counts.

    Raises:
        DependencyError: If plotly is not installed (``pip install smallaxe[viz]``).
        ValidationError: If ``dataframe`` is not a DataFrame, a column is not numeric,
            ``normalize`` is not a bool, ``title`` is not a string, no row has both a
            label and a prediction, or the columns hold more than 50 distinct classes.
        ColumnNotFoundError: If a column is missing.

    Example:
        >>> from smallaxe import viz
        >>> viz.confusion_matrix(model.predict(test), label_col="label").show()
    """
    go = _style.plotly()
    _style.check_bool("normalize", normalize)
    _style.check_title(title)
    table = _confusion_table(dataframe, label_col, prediction_col)
    # With at most 50 classes there are at most 50 * 50 pairs, so one more row than that
    # proves there are too many without collecting the rest.
    limit = _MAX_CLASSES * _MAX_CLASSES
    rows = _frames.collect(table, "Cannot build the confusion matrix", limit=limit + 1)
    if not rows:
        raise ValidationError(
            f"Cannot build the confusion matrix: no row has both a '{label_col}' and a "
            f"'{prediction_col}' value."
        )
    classes = sorted(
        {float(row["actual"]) for row in rows} | {float(row["predicted"]) for row in rows}
    )
    if len(rows) > limit or len(classes) > _MAX_CLASSES:
        raise ValidationError(
            f"confusion_matrix supports at most {_MAX_CLASSES} classes, but '{label_col}' "
            f"and '{prediction_col}' hold more. Is the label continuous? For a regression "
            "use viz.residuals or viz.actual_vs_predicted."
        )

    position = {value: index for index, value in enumerate(classes)}
    counts = [[0] * len(classes) for _ in classes]
    for row in rows:
        counts[position[float(row["actual"])]][position[float(row["predicted"])]] += int(
            row["count"]
        )
    if normalize:
        values: List[List[Any]] = []
        for counted in counts:
            total = sum(counted)
            values.append([count / total if total else 0.0 for count in counted])
    else:
        values = [list(counted) for counted in counts]

    names = [_class_name(value) for value in classes]
    fig = _style.figure(
        go,
        title=title,
        default_title="Confusion matrix" + (" (share of each actual class)" if normalize else ""),
        x_title=f"Predicted ({_style.text(prediction_col)})",
        y_title=f"Actual ({_style.text(label_col)})",
    )
    hover = "Actual %{y}<br>Predicted %{x}<br>Count %{customdata}"
    if normalize:
        hover += "<br>Share of actual %{z:.3f}"
    fig.add_trace(
        go.Heatmap(
            z=values,
            x=names,
            y=names,
            customdata=counts,
            colorscale=_style.COLORSCALE,
            colorbar={"title": {"text": "Share" if normalize else "Count"}},
            hovertemplate=hover + "<extra></extra>",
        )
    )
    # Cell labels as annotations (a heatmap's own texttemplate needs plotly 5.5).
    peak = max(max(row) for row in values) or 1
    fig.update_layout(
        annotations=[
            {
                "x": names[column],
                "y": names[line],
                "text": f"{value:.2f}" if normalize else str(value),
                "showarrow": False,
                "font": {"color": "white" if value > peak / 2 else "black"},
            }
            for line, row in enumerate(values)
            for column, value in enumerate(row)
        ]
    )
    fig.update_xaxes(type="category")
    fig.update_yaxes(type="category", autorange="reversed")
    return fig


# --- ROC and precision-recall curves -------------------------------------------------


def _scored(dataframe: DataFrame, label_col: str, probability_col: str) -> DataFrame:
    """The rows a curve uses, as two columns: ``label`` and ``score``.

    ``label`` is the label as a double; ``score`` is the probability vector as it is, or
    a numeric score cast to double (Spark's ``BinaryClassificationEvaluator``, behind
    the exact area, takes only a double or a vector). Rows with a null or NaN label or a
    null score are left out; a numeric score that is NaN is left out too, a vector's
    entries are checked by :func:`_score_bins`. The fixed names also keep the
    evaluator away from the user's column names, which may hold a dot.
    """
    label = _frames.column(label_col)
    probability = _frames.column(probability_col)
    if isinstance(dataframe.schema[probability_col].dataType, VectorUDT):
        has_score = probability.isNotNull()
        score = probability
    else:
        has_score = _frames.present(probability)
        score = probability.cast("double")
    return dataframe.where(_frames.present(label) & has_score).select(
        label.cast("double").alias("label"), score.alias("score")
    )


def _score_bins(
    dataframe: DataFrame, label_col: str, probability_col: str, n_bins: int, chart: str
) -> DataFrame:
    """Positives and negatives per score bin, plus counts of the rows that are invalid.

    The score is the positive-class probability: element 1 of a probability vector, or
    the value of a numeric column (of any numeric type). Rows with a null or NaN label,
    or a null score (a NaN numeric score too), are left out. The rest are bucketed into
    ``n_bins`` equal-width bins over [0, 1]. One ``groupBy`` returns at most
    ``n_bins + 1`` rows: one per bin that holds a row, and a null bin for the rows that
    cannot be binned. Each row also counts the rows whose label is not 0/1
    (``bad_labels``), whose vector does not have length 2 (``bad_sizes``) or whose score
    is outside [0, 1] (``bad_scores``).

    Raises:
        ValidationError: If ``dataframe`` is not a DataFrame, the label is not numeric,
            or the probability column is neither numeric nor a vector.
        ColumnNotFoundError: If a column is missing.
    """
    validate_columns(dataframe, label_col, probability_col)
    _check_auc_columns(dataframe, label_col, probability_col, chart)
    scored = _scored(dataframe, label_col, probability_col)
    if isinstance(dataframe.schema[probability_col].dataType, VectorUDT):
        values = vector_to_array(F.col("score"))
        size = F.size(values)
        score = F.when(size == 2, values[1])
    else:
        size = F.lit(2)
        score = F.col("score")

    rows = scored.select(F.col("label"), score.alias("score"), size.alias("size"))
    bad_size = F.col("size") != 2
    value = F.col("score")
    bad_score = ~bad_size & (value.isNull() | F.isnan(value) | (value < 0) | (value > 1))
    bad_label = ~F.col("label").isin(0.0, 1.0)
    usable = ~bad_size & ~bad_score & ~bad_label
    # Equal-width bins over [0, 1]; a score of exactly 1 belongs to the top bin.
    bucket = F.when(usable, F.least(F.floor(value * n_bins).cast("int"), F.lit(n_bins - 1)))
    return rows.groupBy(bucket.alias("bin")).agg(
        F.sum(F.when(usable & (F.col("label") == 1), 1).otherwise(0)).alias("positives"),
        F.sum(F.when(usable & (F.col("label") == 0), 1).otherwise(0)).alias("negatives"),
        F.sum(bad_label.cast("int")).alias("bad_labels"),
        F.sum(bad_size.cast("int")).alias("bad_sizes"),
        F.sum(bad_score.cast("int")).alias("bad_scores"),
    )


def _curve_input(
    dataframe: DataFrame, label_col: str, probability_col: str, n_bins: Any, chart: str
) -> Tuple[List[_Bin], int, int, DataFrame]:
    """The checked score bins of a curve, highest bin first.

    Returns:
        ``(bins, positives, negatives, scored rows)``: the non-empty bins as ``(bin,
        positives, negatives)`` from the highest score down, the totals, and the input's
        rows with a label and a score as the columns ``label`` and ``score`` (see
        :func:`_scored`), for the exact area.

    Raises:
        ValidationError: If ``n_bins`` is not an int in [10, 10000], an input is
            invalid (see :func:`_score_bins`), a label is not 0/1, a vector does not have
            length 2, a score is outside [0, 1], no row has a label and a score, or only
            one class is present.
        ColumnNotFoundError: If a column is missing.
    """
    n_bins = _style.check_int("n_bins", n_bins, _MIN_SCORE_BINS, _MAX_SCORE_BINS)
    table = _score_bins(dataframe, label_col, probability_col, n_bins, chart)
    rows = _frames.collect(table, f"Cannot build the {chart}")
    bad: Dict[str, int] = {
        name: sum(int(row[name]) for row in rows)
        for name in ("bad_labels", "bad_sizes", "bad_scores")
    }
    if bad["bad_labels"]:
        raise ValidationError(
            f"The {chart} needs binary labels 0 and 1, but '{label_col}' holds "
            f"{bad['bad_labels']} rows with other values."
        )
    if bad["bad_sizes"]:
        raise ValidationError(
            f"The {chart} needs one probability vector of length 2 per row (a binary "
            f"classifier's predict_proba), but '{probability_col}' holds "
            f"{bad['bad_sizes']} vectors of another length."
        )
    if bad["bad_scores"]:
        raise ValidationError(
            f"The {chart} needs positive-class scores between 0 and 1, but "
            f"'{probability_col}' holds {bad['bad_scores']} rows outside that range."
        )
    bins = sorted(
        ((int(row["bin"]), int(row["positives"]), int(row["negatives"])) for row in rows),
        reverse=True,
    )
    positives = sum(count for _, count, _ in bins)
    negatives = sum(count for _, _, count in bins)
    if positives + negatives == 0:
        raise ValidationError(
            f"Cannot build the {chart}: no row has both a '{label_col}' label and a "
            f"'{probability_col}' score."
        )
    if positives == 0 or negatives == 0:
        present = "1 (positive)" if negatives == 0 else "0 (negative)"
        raise ValidationError(
            f"The {chart} needs both classes, but '{label_col}' holds only label {present}."
        )
    return bins, positives, negatives, _scored(dataframe, label_col, probability_col)


def _exact_area(area: float, chart: str, label_col: str) -> float:
    """``area`` from the metric, or ValidationError when it is undefined (NaN)."""
    if math.isnan(area):
        raise ValidationError(f"The {chart} needs both classes in '{label_col}'.")
    return area


def _roc_points(bins: List[_Bin], positives: int, negatives: int, n_bins: int) -> Any:
    """``(fpr, tpr, hover text)``, from (0, 0) through each bin down to (1, 1)."""
    fpr, tpr, hover = [0.0], [0.0], ["no row above the threshold"]
    true_positives = false_positives = 0
    for index, positive, negative in bins:
        true_positives += positive
        false_positives += negative
        fpr.append(false_positives / negatives)
        tpr.append(true_positives / positives)
        hover.append(f"score ≥ {index / n_bins:.4g}")
    return fpr, tpr, hover


def roc_curve(
    dataframe: DataFrame,
    label_col: str,
    probability_col: str = "probability",
    *,
    n_bins: int = 200,
    title: Optional[str] = None,
) -> "Figure":
    """The ROC curve of a binary classifier, with the exact AUC-ROC in the legend.

    The positive-class score is bucketed into ``n_bins`` equal-width bins over [0, 1]
    with one ``groupBy`` in Spark, and the curve is accumulated from the highest bin
    down: one point per non-empty bin, from (0, 0) to (1, 1). The area shown is
    :func:`smallaxe.metrics.auc_roc` on the same rows, not the binned approximation.
    Rows with a null or NaN label or a null score are left out.

    Args:
        dataframe: PySpark DataFrame with the label and the score, such as the output of
            ``model.predict_proba``.
        label_col: The binary label column, 0 or 1.
        probability_col: The probability vector ``predict_proba`` writes (its element 1,
            the probability of class 1, is the score; vectors must have length 2), or a
            numeric column of any numeric type holding the positive-class score in
            [0, 1]. Default 'probability'.
        n_bins: Number of score bins, an integer in [10, 10000]. Default 200.
        title: The chart title. Defaults to "ROC curve (AUC = <auc_roc>)".

    Returns:
        A ``plotly.graph_objects.Figure`` with the curve (trace 0, named with the AUC)
        and the chance diagonal (trace 1).

    Raises:
        DependencyError: If plotly is not installed (``pip install smallaxe[viz]``).
        ValidationError: If ``dataframe`` is not a DataFrame, the label is not numeric
            or holds values other than 0 and 1, the score column is neither numeric nor
            a vector, a vector does not have length 2, a score is outside [0, 1],
            ``n_bins`` or ``title`` is invalid, no row has a label and a score, or only
            one class is present (the curve needs both).
        ColumnNotFoundError: If a column is missing.

    Example:
        >>> from smallaxe import viz
        >>> viz.roc_curve(model.predict_proba(test), label_col="churn").show()
    """
    go = _style.plotly()
    _style.check_title(title)
    chart = "ROC curve"
    bins, positives, negatives, scored = _curve_input(
        dataframe, label_col, probability_col, n_bins, chart
    )
    area = _exact_area(auc_roc(scored, "label", "score"), chart, label_col)
    fpr, tpr, hover = _roc_points(bins, positives, negatives, n_bins)

    fig = _style.figure(
        go,
        title=title,
        default_title=f"ROC curve (AUC = {area:.4f})",
        x_title="False positive rate",
        y_title="True positive rate",
        subtitle=(
            f"{positives + negatives} rows, {positives} positive; "
            f"{_style.text(probability_col)} in {n_bins} score bins"
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            text=hover,
            mode="lines",
            name=f"ROC (AUC = {area:.4f})",
            line={"color": _style.ACCENT, "width": 2},
            hovertemplate="FPR %{x:.3f}<br>TPR %{y:.3f}<br>%{text}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0.0, 1.0],
            y=[0.0, 1.0],
            mode="lines",
            name="Chance",
            line={"color": _style.NEUTRAL, "dash": "dash"},
            hoverinfo="skip",
        )
    )
    fig.update_xaxes(range=[-0.01, 1.01])
    fig.update_yaxes(range=[-0.01, 1.01])
    return fig


def _pr_points(bins: List[_Bin], positives: int, n_bins: int) -> Any:
    """``(recall, precision, hover text)``: each bin down, after a start at recall 0.

    The start repeats the first point's precision at recall 0, as Spark's own
    precision-recall curve does (``BinaryClassificationMetrics.pr``).
    """
    recall: List[float] = []
    precision: List[float] = []
    hover: List[str] = []
    true_positives = predicted = 0
    for index, positive, negative in bins:
        true_positives += positive
        predicted += positive + negative
        recall.append(true_positives / positives)
        precision.append(true_positives / predicted)
        hover.append(f"score ≥ {index / n_bins:.4g}")
    return [0.0] + recall, precision[:1] + precision, [hover[0]] + hover


def precision_recall_curve(
    dataframe: DataFrame,
    label_col: str,
    probability_col: str = "probability",
    *,
    n_bins: int = 200,
    title: Optional[str] = None,
) -> "Figure":
    """The precision-recall curve of a binary classifier, with the exact AUC-PR.

    Binned like :func:`roc_curve`: one ``groupBy`` over ``n_bins`` equal-width score
    bins, accumulated from the highest bin down, one point per non-empty bin, starting at
    recall 0 with the first point's precision (as Spark's curve does). The area shown is
    :func:`smallaxe.metrics.auc_pr` on the same rows. The baseline is the positive rate:
    the precision of a classifier that guesses.

    Args:
        dataframe: PySpark DataFrame with the label and the score, such as the output of
            ``model.predict_proba``.
        label_col: The binary label column, 0 or 1.
        probability_col: The probability vector ``predict_proba`` writes (its element 1
            is the score; vectors must have length 2), or a numeric column of any
            numeric type holding the positive-class score in [0, 1]. Default
            'probability'.
        n_bins: Number of score bins, an integer in [10, 10000]. Default 200.
        title: The chart title. Defaults to "Precision-recall curve (AUC-PR = <auc_pr>)".

    Returns:
        A ``plotly.graph_objects.Figure`` with the curve (trace 0, named with the AUC-PR)
        and the positive-rate baseline (trace 1).

    Raises:
        DependencyError: If plotly is not installed (``pip install smallaxe[viz]``).
        ValidationError: As :func:`roc_curve` raises it; in particular when only one
            class is present (the curve needs both).
        ColumnNotFoundError: If a column is missing.

    Example:
        >>> from smallaxe import viz
        >>> viz.precision_recall_curve(model.predict_proba(test), label_col="churn").show()
    """
    go = _style.plotly()
    _style.check_title(title)
    chart = "precision-recall curve"
    bins, positives, negatives, scored = _curve_input(
        dataframe, label_col, probability_col, n_bins, chart
    )
    area = _exact_area(auc_pr(scored, "label", "score"), chart, label_col)
    recall, precision, hover = _pr_points(bins, positives, n_bins)
    rate = positives / (positives + negatives)

    fig = _style.figure(
        go,
        title=title,
        default_title=f"Precision-recall curve (AUC-PR = {area:.4f})",
        x_title="Recall",
        y_title="Precision",
        subtitle=(
            f"{positives + negatives} rows, {positives} positive; "
            f"{_style.text(probability_col)} in {n_bins} score bins"
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=recall,
            y=precision,
            text=hover,
            mode="lines",
            name=f"PR (AUC-PR = {area:.4f})",
            line={"color": _style.ACCENT, "width": 2},
            hovertemplate="Recall %{x:.3f}<br>Precision %{y:.3f}<br>%{text}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0.0, 1.0],
            y=[rate, rate],
            mode="lines",
            name=f"Baseline (positive rate {rate:.4f})",
            line={"color": _style.NEUTRAL, "dash": "dash"},
            hoverinfo="skip",
        )
    )
    fig.update_xaxes(range=[-0.01, 1.01])
    fig.update_yaxes(range=[-0.01, 1.01])
    return fig
