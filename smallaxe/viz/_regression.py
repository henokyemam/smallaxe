"""Regression charts: the residual histogram and the actual-versus-predicted density.

Each runs two Spark jobs and collects only aggregates: one global aggregate (count,
range, mean, standard deviation: one row), then one ``groupBy`` over the bin indices
(at most ``n_bins`` rows for the histogram, ``n_bins * n_bins`` for the density).
Rows where the label or the prediction is null or NaN are left out and never counted.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

from smallaxe.exceptions import ValidationError
from smallaxe.metrics._checks import require_numeric, validate_columns
from smallaxe.viz import _frames, _style

if TYPE_CHECKING:
    from plotly.graph_objects import Figure

__all__ = ["residuals", "actual_vs_predicted"]

_MIN_HISTOGRAM_BINS, _MAX_HISTOGRAM_BINS = 1, 10000
_MIN_DENSITY_BINS, _MAX_DENSITY_BINS = 5, 500


def _complete(dataframe: DataFrame, label_col: str, prediction_col: str, chart: str) -> DataFrame:
    """The rows with a number in both columns, after checking the columns.

    Raises:
        ValidationError: If ``dataframe`` is not a DataFrame or a column is not numeric.
        ColumnNotFoundError: If a column is missing.
    """
    validate_columns(dataframe, label_col, prediction_col)
    require_numeric(dataframe, chart, label_col, prediction_col)
    return dataframe.where(
        _frames.present(F.col(label_col)) & _frames.present(F.col(prediction_col))
    )


def _no_rows(chart: str, label_col: str, prediction_col: str) -> ValidationError:
    return ValidationError(
        f"Cannot build the {chart}: no row has both a '{label_col}' and a "
        f"'{prediction_col}' value."
    )


def _bin_index(value: Column, low: float, width: float, n_bins: int) -> Column:
    """The equal-width bin of ``value``: ``floor((value - low) / width)``, the top edge in
    the last bin. A zero ``width`` (a constant column) puts everything in bin 0."""
    if width == 0:
        return F.lit(0)
    return F.least(F.floor((value - F.lit(low)) / F.lit(width)).cast("int"), F.lit(n_bins - 1))


# --- Residuals ------------------------------------------------------------------------


def _residual_table(
    dataframe: DataFrame, label_col: str, prediction_col: str, n_bins: int
) -> Tuple[DataFrame, Dict[str, Any]]:
    """The residual histogram as a small DataFrame, and the residuals' statistics.

    Runs one global aggregate (count, min, max, mean, standard deviation of
    ``label - prediction``); the histogram is returned unevaluated: one row per non-empty
    bin (``bin``, ``count``), at most ``n_bins`` rows.

    Returns:
        ``(histogram, stats)``, where ``stats`` holds ``count``, ``low``, ``high``,
        ``mean``, ``std`` (0.0 for a single row) and ``width`` (0.0 when every residual
        is equal).

    Raises:
        ValidationError: If an input is invalid or no row has both values.
        ColumnNotFoundError: If a column is missing.
    """
    rows = _complete(dataframe, label_col, prediction_col, "residuals").select(
        (F.col(label_col).cast("double") - F.col(prediction_col).cast("double")).alias("residual")
    )
    residual = F.col("residual")
    row = _frames.summary(
        rows.select(
            F.count(residual).alias("count"),
            F.min(residual).alias("low"),
            F.max(residual).alias("high"),
            F.mean(residual).alias("mean"),
            F.stddev_samp(residual).alias("std"),
        ),
        "Cannot build the residuals chart",
    )
    count = int(row["count"])
    if count == 0:
        raise _no_rows("residuals chart", label_col, prediction_col)
    low, high = float(row["low"]), float(row["high"])
    stats: Dict[str, Any] = {
        "count": count,
        "low": low,
        "high": high,
        "mean": float(row["mean"]),
        "std": float(row["std"]) if row["std"] is not None else 0.0,
        "width": (high - low) / n_bins,
    }
    histogram = rows.groupBy(_bin_index(residual, low, stats["width"], n_bins).alias("bin")).count()
    return histogram, stats


def residuals(
    dataframe: DataFrame,
    label_col: str,
    prediction_col: str = "predict_label",
    *,
    n_bins: int = 50,
    title: Optional[str] = None,
) -> "Figure":
    """A histogram of the residuals ``label - prediction``, with a line at 0.

    ``n_bins`` equal-width bins span the smallest to the largest residual; when every
    residual is equal there is one bar. The mean and standard deviation are in the
    subtitle. Rows with a null or NaN label or prediction are left out and not counted.

    Args:
        dataframe: PySpark DataFrame with the label and prediction columns, such as the
            output of ``model.predict``.
        label_col: The actual value column (numeric).
        prediction_col: The predicted value column (numeric). Default 'predict_label'.
        n_bins: Number of bins, an integer in [1, 10000]. Default 50.
        title: The chart title. Defaults to "Residuals (<label_col> - <prediction_col>)".

    Returns:
        A ``plotly.graph_objects.Figure`` holding one bar trace: ``x`` the bin centres,
        ``y`` the counts (summing to the rows counted).

    Raises:
        DependencyError: If plotly is not installed (``pip install smallaxe[viz]``).
        ValidationError: If ``dataframe`` is not a DataFrame, a column is not numeric,
            ``n_bins`` or ``title`` is invalid, or no row has both values.
        ColumnNotFoundError: If a column is missing.

    Example:
        >>> from smallaxe import viz
        >>> viz.residuals(model.predict(test), label_col="price").show()
    """
    go = _style.plotly()
    _style.check_title(title)
    n_bins = _style.check_int("n_bins", n_bins, _MIN_HISTOGRAM_BINS, _MAX_HISTOGRAM_BINS)
    histogram, stats = _residual_table(dataframe, label_col, prediction_col, n_bins)
    counts = {
        int(row["bin"]): int(row["count"])
        for row in _frames.collect(histogram, "Cannot build the residuals chart")
    }
    width = stats["width"]
    if width == 0:
        centres, heights, bar_width = [stats["low"]], [counts.get(0, 0)], 1.0
    else:
        centres = [stats["low"] + (index + 0.5) * width for index in range(n_bins)]
        heights = [counts.get(index, 0) for index in range(n_bins)]
        bar_width = width

    fig = _style.figure(
        go,
        title=title,
        default_title=(f"Residuals ({_style.text(label_col)} - {_style.text(prediction_col)})"),
        x_title="Residual",
        y_title="Rows",
        subtitle=(
            f"mean {_style.number(stats['mean'])}, standard deviation "
            f"{_style.number(stats['std'])}, {stats['count']} rows"
        ),
    )
    fig.add_trace(
        go.Bar(
            x=centres,
            y=heights,
            width=[bar_width] * len(centres),
            marker={"color": _style.ACCENT},
            name="Residuals",
            hovertemplate="Residual %{x:.4g}<br>Rows %{y}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_color=_style.NEUTRAL, line_dash="dash")
    fig.update_layout(bargap=0.05, showlegend=False)
    return fig


# --- Actual versus predicted -----------------------------------------------------------


def _density_table(
    dataframe: DataFrame, label_col: str, prediction_col: str, n_bins: int
) -> Tuple[DataFrame, Dict[str, Any]]:
    """The 2-D binned counts as a small DataFrame, and the shared range.

    One global aggregate finds the joint range of label and prediction; the counts are
    returned unevaluated: one row per non-empty cell (``x_bin`` for the label, ``y_bin``
    for the prediction, ``count``), at most ``n_bins * n_bins`` rows.

    Returns:
        ``(cells, stats)``, where ``stats`` holds ``count``, ``low``, ``high`` and
        ``width`` (0.0 when both columns hold one and the same value).

    Raises:
        ValidationError: If an input is invalid or no row has both values.
        ColumnNotFoundError: If a column is missing.
    """
    actual = F.col(label_col).cast("double")
    predicted = F.col(prediction_col).cast("double")
    rows = _complete(dataframe, label_col, prediction_col, "actual_vs_predicted").select(
        actual.alias("actual"), predicted.alias("predicted")
    )
    row = _frames.summary(
        rows.select(
            F.count("actual").alias("count"),
            F.least(F.min("actual"), F.min("predicted")).alias("low"),
            F.greatest(F.max("actual"), F.max("predicted")).alias("high"),
        ),
        "Cannot build the actual-vs-predicted chart",
    )
    count = int(row["count"])
    if count == 0:
        raise _no_rows("actual-vs-predicted chart", label_col, prediction_col)
    low, high = float(row["low"]), float(row["high"])
    width = (high - low) / n_bins
    cells = rows.groupBy(
        _bin_index(F.col("actual"), low, width, n_bins).alias("x_bin"),
        _bin_index(F.col("predicted"), low, width, n_bins).alias("y_bin"),
    ).count()
    return cells, {"count": count, "low": low, "high": high, "width": width}


def actual_vs_predicted(
    dataframe: DataFrame,
    label_col: str,
    prediction_col: str = "predict_label",
    *,
    n_bins: int = 50,
    title: Optional[str] = None,
) -> "Figure":
    """A 2-D density of actual (x) against predicted (y) values, with the line y = x.

    Both axes share ``n_bins`` equal-width bins over the joint range of the two columns,
    so a perfect model lies on the diagonal. Each cell counts the rows in it (one
    ``groupBy`` over the two bin indices); when both columns hold a single value there is
    one cell. Rows with a null or NaN label or prediction are left out.

    Args:
        dataframe: PySpark DataFrame with the label and prediction columns, such as the
            output of ``model.predict``.
        label_col: The actual value column (numeric), on the x axis.
        prediction_col: The predicted value column (numeric), on the y axis. Default
            'predict_label'.
        n_bins: Bins per axis, an integer in [5, 500]. Default 50.
        title: The chart title. Defaults to "Actual vs predicted".

    Returns:
        A ``plotly.graph_objects.Figure`` with the density heatmap (trace 0: ``z[i][j]``
        counts the rows in predicted bin ``i`` and actual bin ``j``, summing to the rows
        counted) and the y = x line (trace 1).

    Raises:
        DependencyError: If plotly is not installed (``pip install smallaxe[viz]``).
        ValidationError: If ``dataframe`` is not a DataFrame, a column is not numeric,
            ``n_bins`` or ``title`` is invalid, or no row has both values.
        ColumnNotFoundError: If a column is missing.

    Example:
        >>> from smallaxe import viz
        >>> viz.actual_vs_predicted(model.predict(test), label_col="price").show()
    """
    go = _style.plotly()
    _style.check_title(title)
    n_bins = _style.check_int("n_bins", n_bins, _MIN_DENSITY_BINS, _MAX_DENSITY_BINS)
    cells, stats = _density_table(dataframe, label_col, prediction_col, n_bins)
    low, width = stats["low"], stats["width"]
    size = 1 if width == 0 else n_bins
    counts: List[List[int]] = [[0] * size for _ in range(size)]
    for row in _frames.collect(cells, "Cannot build the actual-vs-predicted chart"):
        counts[int(row["y_bin"])][int(row["x_bin"])] = int(row["count"])
    centres = [low] if width == 0 else [low + (index + 0.5) * width for index in range(size)]

    fig = _style.figure(
        go,
        title=title,
        default_title="Actual vs predicted",
        x_title=f"Actual ({_style.text(label_col)})",
        y_title=f"Predicted ({_style.text(prediction_col)})",
        subtitle=f"{stats['count']} rows in {size} x {size} bins",
    )
    fig.add_trace(
        go.Heatmap(
            z=counts,
            x=centres,
            y=centres,
            colorscale=_style.COLORSCALE,
            colorbar={"title": {"text": "Rows"}},
            name="Rows",
            hovertemplate="Actual %{x:.4g}<br>Predicted %{y:.4g}<br>Rows %{z}<extra></extra>",
        )
    )
    edge_low, edge_high = (low - 0.5, low + 0.5) if width == 0 else (low, stats["high"])
    fig.add_trace(
        go.Scatter(
            x=[edge_low, edge_high],
            y=[edge_low, edge_high],
            mode="lines",
            name="y = x",
            line={"color": _style.NEUTRAL, "dash": "dash"},
            hoverinfo="skip",
        )
    )
    return fig
