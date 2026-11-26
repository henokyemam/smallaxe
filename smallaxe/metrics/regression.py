"""Regression metrics for evaluating model predictions.

Every metric raises ``ValidationError`` on an empty DataFrame or a non-numeric label or
prediction column; an argument Spark's evaluator rejects is also a ``ValidationError``.
A metric that is mathematically undefined on the rows given is
``float("nan")``, never 0.0, which would read as a perfect error score.
"""

from typing import Literal

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from smallaxe import _checks
from smallaxe._sql import quoted
from smallaxe.metrics._checks import exact_integers, require_numeric, require_rows, widened
from smallaxe.metrics._checks import validate_columns as _validate_columns

__all__ = ["mse", "rmse", "mae", "r2", "mape"]


def _evaluate(
    dataframe: DataFrame,
    label_col: str,
    prediction_col: str,
    metric: Literal["mse", "rmse", "mae", "r2"],
) -> float:
    evaluator = RegressionEvaluator(
        labelCol=label_col, predictionCol=prediction_col, metricName=metric
    )
    with _checks.spark_argument_errors(f"Cannot compute {metric}"):
        return float(evaluator.evaluate(dataframe))


def mse(
    dataframe: DataFrame, label_col: str = "label", prediction_col: str = "predict_label"
) -> float:
    """Compute Mean Squared Error.

    MSE = (1/n) * sum((y_true - y_pred)^2), computed by Spark's ``RegressionEvaluator``.

    Args:
        dataframe: PySpark DataFrame containing true and predicted values.
        label_col: Name of the column containing true labels. Default is 'label'.
        prediction_col: Name of the column containing predictions. Default is 'predict_label'.

    Returns:
        Mean Squared Error as a float. NaN if a label or prediction is NaN.

    Raises:
        ColumnNotFoundError: If label_col or prediction_col is not in the DataFrame.
        ValidationError: If label_col or prediction_col is not numeric, or the DataFrame
            has no rows.
    """
    _validate_columns(dataframe, label_col, prediction_col)
    require_numeric(dataframe, "mse", label_col, prediction_col)
    require_rows(dataframe, "mse")
    return _evaluate(dataframe, label_col, prediction_col, "mse")


def rmse(
    dataframe: DataFrame, label_col: str = "label", prediction_col: str = "predict_label"
) -> float:
    """Compute Root Mean Squared Error.

    RMSE = sqrt((1/n) * sum((y_true - y_pred)^2)), computed by Spark's
    ``RegressionEvaluator``.

    Args:
        dataframe: PySpark DataFrame containing true and predicted values.
        label_col: Name of the column containing true labels. Default is 'label'.
        prediction_col: Name of the column containing predictions. Default is 'predict_label'.

    Returns:
        Root Mean Squared Error as a float. NaN if a label or prediction is NaN.

    Raises:
        ColumnNotFoundError: If label_col or prediction_col is not in the DataFrame.
        ValidationError: If label_col or prediction_col is not numeric, or the DataFrame
            has no rows.
    """
    _validate_columns(dataframe, label_col, prediction_col)
    require_numeric(dataframe, "rmse", label_col, prediction_col)
    require_rows(dataframe, "rmse")
    return _evaluate(dataframe, label_col, prediction_col, "rmse")


def mae(
    dataframe: DataFrame, label_col: str = "label", prediction_col: str = "predict_label"
) -> float:
    """Compute Mean Absolute Error.

    MAE = (1/n) * sum(|y_true - y_pred|), computed by Spark's ``RegressionEvaluator``.

    Args:
        dataframe: PySpark DataFrame containing true and predicted values.
        label_col: Name of the column containing true labels. Default is 'label'.
        prediction_col: Name of the column containing predictions. Default is 'predict_label'.

    Returns:
        Mean Absolute Error as a float. NaN if a label or prediction is NaN.

    Raises:
        ColumnNotFoundError: If label_col or prediction_col is not in the DataFrame.
        ValidationError: If label_col or prediction_col is not numeric, or the DataFrame
            has no rows.
    """
    _validate_columns(dataframe, label_col, prediction_col)
    require_numeric(dataframe, "mae", label_col, prediction_col)
    require_rows(dataframe, "mae")
    return _evaluate(dataframe, label_col, prediction_col, "mae")


def r2(
    dataframe: DataFrame, label_col: str = "label", prediction_col: str = "predict_label"
) -> float:
    """Compute R-squared (Coefficient of Determination).

    R2 = 1 - SS_res / SS_tot, where SS_res = sum((y_true - y_pred)^2) and
    SS_tot = sum((y_true - mean(y_true))^2), computed by Spark's ``RegressionEvaluator``.

    When every label is the same, SS_tot is zero and the ratio is undefined; as in
    scikit-learn, the score is then 1.0 if every prediction equals the label and 0.0
    otherwise.

    Args:
        dataframe: PySpark DataFrame containing true and predicted values.
        label_col: Name of the column containing true labels. Default is 'label'.
        prediction_col: Name of the column containing predictions. Default is 'predict_label'.

    Returns:
        R-squared as a float; negative for predictions worse than the mean. NaN when
        the evaluator yields NaN (a NaN label or prediction) or no row has a label.

    Raises:
        ColumnNotFoundError: If label_col or prediction_col is not in the DataFrame.
        ValidationError: If label_col or prediction_col is not numeric, or the DataFrame
            has no rows.
    """
    _validate_columns(dataframe, label_col, prediction_col)
    require_numeric(dataframe, "r2", label_col, prediction_col)
    require_rows(dataframe, "r2")

    label = F.col(label_col)
    stats = dataframe.select(
        F.min(label).alias("min_label"),
        F.max(label).alias("max_label"),
        F.sum((label != F.col(prediction_col)).cast("long")).alias("mismatches"),
    ).first()
    assert stats is not None  # an aggregate without GROUP BY always yields one row

    if stats["min_label"] is None:
        return float("nan")
    if stats["min_label"] == stats["max_label"]:
        return 1.0 if not stats["mismatches"] else 0.0

    return _evaluate(dataframe, label_col, prediction_col, "r2")


def mape(
    dataframe: DataFrame, label_col: str = "label", prediction_col: str = "predict_label"
) -> float:
    """Compute Mean Absolute Percentage Error.

    MAPE = (100/n) * sum(|y_true - y_pred| / |y_true|)

    Rows where label_col is zero (or null) are excluded from the calculation
    to avoid division by zero.

    Args:
        dataframe: PySpark DataFrame containing true and predicted values.
        label_col: Name of the column containing true labels. Default is 'label'.
        prediction_col: Name of the column containing predictions. Default is 'predict_label'.

    Returns:
        Mean Absolute Percentage Error as a float (in percentage, e.g., 10.5 for 10.5%).
        NaN if every label is zero, since no row can be scored.

    Raises:
        ColumnNotFoundError: If label_col or prediction_col is not in the DataFrame.
        ValidationError: If label_col or prediction_col is not numeric, or the
            DataFrame has no rows.
    """
    _validate_columns(dataframe, label_col, prediction_col)
    require_numeric(dataframe, "mape", label_col, prediction_col)
    require_rows(dataframe, "mape")

    # Spark 3's default arithmetic, made independent of ANSI mode. Integer columns as
    # exact decimals: ``label - prediction`` and ``abs(label)`` cannot overflow (ANSI
    # mode raises; without it the value wraps). Any other columns through try_subtract /
    # try_divide, which give Spark 3's default result (null on a decimal overflow) where
    # ANSI mode's operators raise. Zero labels are filtered out first, so the division
    # never sees a zero denominator.
    integers = exact_integers(dataframe, label_col, prediction_col)
    if integers is not None:
        label, prediction = integers
        error = F.abs(label - prediction).cast("double") / F.abs(label).cast("double")
    else:
        # An integer beside a float column is subtracted as a float, as Spark 3's
        # default mode does (widened); the denominator keeps the label's own type.
        label_sql, prediction_sql = widened(dataframe, label_col, prediction_col)
        error = F.expr(
            f"try_divide(abs(try_subtract({label_sql}, {prediction_sql})), "
            f"abs({quoted(label_col)}))"
        )
    result = (
        dataframe.filter(F.col(quoted(label_col)) != 0).select(F.avg(error).alias("mape")).first()
    )
    assert result is not None  # an aggregate without GROUP BY always yields one row

    if result["mape"] is None:
        return float("nan")
    return float(result["mape"] * 100)
