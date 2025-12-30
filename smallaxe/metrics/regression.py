"""Regression metrics for evaluating model predictions."""

from typing import List

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from smallaxe.exceptions import ColumnNotFoundError


def _validate_columns(df: DataFrame, label_col: str, prediction_col: str) -> None:
    """Validate that required columns exist in the DataFrame.

    Args:
        df: PySpark DataFrame.
        label_col: Name of the column containing true labels.
        prediction_col: Name of the column containing predictions.

    Raises:
        ColumnNotFoundError: If any required column is missing.
    """
    available_columns: List[str] = df.columns
    for col in [label_col, prediction_col]:
        if col not in available_columns:
            raise ColumnNotFoundError(column=col, available_columns=available_columns)


# def mse(df: DataFrame, label_col: str = "label", prediction_col: str = "predict_label") -> float:
#     """Compute Mean Squared Error.

#     MSE = (1/n) * sum((y_true - y_pred)^2)

#     Args:
#         df: PySpark DataFrame containing true and predicted values.
#         label_col: Name of the column containing true labels. Default is 'label'.
#         prediction_col: Name of the column containing predictions. Default is 'predict_label'.

#     Returns:
#         Mean Squared Error as a float.

#     Raises:
#         ColumnNotFoundError: If label_col or prediction_col is not in the DataFrame.
#     """
#     _validate_columns(df, label_col, prediction_col)

#     result = df.select(
#         F.avg(F.pow(F.col(label_col) - F.col(prediction_col), 2)).alias("mse")
#     ).first()

#     return float(result["mse"]) if result["mse"] is not None else 0.0


def mse(df: DataFrame, label_col: str = "label", prediction_col: str = "predict_label") -> float:
    """Compute Mean Squared Error using Spark's RegressionEvaluator."""
    _validate_columns(df, label_col, prediction_col)

    # 1. Handle Empty DataFrame (consistent with your other metric functions)
    if df.limit(1).count() == 0:
        return 0.0

    # 2. Setup the Regression Evaluator
    evaluator = RegressionEvaluator(
        labelCol=label_col, predictionCol=prediction_col, metricName="mse"
    )

    # 3. Calculate MSE
    # Spark returns a float; if the DF is empty (caught above) or invalid,
    try:
        result = evaluator.evaluate(df)
        return float(result)
    except Exception:
        return 0.0


# def rmse(df: DataFrame, label_col: str = "label", prediction_col: str = "predict_label") -> float:
#     """Compute Root Mean Squared Error.

#     RMSE = sqrt(MSE) = sqrt((1/n) * sum((y_true - y_pred)^2))

#     Args:
#         df: PySpark DataFrame containing true and predicted values.
#         label_col: Name of the column containing true labels. Default is 'label'.
#         prediction_col: Name of the column containing predictions. Default is 'predict_label'.

#     Returns:
#         Root Mean Squared Error as a float.

#     Raises:
#         ColumnNotFoundError: If label_col or prediction_col is not in the DataFrame.
#     """
#     _validate_columns(df, label_col, prediction_col)

#     result = df.select(
#         F.sqrt(F.avg(F.pow(F.col(label_col) - F.col(prediction_col), 2))).alias("rmse")
#     ).first()

#     return float(result["rmse"]) if result["rmse"] is not None else 0.0


def rmse(df: DataFrame, label_col: str = "label", prediction_col: str = "predict_label") -> float:
    """Compute Root Mean Squared Error using Spark's RegressionEvaluator."""
    _validate_columns(df, label_col, prediction_col)

    # 1. Handle Empty DataFrame
    # Returns 0.0 to match the behavior of your previous metric functions
    if df.limit(1).count() == 0:
        return 0.0

    # 2. Setup the Regression Evaluator for RMSE
    evaluator = RegressionEvaluator(
        labelCol=label_col, predictionCol=prediction_col, metricName="rmse"
    )

    # 3. Calculate RMSE
    try:
        result = evaluator.evaluate(df)
        return float(result)
    except Exception:
        # Fallback for unexpected calculation errors or data issues
        return 0.0


# def mae(df: DataFrame, label_col: str = "label", prediction_col: str = "predict_label") -> float:
#     """Compute Mean Absolute Error.

#     MAE = (1/n) * sum(|y_true - y_pred|)

#     Args:
#         df: PySpark DataFrame containing true and predicted values.
#         label_col: Name of the column containing true labels. Default is 'label'.
#         prediction_col: Name of the column containing predictions. Default is 'predict_label'.

#     Returns:
#         Mean Absolute Error as a float.

#     Raises:
#         ColumnNotFoundError: If label_col or prediction_col is not in the DataFrame.
#     """
#     _validate_columns(df, label_col, prediction_col)

#     result = df.select(F.avg(F.abs(F.col(label_col) - F.col(prediction_col))).alias("mae")).first()

#     return float(result["mae"]) if result["mae"] is not None else 0.0


def mae(df: DataFrame, label_col: str = "label", prediction_col: str = "predict_label") -> float:
    """Compute Mean Absolute Error using Spark's RegressionEvaluator."""
    _validate_columns(df, label_col, prediction_col)

    # 1. Handle Empty DataFrame
    # Returns 0.0 to remain consistent with your previous logic
    if df.limit(1).count() == 0:
        return 0.0

    # 2. Setup the Regression Evaluator for MAE
    evaluator = RegressionEvaluator(
        labelCol=label_col, predictionCol=prediction_col, metricName="mae"
    )

    # 3. Calculate MAE
    try:
        result = evaluator.evaluate(df)
        return float(result)
    except Exception:
        # Fallback for data issues or unexpected calculation errors
        return 0.0


# def r2(df: DataFrame, label_col: str = "label", prediction_col: str = "predict_label") -> float:
#     """Compute R-squared (Coefficient of Determination).

#     R2 = 1 - (SS_res / SS_tot)
#     where:
#         SS_res = sum((y_true - y_pred)^2)  (residual sum of squares)
#         SS_tot = sum((y_true - y_mean)^2)  (total sum of squares)

#     Args:
#         df: PySpark DataFrame containing true and predicted values.
#         label_col: Name of the column containing true labels. Default is 'label'.
#         prediction_col: Name of the column containing predictions. Default is 'predict_label'.

#     Returns:
#         R-squared score as a float. Can be negative for very poor predictions.

#     Raises:
#         ColumnNotFoundError: If label_col or prediction_col is not in the DataFrame.
#     """
#     _validate_columns(df, label_col, prediction_col)

#     # Compute mean of true labels
#     mean_label = df.select(F.avg(F.col(label_col))).first()[0]

#     if mean_label is None:
#         return 0.0

#     # Compute SS_res and SS_tot
#     result = df.select(
#         F.sum(F.pow(F.col(label_col) - F.col(prediction_col), 2)).alias("ss_res"),
#         F.sum(F.pow(F.col(label_col) - F.lit(mean_label), 2)).alias("ss_tot"),
#     ).first()

#     ss_res = result["ss_res"]
#     ss_tot = result["ss_tot"]

#     if ss_res is None or ss_tot is None:
#         return 0.0

#     # Handle case where SS_tot is zero (all true values are the same)
#     if ss_tot == 0:
#         return 1.0 if ss_res == 0 else 0.0

#     return float(1 - (ss_res / ss_tot))


def r2(df: DataFrame, label_col: str = "label", prediction_col: str = "predict_label") -> float:
    """Compute R-squared (Coefficient of Determination) using Spark's RegressionEvaluator."""
    _validate_columns(df, label_col, prediction_col)

    # 1. Handle Empty DataFrame
    if df.limit(1).count() == 0:
        return 0.0

    # 2. Handle Constant Labels (SS_tot = 0)
    # If the min and max of the label column are the same, the labels are constant.
    stats = df.select(
        F.min(label_col).alias("min_label"),
        F.max(label_col).alias("max_label"),
        F.count_distinct(F.when(F.col(label_col) == F.col(prediction_col), 1)).alias(
            "perfect_match"
        ),
    ).collect()[0]

    if stats["min_label"] == stats["max_label"]:
        # Check if predictions also match that constant value
        # We check if there are any rows where label != prediction
        mismatch_count = df.filter(F.col(label_col) != F.col(prediction_col)).count()
        return 1.0 if mismatch_count == 0 else 0.0

    # 3. Setup the Regression Evaluator for R2
    evaluator = RegressionEvaluator(
        labelCol=label_col, predictionCol=prediction_col, metricName="r2"
    )

    # 4. Calculate R2
    try:
        result = evaluator.evaluate(df)

        # Spark's R2 evaluator returns NaN or -inf in edge cases
        import math

        if math.isnan(result) or result == float("-inf"):
            return 0.0

        return float(result)
    except Exception:
        return 0.0


def mape(df: DataFrame, label_col: str = "label", prediction_col: str = "predict_label") -> float:
    """Compute Mean Absolute Percentage Error.

    MAPE = (100/n) * sum(|y_true - y_pred| / |y_true|)

    Note: Rows where label_col is zero are excluded from the calculation
    to avoid division by zero.

    Args:
        df: PySpark DataFrame containing true and predicted values.
        label_col: Name of the column containing true labels. Default is 'label'.
        prediction_col: Name of the column containing predictions. Default is 'predict_label'.

    Returns:
        Mean Absolute Percentage Error as a float (in percentage, e.g., 10.5 for 10.5%).
        Returns 0.0 if all true values are zero.

    Raises:
        ColumnNotFoundError: If label_col or prediction_col is not in the DataFrame.
    """
    _validate_columns(df, label_col, prediction_col)

    # Filter out rows where the true label is zero to avoid division by zero
    df_filtered = df.filter(F.col(label_col) != 0)

    # If all values are zero, return 0.0
    if df_filtered.count() == 0:
        return 0.0

    result = df_filtered.select(
        F.avg(F.abs(F.col(label_col) - F.col(prediction_col)) / F.abs(F.col(label_col))).alias(
            "mape"
        )
    ).first()

    mape_value = result["mape"]

    if mape_value is None:
        return 0.0

    # Return as percentage
    return float(mape_value * 100)
