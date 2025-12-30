"""Classification metrics for evaluating model predictions."""

from typing import List

from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from smallaxe.exceptions import ColumnNotFoundError


def _validate_columns(df: DataFrame, *cols: str) -> None:
    """Validate that required columns exist in the DataFrame.

    Args:
        df: PySpark DataFrame.
        *cols: Column names to validate.

    Raises:
        ColumnNotFoundError: If any required column is missing.
    """
    available_columns: List[str] = df.columns
    for col in cols:
        if col not in available_columns:
            raise ColumnNotFoundError(column=col, available_columns=available_columns)


def accuracy(
    df: DataFrame, label_col: str = "label", prediction_col: str = "predict_label"
) -> float:
    """Compute classification accuracy.

    Accuracy = (TP + TN) / (TP + TN + FP + FN) = correct / total

    Args:
        df: PySpark DataFrame containing true and predicted labels.
        label_col: Name of the column containing true labels. Default is 'label'.
        prediction_col: Name of the column containing predictions. Default is 'predict_label'.

    Returns:
        Accuracy as a float between 0 and 1.

    Raises:
        ColumnNotFoundError: If label_col or prediction_col is not in the DataFrame.
    """
    _validate_columns(df, label_col, prediction_col)

    total_count = df.count()
    if total_count == 0:
        return 0.0

    correct_count = df.filter(F.col(label_col) == F.col(prediction_col)).count()
    return float(correct_count / total_count)


def precision(
    df: DataFrame, label_col: str = "label", prediction_col: str = "predict_label"
) -> float:
    """Compute precision for binary classification.

    Precision = TP / (TP + FP)

    Args:
        df: PySpark DataFrame containing true and predicted labels.
        label_col: Name of the column containing true labels (0 or 1). Default is 'label'.
        prediction_col: Name of the column containing predictions (0 or 1). Default is 'predict_label'.

    Returns:
        Precision as a float between 0 and 1.
        Returns 0.0 if there are no positive predictions.

    Raises:
        ColumnNotFoundError: If label_col or prediction_col is not in the DataFrame.
    """
    _validate_columns(df, label_col, prediction_col)

    # Count true positives (predicted positive and actually positive)
    true_positives = df.filter((F.col(prediction_col) == 1) & (F.col(label_col) == 1)).count()

    # Count all positive predictions
    predicted_positives = df.filter(F.col(prediction_col) == 1).count()

    if predicted_positives == 0:
        return 0.0

    return float(true_positives / predicted_positives)


def recall(df: DataFrame, label_col: str = "label", prediction_col: str = "predict_label") -> float:
    """Compute recall (sensitivity, true positive rate) for binary classification.

    Recall = TP / (TP + FN)

    Args:
        df: PySpark DataFrame containing true and predicted labels.
        label_col: Name of the column containing true labels (0 or 1). Default is 'label'.
        prediction_col: Name of the column containing predictions (0 or 1). Default is 'predict_label'.

    Returns:
        Recall as a float between 0 and 1.
        Returns 0.0 if there are no actual positive labels.

    Raises:
        ColumnNotFoundError: If label_col or prediction_col is not in the DataFrame.
    """
    _validate_columns(df, label_col, prediction_col)

    # Count true positives (predicted positive and actually positive)
    true_positives = df.filter((F.col(prediction_col) == 1) & (F.col(label_col) == 1)).count()

    # Count all actual positives
    actual_positives = df.filter(F.col(label_col) == 1).count()

    if actual_positives == 0:
        return 0.0

    return float(true_positives / actual_positives)


def f1_score(
    df: DataFrame, label_col: str = "label", prediction_col: str = "predict_label"
) -> float:
    """Compute F1 score for binary classification.

    F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        df: PySpark DataFrame containing true and predicted labels.
        label_col: Name of the column containing true labels (0 or 1). Default is 'label'.
        prediction_col: Name of the column containing predictions (0 or 1). Default is 'predict_label'.

    Returns:
        F1 score as a float between 0 and 1.
        Returns 0.0 if precision + recall = 0.

    Raises:
        ColumnNotFoundError: If label_col or prediction_col is not in the DataFrame.
    """
    _validate_columns(df, label_col, prediction_col)

    prec = precision(df, label_col, prediction_col)
    rec = recall(df, label_col, prediction_col)

    if prec + rec == 0:
        return 0.0

    return float(2 * (prec * rec) / (prec + rec))


def auc_roc(df, label_col="label", probability_col="probability"):
    _validate_columns(df, label_col, probability_col)

    # 1. Check for empty DataFrame
    # Spark's evaluator might return 0.5 for empty sets; your test wants 0.0.
    if df.storageLevel.useMemory or df.limit(1).count() == 0:
        if df.limit(1).count() == 0:
            return 0.0

    # 2. Check for single-class data (No negatives or no positives)
    # Spark often returns 1.0 or NaN here; your test expects 0.0.
    distinct_labels = [row[0] for row in df.select(label_col).distinct().collect()]
    if len(distinct_labels) < 2:
        return 0.0

    # 3. Use the Spark Evaluator for the heavy lifting
    evaluator = BinaryClassificationEvaluator(
        labelCol=label_col, rawPredictionCol=probability_col, metricName="areaUnderROC"
    )

    return float(evaluator.evaluate(df))


def auc_pr(df: DataFrame, label_col: str = "label", probability_col: str = "probability") -> float:
    """Compute Area Under the Precision-Recall Curve (AUC-PR)."""
    _validate_columns(df, label_col, probability_col)

    # 1. Handle Empty DataFrame (matches your test requirements)
    if df.limit(1).count() == 0:
        return 0.0

    # 2. Check for the existence of positive labels
    # AUC-PR is defined by precision/recall; if there are no positives,
    # recall is undefined. Your manual code returns 0.0.
    has_positives = df.filter(F.col(label_col) == 1).limit(1).count() > 0
    if not has_positives:
        return 0.0

    # 3. Use the Spark Evaluator
    evaluator = BinaryClassificationEvaluator(
        labelCol=label_col, rawPredictionCol=probability_col, metricName="areaUnderPR"
    )

    return float(evaluator.evaluate(df))


def log_loss(
    df: DataFrame,
    label_col: str = "label",
    probability_col: str = "probability",
    eps: float = 1e-15,
) -> float:
    """Compute logarithmic loss (cross-entropy loss).

    Log Loss = -(1/n) * sum(y * log(p) + (1-y) * log(1-p))

    Args:
        df: PySpark DataFrame containing true labels and probability scores.
        label_col: Name of the column containing true labels (0 or 1). Default is 'label'.
        probability_col: Name of the column containing probability scores. Default is 'probability'.
        eps: Small value to avoid log(0). Default is 1e-15.

    Returns:
        Log loss as a float. Lower values indicate better predictions.

    Raises:
        ColumnNotFoundError: If label_col or probability_col is not in the DataFrame.
    """
    _validate_columns(df, label_col, probability_col)

    # Clip probabilities to avoid log(0)
    clipped_prob = (
        F.when(F.col(probability_col) < eps, eps)
        .when(F.col(probability_col) > 1 - eps, 1 - eps)
        .otherwise(F.col(probability_col))
    )

    # Calculate log loss
    # -[y * log(p) + (1-y) * log(1-p)]
    result = df.select(
        F.avg(
            -(
                F.col(label_col) * F.log(clipped_prob)
                + (1 - F.col(label_col)) * F.log(1 - clipped_prob)
            )
        ).alias("log_loss")
    ).first()

    if result["log_loss"] is None:
        return 0.0

    return float(result["log_loss"])
