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


# def auc_roc(df, label_col="label", probability_col="probability"):
#     _validate_columns(df, label_col, probability_col)

#     data = df.select(
#         F.col(label_col).alias("label"), F.col(probability_col).alias("prob")
#     ).collect()

#     if not data:
#         return 0.0

#     # Sort data by probability
#     data.sort(key=lambda x: x["prob"])

#     n = len(data)
#     positives = sum(1 for row in data if row["label"] == 1)
#     negatives = n - positives

#     if positives == 0 or negatives == 0:
#         return 0.0

#     rank_sum = 0.0
#     i = 0
#     while i < n:
#         # Find a block of tied probabilities
#         start_idx = i
#         while i + 1 < n and data[i+1]["prob"] == data[i]["prob"]:
#             i += 1
#         end_idx = i

#         # Calculate the average rank for this block
#         # Ranks are 1-based, so average of [start+1, ..., end+1]
#         avg_rank = (start_idx + 1 + end_idx + 1) / 2.0

#         # Count how many positives are in this tied block
#         num_pos_in_block = sum(1 for j in range(start_idx, end_idx + 1) if data[j]["label"] == 1)

#         rank_sum += num_pos_in_block * avg_rank
#         i += 1

#     # Wilcoxon-Mann-Whitney formula
#     auc = (rank_sum - (positives * (positives + 1) / 2.0)) / (positives * negatives)
#     return float(auc)
# def auc_roc(df: DataFrame, label_col: str = "label", probability_col: str = "probability") -> float:
#     """Compute Area Under the ROC Curve (AUC-ROC).

#     Uses the trapezoidal rule to calculate the area under the ROC curve.
#     The ROC curve plots True Positive Rate vs False Positive Rate at various thresholds.

#     Args:
#         df: PySpark DataFrame containing true labels and probability scores.
#         label_col: Name of the column containing true labels (0 or 1). Default is 'label'.
#         probability_col: Name of the column containing probability scores. Default is 'probability'.

#     Returns:
#         AUC-ROC as a float between 0 and 1.
#         Returns 0.5 for random predictions.

#     Raises:
#         ColumnNotFoundError: If label_col or probability_col is not in the DataFrame.
#     """
#     _validate_columns(df, label_col, probability_col)

#     # Collect data for AUC calculation
#     data = df.select(
#         F.col(label_col).alias("label"), F.col(probability_col).alias("prob")
#     ).collect()

#     if len(data) == 0:
#         return 0.0

#     # Count positives and negatives
#     positives = sum(1 for row in data if row["label"] == 1)
#     negatives = len(data) - positives

#     if positives == 0 or negatives == 0:
#         return 0.0

#     # Calculate AUC using the Wilcoxon-Mann-Whitney statistic
#     # AUC = (sum of ranks of positives - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
#     rank_sum = 0
#     for i, row in enumerate(sorted(data, key=lambda x: x["prob"])):
#         if row["label"] == 1:
#             rank_sum += i + 1  # 1-based rank

#     # Handle ties by using average rank
#     # For simplicity, we'll use the Wilcoxon-Mann-Whitney approach
#     auc = (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)

#     return float(auc)


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


# def auc_pr(df: DataFrame, label_col: str = "label", probability_col: str = "probability") -> float:
#     """Compute Area Under the Precision-Recall Curve (AUC-PR).

#     Uses the trapezoidal rule to calculate the area under the PR curve.
#     The PR curve plots Precision vs Recall at various thresholds.

#     Args:
#         df: PySpark DataFrame containing true labels and probability scores.
#         label_col: Name of the column containing true labels (0 or 1). Default is 'label'.
#         probability_col: Name of the column containing probability scores. Default is 'probability'.

#     Returns:
#         AUC-PR as a float between 0 and 1.

#     Raises:
#         ColumnNotFoundError: If label_col or probability_col is not in the DataFrame.
#     """
#     _validate_columns(df, label_col, probability_col)

#     # Collect data for AUC-PR calculation
#     data = df.select(
#         F.col(label_col).alias("label"), F.col(probability_col).alias("prob")
#     ).collect()

#     if len(data) == 0:
#         return 0.0

#     total_positives = sum(1 for row in data if row["label"] == 1)

#     if total_positives == 0:
#         return 0.0

#     # Sort by probability descending
#     sorted_data = sorted(data, key=lambda x: x["prob"], reverse=True)

#     # Calculate precision and recall at each threshold
#     precisions = []
#     recalls = []

#     true_positives = 0
#     for i, row in enumerate(sorted_data):
#         if row["label"] == 1:
#             true_positives += 1
#         predicted_positives = i + 1
#         prec = true_positives / predicted_positives
#         rec = true_positives / total_positives

#         precisions.append(prec)
#         recalls.append(rec)

#     # Add starting point (recall=0)
#     recalls = [0.0] + recalls
#     precisions = [1.0] + precisions  # precision starts at 1 when recall is 0

#     # Calculate AUC using trapezoidal rule
#     auc = 0.0
#     for i in range(1, len(recalls)):
#         # Area of trapezoid
#         auc += (recalls[i] - recalls[i - 1]) * (precisions[i] + precisions[i - 1]) / 2

#     return float(auc)


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
