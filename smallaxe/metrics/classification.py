"""Classification metrics for evaluating model predictions."""

from typing import Dict, List

from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from smallaxe.exceptions import ColumnNotFoundError, ValidationError

MULTICLASS_AVERAGED_METRICS = {
    f"{metric}_{average}"
    for metric in ("precision", "recall", "f1_score")
    for average in ("macro", "weighted")
}


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
    df: DataFrame,
    label_col: str = "label",
    prediction_col: str = "predict_label",
    *,
    average: str = "binary",
) -> float:
    """Compute precision with binary, macro, or support-weighted averaging.

    Precision = TP / (TP + FP)

    Args:
        df: PySpark DataFrame containing true and predicted labels.
        label_col: Name of the column containing true labels. Default is 'label'.
        prediction_col: Name of the column containing predictions. Default is 'predict_label'.
        average: 'binary' scores class 1 (default); 'macro' averages all classes
            equally; 'weighted' weights each class by its true-label count.
            Undefined per-class scores are zero.

    Returns:
        Precision as a float between 0 and 1.
        Undefined per-class precision is 0.0.

    Raises:
        ColumnNotFoundError: If label_col or prediction_col is not in the DataFrame.
        ValidationError: If average is not 'binary', 'macro', or 'weighted'.
    """
    if average != "binary":
        return _averaged_score(df, label_col, prediction_col, "precision", average)
    _validate_columns(df, label_col, prediction_col)

    # Count true positives (predicted positive and actually positive)
    true_positives = df.filter((F.col(prediction_col) == 1) & (F.col(label_col) == 1)).count()

    # Count all positive predictions
    predicted_positives = df.filter(F.col(prediction_col) == 1).count()

    if predicted_positives == 0:
        return 0.0

    return float(true_positives / predicted_positives)


def recall(
    df: DataFrame,
    label_col: str = "label",
    prediction_col: str = "predict_label",
    *,
    average: str = "binary",
) -> float:
    """Compute recall with binary, macro, or support-weighted averaging.

    Recall = TP / (TP + FN)

    Args:
        df: PySpark DataFrame containing true and predicted labels.
        label_col: Name of the column containing true labels. Default is 'label'.
        prediction_col: Name of the column containing predictions. Default is 'predict_label'.
        average: 'binary' scores class 1 (default); 'macro' averages all classes
            equally; 'weighted' weights each class by its true-label count.
            Undefined per-class scores are zero.

    Returns:
        Recall as a float between 0 and 1.
        Undefined per-class recall is 0.0.

    Raises:
        ColumnNotFoundError: If label_col or prediction_col is not in the DataFrame.
        ValidationError: If average is not 'binary', 'macro', or 'weighted'.
    """
    if average != "binary":
        return _averaged_score(df, label_col, prediction_col, "recall", average)
    _validate_columns(df, label_col, prediction_col)

    # Count true positives (predicted positive and actually positive)
    true_positives = df.filter((F.col(prediction_col) == 1) & (F.col(label_col) == 1)).count()

    # Count all actual positives
    actual_positives = df.filter(F.col(label_col) == 1).count()

    if actual_positives == 0:
        return 0.0

    return float(true_positives / actual_positives)


def f1_score(
    df: DataFrame,
    label_col: str = "label",
    prediction_col: str = "predict_label",
    *,
    average: str = "binary",
) -> float:
    """Compute F1 with binary, macro, or support-weighted averaging.

    F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        df: PySpark DataFrame containing true and predicted labels.
        label_col: Name of the column containing true labels. Default is 'label'.
        prediction_col: Name of the column containing predictions. Default is 'predict_label'.
        average: 'binary' scores class 1 (default); 'macro' averages per-class
            F1 equally; 'weighted' weights each class by its true-label count.
            Undefined per-class scores are zero.

    Returns:
        F1 score as a float between 0 and 1.
        Undefined per-class F1 is 0.0.

    Raises:
        ColumnNotFoundError: If label_col or prediction_col is not in the DataFrame.
        ValidationError: If average is not 'binary', 'macro', or 'weighted'.
    """
    if average != "binary":
        return _averaged_score(df, label_col, prediction_col, "f1_score", average)
    _validate_columns(df, label_col, prediction_col)

    prec = precision(df, label_col, prediction_col)
    rec = recall(df, label_col, prediction_col)

    if prec + rec == 0:
        return 0.0

    return float(2 * (prec * rec) / (prec + rec))


def _averaged_score(
    df: DataFrame, label_col: str, prediction_col: str, metric: str, average: str
) -> float:
    if average not in {"macro", "weighted"}:
        raise ValidationError("average must be 'binary', 'macro', or 'weighted'.")
    return multiclass_scores(df, label_col, prediction_col)[f"{metric}_{average}"]


def multiclass_scores(
    df: DataFrame, label_col: str = "label", prediction_col: str = "predict_label"
) -> Dict[str, float]:
    """Compute accuracy and macro/weighted precision, recall, and F1 together.

    Classes are the union of true and predicted labels. Macro scores give each
    class equal weight; weighted scores use true-label support. Undefined
    per-class scores and scores on an empty frame are zero. The unsuffixed
    ``precision``, ``recall``, and ``f1_score`` keys alias the macro scores.

    Counts and averages stay distributed in Spark; only the final scores are
    collected. F1 is averaged per class, rather than computed from averaged
    precision and recall.
    """
    _validate_columns(df, label_col, prediction_col)
    # Each input row contributes one actual and one predicted class count.
    # This also retains classes that appear only in predictions or true labels.
    counts = df.select(
        F.explode(
            F.array(
                F.struct(
                    F.col(label_col).alias("class"),
                    F.lit(1).alias("support"),
                    F.lit(0).alias("predicted"),
                    F.lit(0).alias("tp"),
                ),
                F.struct(
                    F.col(prediction_col).alias("class"),
                    F.lit(0).alias("support"),
                    F.lit(1).alias("predicted"),
                    (F.col(label_col) == F.col(prediction_col)).cast("int").alias("tp"),
                ),
            )
        ).alias("count")
    )
    per_class = (
        counts.select("count.*")
        .groupBy("class")
        .agg(
            F.sum("support").alias("support"),
            F.sum("predicted").alias("predicted"),
            F.sum("tp").alias("tp"),
        )
    )
    for name, numerator, denominator in (
        ("precision", F.col("tp"), F.col("predicted")),
        ("recall", F.col("tp"), F.col("support")),
        ("f1_score", 2 * F.col("tp"), F.col("support") + F.col("predicted")),
    ):
        per_class = per_class.withColumn(
            name, F.when(denominator > 0, numerator / denominator).otherwise(0.0)
        )
    total = F.sum("support")
    averages = [F.when(total > 0, F.sum("tp") / total).otherwise(0.0).alias("accuracy")]
    for name in ("precision", "recall", "f1_score"):
        averages.extend(
            [
                F.coalesce(F.avg(name), F.lit(0.0)).alias(f"{name}_macro"),
                F.when(total > 0, F.sum(F.col(name) * F.col("support")) / total)
                .otherwise(0.0)
                .alias(f"{name}_weighted"),
            ]
        )
    scores = {
        name: float(value) for name, value in per_class.agg(*averages).first().asDict().items()
    }
    for name in ("precision", "recall", "f1_score"):
        scores[name] = scores[f"{name}_macro"]
    return scores


def auc_roc(df: DataFrame, label_col: str = "label", probability_col: str = "probability") -> float:
    """Compute Area Under the ROC Curve (AUC-ROC).

    Args:
        df: PySpark DataFrame containing true labels and probability scores.
        label_col: Name of the column containing true labels (0 or 1). Default is 'label'.
        probability_col: Name of the column containing probability scores. Default is 'probability'.

    Returns:
        AUC-ROC as a float between 0 and 1.
        Returns 0.0 if the DataFrame is empty or contains only one class.

    Raises:
        ColumnNotFoundError: If label_col or probability_col is not in the DataFrame.
    """
    _validate_columns(df, label_col, probability_col)

    if df.limit(1).count() == 0:
        return 0.0

    distinct_labels = [row[0] for row in df.select(label_col).distinct().collect()]
    if len(distinct_labels) < 2:
        return 0.0

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
