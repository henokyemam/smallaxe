"""Classification metrics for evaluating model predictions.

Every metric raises ``ValidationError`` on an empty DataFrame and lets Spark's own
errors propagate. AUC-ROC, AUC-PR, and log loss are ``float("nan")`` when they are
mathematically undefined on the rows given, never 0.0. Per-class precision, recall,
and F1 with a zero denominator are 0.0, as in scikit-learn's ``zero_division=0``.
"""

from typing import Dict

from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import NumericType

from smallaxe.exceptions import ValidationError
from smallaxe.metrics._checks import require_numeric, require_rows
from smallaxe.metrics._checks import validate_columns as _validate_columns

MULTICLASS_AVERAGED_METRICS = {
    f"{metric}_{average}"
    for metric in ("precision", "recall", "f1_score")
    for average in ("macro", "weighted")
}


def accuracy(
    df: DataFrame, label_col: str = "label", prediction_col: str = "predict_label"
) -> float:
    """Compute classification accuracy.

    Accuracy = (TP + TN) / (TP + TN + FP + FN) = correct / total

    Every row counts in the total; a row with a null label or prediction is never
    correct, so a frame where every row has one scores 0.0. (``multiclass_scores``
    instead ignores such rows.)

    Args:
        df: PySpark DataFrame containing true and predicted labels.
        label_col: Name of the column containing true labels. Default is 'label'.
        prediction_col: Name of the column containing predictions. Default is 'predict_label'.

    Returns:
        Accuracy as a float between 0 and 1.

    Raises:
        ColumnNotFoundError: If label_col or prediction_col is not in the DataFrame.
        ValidationError: If the DataFrame has no rows.
    """
    _validate_columns(df, label_col, prediction_col)

    total_count = df.count()
    if total_count == 0:
        raise ValidationError("Cannot compute accuracy: the DataFrame has no rows.")

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
        ValidationError: If the DataFrame has no rows, if average is not 'binary',
            'macro', or 'weighted', or if average is 'macro' or 'weighted' and a
            column is not numeric.
    """
    if average != "binary":
        return _averaged_score(df, label_col, prediction_col, "precision", average)
    _validate_columns(df, label_col, prediction_col)
    require_rows(df, "precision")

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
        ValidationError: If the DataFrame has no rows, if average is not 'binary',
            'macro', or 'weighted', or if average is 'macro' or 'weighted' and a
            column is not numeric.
    """
    if average != "binary":
        return _averaged_score(df, label_col, prediction_col, "recall", average)
    _validate_columns(df, label_col, prediction_col)
    require_rows(df, "recall")

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
        ValidationError: If the DataFrame has no rows, if average is not 'binary',
            'macro', or 'weighted', or if average is 'macro' or 'weighted' and a
            column is not numeric.
    """
    if average != "binary":
        return _averaged_score(df, label_col, prediction_col, "f1_score", average)
    _validate_columns(df, label_col, prediction_col)
    require_rows(df, "f1_score")

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
    return _multiclass_scores(df, label_col, prediction_col, metric)[f"{metric}_{average}"]


def multiclass_scores(
    df: DataFrame, label_col: str = "label", prediction_col: str = "predict_label"
) -> Dict[str, float]:
    """Compute accuracy and macro/weighted precision, recall, and F1 together.

    Classes are the union of true and predicted labels, compared as numbers, so
    a label ``1`` and a prediction ``1.0`` are the same class. Rows with a null
    label or prediction are ignored; if every row has one, there is nothing to
    score and every value is NaN. Macro scores give each class equal weight;
    weighted scores use true-label support. Undefined per-class scores are zero.
    The unsuffixed ``precision``, ``recall``, and ``f1_score`` keys alias the
    macro scores.

    Counts and averages stay distributed in Spark; only the final scores are
    collected. F1 is averaged per class, rather than computed from averaged
    precision and recall.

    Args:
        df: PySpark DataFrame containing true and predicted labels.
        label_col: Name of the numeric column containing true labels. Default is 'label'.
        prediction_col: Name of the numeric column containing predictions.
            Default is 'predict_label'.

    Returns:
        Dict with ``accuracy``, ``precision``, ``recall``, ``f1_score``, and
        their ``*_macro`` and ``*_weighted`` variants, each a float in [0, 1].

    Raises:
        ColumnNotFoundError: If label_col or prediction_col is not in the DataFrame.
        ValidationError: If label_col or prediction_col is not numeric, or the
            DataFrame has no rows.
    """
    return _multiclass_scores(df, label_col, prediction_col, "multiclass_scores")


def _multiclass_scores(
    df: DataFrame, label_col: str, prediction_col: str, metric: str
) -> Dict[str, float]:
    """:func:`multiclass_scores`, naming ``metric`` in the empty-frame error."""
    _validate_columns(df, label_col, prediction_col)
    for col in (label_col, prediction_col):
        data_type = df.schema[col].dataType
        if not isinstance(data_type, NumericType):
            raise ValidationError(
                f"Column '{col}' must be numeric for multiclass scoring, "
                f"got {data_type.simpleString()}."
            )
    require_rows(df, metric)
    label = F.col(label_col).cast("double")
    prediction = F.col(prediction_col).cast("double")
    # Each input row contributes one actual and one predicted class count.
    # This also retains classes that appear only in predictions or true labels.
    counts = df.where(label.isNotNull() & prediction.isNotNull()).select(
        F.explode(
            F.array(
                F.struct(
                    label.alias("class"),
                    F.lit(1).alias("support"),
                    F.lit(0).alias("predicted"),
                    F.lit(0).alias("tp"),
                ),
                F.struct(
                    prediction.alias("class"),
                    F.lit(0).alias("support"),
                    F.lit(1).alias("predicted"),
                    (label == prediction).cast("int").alias("tp"),
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
    # ``total`` is zero (null) only when every row has a null label or prediction.
    total = F.sum("support")
    undefined = F.lit(float("nan"))
    averages = [F.when(total > 0, F.sum("tp") / total).otherwise(undefined).alias("accuracy")]
    for name in ("precision", "recall", "f1_score"):
        averages.extend(
            [
                F.coalesce(F.avg(name), undefined).alias(f"{name}_macro"),
                F.when(total > 0, F.sum(F.col(name) * F.col("support")) / total)
                .otherwise(undefined)
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
        AUC-ROC as a float between 0 and 1. NaN when only one class is present,
        since the ROC curve needs both positives and negatives.

    Raises:
        ColumnNotFoundError: If label_col or probability_col is not in the DataFrame.
        ValidationError: If the DataFrame has no rows.
    """
    _validate_columns(df, label_col, probability_col)
    require_rows(df, "auc_roc")

    label = F.col(label_col)
    n_classes = df.where(label.isNotNull()).select(label).distinct().limit(2).count()
    if n_classes < 2:
        return float("nan")

    evaluator = BinaryClassificationEvaluator(
        labelCol=label_col, rawPredictionCol=probability_col, metricName="areaUnderROC"
    )
    return float(evaluator.evaluate(df))


def auc_pr(df: DataFrame, label_col: str = "label", probability_col: str = "probability") -> float:
    """Compute Area Under the Precision-Recall Curve (AUC-PR).

    Args:
        df: PySpark DataFrame containing true labels and probability scores.
        label_col: Name of the column containing true labels (0 or 1). Default is 'label'.
        probability_col: Name of the column containing probability scores. Default is 'probability'.

    Returns:
        AUC-PR as a float between 0 and 1. NaN when no row has a positive (1) label,
        since recall is then undefined. A frame of positives only is defined.

    Raises:
        ColumnNotFoundError: If label_col or probability_col is not in the DataFrame.
        ValidationError: If the DataFrame has no rows.
    """
    _validate_columns(df, label_col, probability_col)
    require_rows(df, "auc_pr")

    if df.filter(F.col(label_col) == 1).limit(1).count() == 0:
        return float("nan")

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
        Log loss as a float. Lower values indicate better predictions. NaN when no
        row has both a label and a probability.

    Raises:
        ColumnNotFoundError: If label_col or probability_col is not in the DataFrame.
        ValidationError: If label_col or probability_col is not numeric, or the
            DataFrame has no rows.
    """
    _validate_columns(df, label_col, probability_col)
    require_numeric(df, "log_loss", label_col, probability_col)
    require_rows(df, "log_loss")

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
    assert result is not None  # an aggregate without GROUP BY always yields one row

    if result["log_loss"] is None:
        return float("nan")
    return float(result["log_loss"])
