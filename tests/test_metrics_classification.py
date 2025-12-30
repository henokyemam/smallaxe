"""Tests for classification metrics."""

import math

import pytest

from smallaxe.exceptions import ColumnNotFoundError
from smallaxe.metrics.classification import (
    accuracy,
    auc_pr,
    auc_roc,
    f1_score,
    log_loss,
    precision,
    recall,
)


@pytest.fixture
def binary_classification_df(spark_session):
    """Create a DataFrame with known values for binary classification testing.

    Confusion matrix:
    - TP = 3 (predicted 1, actual 1)
    - TN = 2 (predicted 0, actual 0)
    - FP = 1 (predicted 1, actual 0)
    - FN = 2 (predicted 0, actual 1)

    Total = 8
    Accuracy = (3 + 2) / 8 = 5/8 = 0.625
    Precision = 3 / (3 + 1) = 3/4 = 0.75
    Recall = 3 / (3 + 2) = 3/5 = 0.6
    F1 = 2 * 0.75 * 0.6 / (0.75 + 0.6) = 0.9 / 1.35 = 0.6667
    """
    data = [
        (1, 1, 1),  # TP
        (2, 1, 1),  # TP
        (3, 1, 1),  # TP
        (4, 0, 0),  # TN
        (5, 0, 0),  # TN
        (6, 0, 1),  # FP
        (7, 1, 0),  # FN
        (8, 1, 0),  # FN
    ]
    columns = ["id", "label", "predict_label"]
    return spark_session.createDataFrame(data, columns)


@pytest.fixture
def perfect_classification_df(spark_session):
    """Create a DataFrame with perfect predictions."""
    data = [
        (1, 0, 0),
        (2, 0, 0),
        (3, 1, 1),
        (4, 1, 1),
        (5, 1, 1),
    ]
    columns = ["id", "label", "predict_label"]
    return spark_session.createDataFrame(data, columns)


@pytest.fixture
def worst_classification_df(spark_session):
    """Create a DataFrame with worst predictions (all incorrect)."""
    data = [
        (1, 0, 1),
        (2, 0, 1),
        (3, 1, 0),
        (4, 1, 0),
    ]
    columns = ["id", "label", "predict_label"]
    return spark_session.createDataFrame(data, columns)


@pytest.fixture
def probability_df(spark_session):
    """Create a DataFrame with probability scores for AUC/log_loss testing.

    Labels: [1, 1, 1, 0, 0, 0]
    Probs:  [0.9, 0.8, 0.6, 0.4, 0.3, 0.1]

    This is well-separated, so AUC should be high.
    """
    data = [
        (1, 1, 0.9),
        (2, 1, 0.8),
        (3, 1, 0.6),
        (4, 0, 0.4),
        (5, 0, 0.3),
        (6, 0, 0.1),
    ]
    columns = ["id", "label", "probability"]
    return spark_session.createDataFrame(data, columns)


@pytest.fixture
def perfect_probability_df(spark_session):
    """Create a DataFrame with perfect probability predictions."""
    data = [
        (1, 1, 1.0),
        (2, 1, 1.0),
        (3, 0, 0.0),
        (4, 0, 0.0),
    ]
    columns = ["id", "label", "probability"]
    return spark_session.createDataFrame(data, columns)


@pytest.fixture
def random_probability_df(spark_session):
    """Create a DataFrame with random (uninformative) probability predictions.

    All probabilities are 0.5.
    """
    data = [
        (1, 1, 0.5),
        (2, 1, 0.5),
        (3, 0, 0.5),
        (4, 0, 0.5),
    ]
    columns = ["id", "label", "probability"]
    return spark_session.createDataFrame(data, columns)


@pytest.fixture
def custom_columns_df(spark_session):
    """Create a DataFrame with custom column names."""
    data = [
        (1, 1, 1, 0.9),
        (2, 0, 0, 0.2),
        (3, 1, 1, 0.8),
        (4, 0, 0, 0.3),
    ]
    columns = ["id", "actual", "predicted", "score"]
    return spark_session.createDataFrame(data, columns)


class TestAccuracy:
    """Tests for accuracy metric."""

    def test_accuracy_known_values(self, binary_classification_df):
        """Test accuracy with known values: 5/8 = 0.625."""
        result = accuracy(binary_classification_df)
        assert math.isclose(result, 0.625, rel_tol=1e-6)

    def test_accuracy_perfect_predictions(self, perfect_classification_df):
        """Test accuracy is 1.0 for perfect predictions."""
        result = accuracy(perfect_classification_df)
        assert result == 1.0

    def test_accuracy_worst_predictions(self, worst_classification_df):
        """Test accuracy is 0.0 for completely wrong predictions."""
        result = accuracy(worst_classification_df)
        assert result == 0.0

    def test_accuracy_missing_column_raises_error(self, binary_classification_df):
        """Test that missing column raises ColumnNotFoundError."""
        with pytest.raises(ColumnNotFoundError, match="nonexistent"):
            accuracy(binary_classification_df, label_col="nonexistent")

    def test_accuracy_custom_columns(self, custom_columns_df):
        """Test accuracy with custom column names."""
        result = accuracy(custom_columns_df, label_col="actual", prediction_col="predicted")
        assert result == 1.0


class TestPrecision:
    """Tests for precision metric."""

    def test_precision_known_values(self, binary_classification_df):
        """Test precision with known values: 3/4 = 0.75."""
        result = precision(binary_classification_df)
        assert math.isclose(result, 0.75, rel_tol=1e-6)

    def test_precision_perfect_predictions(self, perfect_classification_df):
        """Test precision is 1.0 for perfect predictions."""
        result = precision(perfect_classification_df)
        assert result == 1.0

    def test_precision_all_false_positives(self, worst_classification_df):
        """Test precision is 0.0 when all positive predictions are wrong."""
        result = precision(worst_classification_df)
        assert result == 0.0

    def test_precision_no_positive_predictions(self, spark_session):
        """Test precision returns 0.0 when there are no positive predictions."""
        data = [(1, 1, 0), (2, 0, 0)]
        df = spark_session.createDataFrame(data, ["id", "label", "predict_label"])
        result = precision(df)
        assert result == 0.0

    def test_precision_missing_column_raises_error(self, binary_classification_df):
        """Test that missing column raises ColumnNotFoundError."""
        with pytest.raises(ColumnNotFoundError, match="missing"):
            precision(binary_classification_df, prediction_col="missing")

    def test_precision_custom_columns(self, custom_columns_df):
        """Test precision with custom column names."""
        result = precision(custom_columns_df, label_col="actual", prediction_col="predicted")
        assert result == 1.0


class TestRecall:
    """Tests for recall metric."""

    def test_recall_known_values(self, binary_classification_df):
        """Test recall with known values: 3/5 = 0.6."""
        result = recall(binary_classification_df)
        assert math.isclose(result, 0.6, rel_tol=1e-6)

    def test_recall_perfect_predictions(self, perfect_classification_df):
        """Test recall is 1.0 for perfect predictions."""
        result = recall(perfect_classification_df)
        assert result == 1.0

    def test_recall_all_false_negatives(self, worst_classification_df):
        """Test recall is 0.0 when all positives are missed."""
        result = recall(worst_classification_df)
        assert result == 0.0

    def test_recall_no_actual_positives(self, spark_session):
        """Test recall returns 0.0 when there are no actual positives."""
        data = [(1, 0, 1), (2, 0, 0)]
        df = spark_session.createDataFrame(data, ["id", "label", "predict_label"])
        result = recall(df)
        assert result == 0.0

    def test_recall_missing_column_raises_error(self, binary_classification_df):
        """Test that missing column raises ColumnNotFoundError."""
        with pytest.raises(ColumnNotFoundError, match="wrong"):
            recall(binary_classification_df, label_col="wrong")

    def test_recall_custom_columns(self, custom_columns_df):
        """Test recall with custom column names."""
        result = recall(custom_columns_df, label_col="actual", prediction_col="predicted")
        assert result == 1.0


class TestF1Score:
    """Tests for F1 score metric."""

    def test_f1_score_known_values(self, binary_classification_df):
        """Test F1 score with known values.

        Precision = 0.75, Recall = 0.6
        F1 = 2 * 0.75 * 0.6 / (0.75 + 0.6) = 0.9 / 1.35 = 0.6667
        """
        result = f1_score(binary_classification_df)
        expected = 2 * 0.75 * 0.6 / (0.75 + 0.6)
        assert math.isclose(result, expected, rel_tol=1e-4)

    def test_f1_score_perfect_predictions(self, perfect_classification_df):
        """Test F1 score is 1.0 for perfect predictions."""
        result = f1_score(perfect_classification_df)
        assert result == 1.0

    def test_f1_score_worst_predictions(self, worst_classification_df):
        """Test F1 score is 0.0 for completely wrong predictions."""
        result = f1_score(worst_classification_df)
        assert result == 0.0

    def test_f1_score_zero_precision_and_recall(self, spark_session):
        """Test F1 score returns 0.0 when both precision and recall are 0."""
        # No positive predictions and no actual positives
        data = [(1, 0, 0), (2, 0, 0)]
        df = spark_session.createDataFrame(data, ["id", "label", "predict_label"])
        result = f1_score(df)
        assert result == 0.0

    def test_f1_score_missing_column_raises_error(self, binary_classification_df):
        """Test that missing column raises ColumnNotFoundError."""
        with pytest.raises(ColumnNotFoundError):
            f1_score(binary_classification_df, label_col="bad")

    def test_f1_score_custom_columns(self, custom_columns_df):
        """Test F1 score with custom column names."""
        result = f1_score(custom_columns_df, label_col="actual", prediction_col="predicted")
        assert result == 1.0


class TestAucRoc:
    """Tests for AUC-ROC metric."""

    def test_auc_roc_well_separated(self, probability_df):
        """Test AUC-ROC with well-separated predictions.

        With probabilities [0.9, 0.8, 0.6] for positives and [0.4, 0.3, 0.1] for negatives,
        the AUC should be 1.0 (perfect separation).
        """
        result = auc_roc(probability_df)
        assert math.isclose(result, 1.0, rel_tol=1e-6)

    def test_auc_roc_perfect_predictions(self, perfect_probability_df):
        """Test AUC-ROC is 1.0 for perfect probability predictions."""
        result = auc_roc(perfect_probability_df)
        assert result == 1.0

    def test_auc_roc_random_predictions(self, random_probability_df):
        """Test AUC-ROC is approximately 0.5 for random predictions."""
        result = auc_roc(random_probability_df)
        # With all same probabilities, AUC should be 0.5 (ties result in 0.5)
        assert math.isclose(result, 0.5, rel_tol=0.1)

    def test_auc_roc_inverted_predictions(self, spark_session):
        """Test AUC-ROC is 0.0 for completely inverted predictions."""
        data = [
            (1, 1, 0.1),
            (2, 1, 0.2),
            (3, 0, 0.8),
            (4, 0, 0.9),
        ]
        df = spark_session.createDataFrame(data, ["id", "label", "probability"])
        result = auc_roc(df)
        # Inverted predictions result in AUC = 0
        assert math.isclose(result, 0.0, rel_tol=1e-6)

    def test_auc_roc_missing_column_raises_error(self, probability_df):
        """Test that missing column raises ColumnNotFoundError."""
        with pytest.raises(ColumnNotFoundError, match="missing_prob"):
            auc_roc(probability_df, probability_col="missing_prob")

    def test_auc_roc_no_positives(self, spark_session):
        """Test AUC-ROC returns 0.0 when there are no positive labels."""
        data = [(1, 0, 0.9), (2, 0, 0.1)]
        df = spark_session.createDataFrame(data, ["id", "label", "probability"])
        result = auc_roc(df)
        assert result == 0.0

    def test_auc_roc_no_negatives(self, spark_session):
        """Test AUC-ROC returns 0.0 when there are no negative labels."""
        data = [(1, 1, 0.9), (2, 1, 0.1)]
        df = spark_session.createDataFrame(data, ["id", "label", "probability"])
        result = auc_roc(df)
        assert result == 0.0

    def test_auc_roc_custom_columns(self, custom_columns_df):
        """Test AUC-ROC with custom column names."""
        result = auc_roc(custom_columns_df, label_col="actual", probability_col="score")
        assert result == 1.0


class TestAucPr:
    """Tests for AUC-PR metric."""

    def test_auc_pr_well_separated(self, probability_df):
        """Test AUC-PR with well-separated predictions."""
        result = auc_pr(probability_df)
        assert result > 0.9  # Should be high for well-separated data

    def test_auc_pr_perfect_predictions(self, perfect_probability_df):
        """Test AUC-PR is 1.0 for perfect probability predictions."""
        result = auc_pr(perfect_probability_df)
        assert math.isclose(result, 1.0, rel_tol=1e-6)

    def test_auc_pr_missing_column_raises_error(self, probability_df):
        """Test that missing column raises ColumnNotFoundError."""
        with pytest.raises(ColumnNotFoundError, match="bad_col"):
            auc_pr(probability_df, label_col="bad_col")

    def test_auc_pr_no_positives(self, spark_session):
        """Test AUC-PR returns 0.0 when there are no positive labels."""
        data = [(1, 0, 0.9), (2, 0, 0.1)]
        df = spark_session.createDataFrame(data, ["id", "label", "probability"])
        result = auc_pr(df)
        assert result == 0.0

    def test_auc_pr_custom_columns(self, custom_columns_df):
        """Test AUC-PR with custom column names."""
        result = auc_pr(custom_columns_df, label_col="actual", probability_col="score")
        assert result == 1.0


class TestLogLoss:
    """Tests for log loss metric."""

    def test_log_loss_known_values(self, probability_df):
        """Test log loss with known values.

        For well-calibrated predictions, log loss should be relatively low.
        """
        result = log_loss(probability_df)
        assert result > 0  # Log loss is always positive
        assert result < 1  # Should be low for good predictions

    def test_log_loss_perfect_predictions(self, perfect_probability_df):
        """Test log loss is close to 0 for perfect predictions.

        Note: We use eps to avoid log(0), so result won't be exactly 0.
        """
        result = log_loss(perfect_probability_df)
        assert result < 0.01  # Should be very small

    def test_log_loss_worst_predictions(self, spark_session):
        """Test log loss is high for completely wrong predictions."""
        data = [
            (1, 1, 0.0),  # Should be 1, predicted 0
            (2, 0, 1.0),  # Should be 0, predicted 1
        ]
        df = spark_session.createDataFrame(data, ["id", "label", "probability"])
        result = log_loss(df)
        # With clipping to avoid log(0), the loss will be very high
        assert result > 30  # Should be very high (approaching infinity)

    def test_log_loss_random_predictions(self, random_probability_df):
        """Test log loss for random (0.5) predictions.

        Log loss for p=0.5 is -log(0.5) = log(2) ≈ 0.693
        """
        result = log_loss(random_probability_df)
        expected = math.log(2)  # ≈ 0.693
        assert math.isclose(result, expected, rel_tol=1e-4)

    def test_log_loss_missing_column_raises_error(self, probability_df):
        """Test that missing column raises ColumnNotFoundError."""
        with pytest.raises(ColumnNotFoundError, match="missing"):
            log_loss(probability_df, probability_col="missing")

    def test_log_loss_custom_columns(self, custom_columns_df):
        """Test log loss with custom column names."""
        result = log_loss(custom_columns_df, label_col="actual", probability_col="score")
        assert result > 0
        assert result < 1


class TestColumnValidation:
    """Tests for column validation across all classification metrics."""

    def test_all_label_metrics_validate_label_col(self, spark_session):
        """Test that all label-based metrics validate label column."""
        df = spark_session.createDataFrame([(1, 1)], ["id", "predict_label"])

        for metric_fn in [accuracy, precision, recall, f1_score]:
            with pytest.raises(ColumnNotFoundError, match="label"):
                metric_fn(df)

    def test_all_label_metrics_validate_prediction_col(self, spark_session):
        """Test that all label-based metrics validate prediction column."""
        df = spark_session.createDataFrame([(1, 1)], ["id", "label"])

        for metric_fn in [accuracy, precision, recall, f1_score]:
            with pytest.raises(ColumnNotFoundError, match="predict_label"):
                metric_fn(df)

    def test_all_probability_metrics_validate_columns(self, spark_session):
        """Test that all probability-based metrics validate columns."""
        df = spark_session.createDataFrame([(1, 0.5)], ["id", "probability"])

        for metric_fn in [auc_roc, auc_pr, log_loss]:
            with pytest.raises(ColumnNotFoundError, match="label"):
                metric_fn(df)

        df2 = spark_session.createDataFrame([(1, 1)], ["id", "label"])
        for metric_fn in [auc_roc, auc_pr, log_loss]:
            with pytest.raises(ColumnNotFoundError, match="probability"):
                metric_fn(df2)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_row(self, spark_session):
        """Test metrics with a single row DataFrame."""
        df = spark_session.createDataFrame([(1, 1, 1)], ["id", "label", "predict_label"])

        assert accuracy(df) == 1.0
        assert precision(df) == 1.0
        assert recall(df) == 1.0
        assert f1_score(df) == 1.0

    def test_all_positive_labels(self, spark_session):
        """Test metrics when all labels are positive."""
        data = [(1, 1, 1), (2, 1, 1), (3, 1, 0)]
        df = spark_session.createDataFrame(data, ["id", "label", "predict_label"])

        assert math.isclose(accuracy(df), 2 / 3, rel_tol=1e-6)
        assert precision(df) == 1.0  # All positive predictions are correct
        assert math.isclose(recall(df), 2 / 3, rel_tol=1e-6)  # 2 out of 3 positives found

    def test_all_negative_labels(self, spark_session):
        """Test metrics when all labels are negative."""
        data = [(1, 0, 0), (2, 0, 1), (3, 0, 0)]
        df = spark_session.createDataFrame(data, ["id", "label", "predict_label"])

        assert math.isclose(accuracy(df), 2 / 3, rel_tol=1e-6)
        assert precision(df) == 0.0  # The one positive prediction is wrong
        assert recall(df) == 0.0  # No true positives to find

    def test_empty_dataframe(self, spark_session):
        """Test metrics with empty DataFrame."""
        df = spark_session.createDataFrame([], "id: int, label: int, predict_label: int")

        assert accuracy(df) == 0.0
        assert precision(df) == 0.0
        assert recall(df) == 0.0
        assert f1_score(df) == 0.0

    def test_empty_probability_df(self, spark_session):
        """Test probability metrics with empty DataFrame."""
        df = spark_session.createDataFrame([], "id: int, label: int, probability: double")

        assert auc_roc(df) == 0.0
        assert auc_pr(df) == 0.0
        assert log_loss(df) == 0.0

    def test_probability_edge_values(self, spark_session):
        """Test log loss with probability values at 0 and 1."""
        data = [
            (1, 1, 1.0),
            (2, 0, 0.0),
        ]
        df = spark_session.createDataFrame(data, ["id", "label", "probability"])

        # Should not raise an error due to clipping
        result = log_loss(df)
        assert result < 0.01  # Should be very close to 0 (perfect predictions)

    def test_multiclass_labels_as_binary(self, spark_session):
        """Test that metrics work when labels are integers (not just 0/1)."""
        # Using 2 and 3 as labels should still work if we only check equality
        data = [(1, 2, 2), (2, 3, 3), (3, 2, 3)]
        df = spark_session.createDataFrame(data, ["id", "label", "predict_label"])

        assert math.isclose(accuracy(df), 2 / 3, rel_tol=1e-6)
