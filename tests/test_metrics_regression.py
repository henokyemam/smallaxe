"""Tests for regression metrics."""

import math

import pytest
from pyspark.sql import functions as F

from smallaxe.exceptions import ColumnNotFoundError
from smallaxe.metrics.regression import mae, mape, mse, r2, rmse


@pytest.fixture
def regression_df(spark_session):
    """Create a DataFrame with known values for regression metric testing."""
    # Simple data where we can calculate metrics by hand
    # label: [1, 2, 3, 4, 5]
    # predict: [1.1, 2.1, 2.9, 4.2, 4.8]
    # errors: [0.1, 0.1, 0.1, 0.2, 0.2]
    data = [
        (1, 1.0, 1.1),
        (2, 2.0, 2.1),
        (3, 3.0, 2.9),
        (4, 4.0, 4.2),
        (5, 5.0, 4.8),
    ]
    columns = ["id", "label", "predict_label"]
    return spark_session.createDataFrame(data, columns)


@pytest.fixture
def perfect_predictions_df(spark_session):
    """Create a DataFrame with perfect predictions."""
    data = [
        (1, 10.0, 10.0),
        (2, 20.0, 20.0),
        (3, 30.0, 30.0),
        (4, 40.0, 40.0),
        (5, 50.0, 50.0),
    ]
    columns = ["id", "label", "predict_label"]
    return spark_session.createDataFrame(data, columns)


@pytest.fixture
def poor_predictions_df(spark_session):
    """Create a DataFrame with poor predictions (using mean as prediction)."""
    # label: [10, 20, 30, 40, 50] - mean = 30
    # predict: [30, 30, 30, 30, 30] - predicting mean for all
    data = [
        (1, 10.0, 30.0),
        (2, 20.0, 30.0),
        (3, 30.0, 30.0),
        (4, 40.0, 30.0),
        (5, 50.0, 30.0),
    ]
    columns = ["id", "label", "predict_label"]
    return spark_session.createDataFrame(data, columns)


@pytest.fixture
def df_with_zeros(spark_session):
    """Create a DataFrame with zero values for MAPE testing."""
    data = [
        (1, 0.0, 0.5),
        (2, 10.0, 10.0),
        (3, 20.0, 22.0),
        (4, 0.0, 1.0),
        (5, 30.0, 27.0),
    ]
    columns = ["id", "label", "predict_label"]
    return spark_session.createDataFrame(data, columns)


@pytest.fixture
def all_zeros_df(spark_session):
    """Create a DataFrame where all true values are zero."""
    data = [
        (1, 0.0, 1.0),
        (2, 0.0, 2.0),
        (3, 0.0, 3.0),
    ]
    columns = ["id", "label", "predict_label"]
    return spark_session.createDataFrame(data, columns)


@pytest.fixture
def constant_labels_df(spark_session):
    """Create a DataFrame where all true labels are the same."""
    data = [
        (1, 5.0, 5.0),
        (2, 5.0, 5.0),
        (3, 5.0, 5.0),
    ]
    columns = ["id", "label", "predict_label"]
    return spark_session.createDataFrame(data, columns)


@pytest.fixture
def custom_columns_df(spark_session):
    """Create a DataFrame with custom column names."""
    data = [
        (1, 1.0, 1.5),
        (2, 2.0, 2.5),
        (3, 3.0, 3.5),
    ]
    columns = ["id", "actual", "predicted"]
    return spark_session.createDataFrame(data, columns)


class TestMSE:
    """Tests for Mean Squared Error."""

    def test_mse_known_values(self, regression_df):
        """Test MSE with known values.

        errors: [0.1, 0.1, 0.1, 0.2, 0.2]
        squared_errors: [0.01, 0.01, 0.01, 0.04, 0.04]
        MSE = (0.01 + 0.01 + 0.01 + 0.04 + 0.04) / 5 = 0.11 / 5 = 0.022
        """
        result = mse(regression_df)
        assert math.isclose(result, 0.022, rel_tol=1e-6)

    def test_mse_perfect_predictions(self, perfect_predictions_df):
        """Test MSE is 0 for perfect predictions."""
        result = mse(perfect_predictions_df)
        assert result == 0.0

    def test_mse_missing_column_raises_error(self, regression_df):
        """Test that missing column raises ColumnNotFoundError."""
        with pytest.raises(ColumnNotFoundError, match="nonexistent"):
            mse(regression_df, label_col="nonexistent")

    def test_mse_custom_columns(self, custom_columns_df):
        """Test MSE with custom column names."""
        # errors: [0.5, 0.5, 0.5]
        # MSE = (0.25 + 0.25 + 0.25) / 3 = 0.25
        result = mse(custom_columns_df, label_col="actual", prediction_col="predicted")
        assert math.isclose(result, 0.25, rel_tol=1e-6)


class TestRMSE:
    """Tests for Root Mean Squared Error."""

    def test_rmse_known_values(self, regression_df):
        """Test RMSE with known values.

        MSE = 0.022
        RMSE = sqrt(0.022) ≈ 0.1483
        """
        result = rmse(regression_df)
        expected = math.sqrt(0.022)
        assert math.isclose(result, expected, rel_tol=1e-6)

    def test_rmse_perfect_predictions(self, perfect_predictions_df):
        """Test RMSE is 0 for perfect predictions."""
        result = rmse(perfect_predictions_df)
        assert result == 0.0

    def test_rmse_missing_column_raises_error(self, regression_df):
        """Test that missing column raises ColumnNotFoundError."""
        with pytest.raises(ColumnNotFoundError, match="missing_col"):
            rmse(regression_df, prediction_col="missing_col")

    def test_rmse_custom_columns(self, custom_columns_df):
        """Test RMSE with custom column names."""
        # MSE = 0.25, RMSE = 0.5
        result = rmse(custom_columns_df, label_col="actual", prediction_col="predicted")
        assert math.isclose(result, 0.5, rel_tol=1e-6)


class TestMAE:
    """Tests for Mean Absolute Error."""

    def test_mae_known_values(self, regression_df):
        """Test MAE with known values.

        absolute_errors: [0.1, 0.1, 0.1, 0.2, 0.2]
        MAE = (0.1 + 0.1 + 0.1 + 0.2 + 0.2) / 5 = 0.7 / 5 = 0.14
        """
        result = mae(regression_df)
        assert math.isclose(result, 0.14, rel_tol=1e-6)

    def test_mae_perfect_predictions(self, perfect_predictions_df):
        """Test MAE is 0 for perfect predictions."""
        result = mae(perfect_predictions_df)
        assert result == 0.0

    def test_mae_missing_column_raises_error(self, regression_df):
        """Test that missing column raises ColumnNotFoundError."""
        with pytest.raises(ColumnNotFoundError, match="bad_col"):
            mae(regression_df, label_col="bad_col")

    def test_mae_custom_columns(self, custom_columns_df):
        """Test MAE with custom column names."""
        # absolute_errors: [0.5, 0.5, 0.5]
        # MAE = 1.5 / 3 = 0.5
        result = mae(custom_columns_df, label_col="actual", prediction_col="predicted")
        assert math.isclose(result, 0.5, rel_tol=1e-6)


class TestR2:
    """Tests for R-squared (Coefficient of Determination)."""

    def test_r2_perfect_predictions(self, perfect_predictions_df):
        """Test R2 is 1.0 for perfect predictions."""
        result = r2(perfect_predictions_df)
        assert math.isclose(result, 1.0, rel_tol=1e-6)

    def test_r2_poor_predictions(self, poor_predictions_df):
        """Test R2 is 0.0 when predicting mean for all values.

        When predictions equal the mean of true values, R2 = 0.
        """
        result = r2(poor_predictions_df)
        assert math.isclose(result, 0.0, rel_tol=1e-6)

    def test_r2_known_values(self, regression_df):
        """Test R2 with known values.

        labels: [1, 2, 3, 4, 5], mean = 3
        SS_tot = (1-3)^2 + (2-3)^2 + (3-3)^2 + (4-3)^2 + (5-3)^2 = 4 + 1 + 0 + 1 + 4 = 10

        predictions: [1.1, 2.1, 2.9, 4.2, 4.8]
        SS_res = (1-1.1)^2 + (2-2.1)^2 + (3-2.9)^2 + (4-4.2)^2 + (5-4.8)^2
               = 0.01 + 0.01 + 0.01 + 0.04 + 0.04 = 0.11

        R2 = 1 - (SS_res / SS_tot) = 1 - (0.11 / 10) = 1 - 0.011 = 0.989
        """
        result = r2(regression_df)
        assert math.isclose(result, 0.989, rel_tol=1e-6)

    def test_r2_missing_column_raises_error(self, regression_df):
        """Test that missing column raises ColumnNotFoundError."""
        with pytest.raises(ColumnNotFoundError):
            r2(regression_df, prediction_col="unknown")

    def test_r2_constant_labels(self, constant_labels_df):
        """Test R2 when all true labels are constant (SS_tot = 0)."""
        # When SS_tot = 0 and SS_res = 0, R2 = 1.0
        result = r2(constant_labels_df)
        assert result == 1.0

    def test_r2_negative_possible(self, spark_session):
        """Test that R2 can be negative for very poor predictions."""
        # predictions are further from true values than the mean
        data = [
            (1, 1.0, 10.0),  # mean of labels is 2.0, prediction is very far
            (2, 2.0, 10.0),
            (3, 3.0, 10.0),
        ]
        df = spark_session.createDataFrame(data, ["id", "label", "predict_label"])
        result = r2(df)
        assert result < 0.0


class TestMAPE:
    """Tests for Mean Absolute Percentage Error."""

    def test_mape_known_values(self, regression_df):
        """Test MAPE with known values.

        label: [1, 2, 3, 4, 5]
        predict: [1.1, 2.1, 2.9, 4.2, 4.8]

        APE: |1-1.1|/1 = 0.1, |2-2.1|/2 = 0.05, |3-2.9|/3 = 0.0333,
             |4-4.2|/4 = 0.05, |5-4.8|/5 = 0.04

        MAPE = 100 * (0.1 + 0.05 + 0.0333 + 0.05 + 0.04) / 5
             = 100 * 0.2733 / 5 = 5.467%
        """
        result = mape(regression_df)
        expected = 100 * (0.1 + 0.05 + 1 / 30 + 0.05 + 0.04) / 5
        assert math.isclose(result, expected, rel_tol=1e-4)

    def test_mape_perfect_predictions(self, perfect_predictions_df):
        """Test MAPE is 0 for perfect predictions."""
        result = mape(perfect_predictions_df)
        assert result == 0.0

    def test_mape_skips_zero_labels(self, df_with_zeros):
        """Test that MAPE skips rows where label is zero.

        Only non-zero labels: [10, 20, 30]
        predictions for these: [10, 22, 27]
        APE: |10-10|/10 = 0, |20-22|/20 = 0.1, |30-27|/30 = 0.1
        MAPE = 100 * (0 + 0.1 + 0.1) / 3 = 6.67%
        """
        result = mape(df_with_zeros)
        expected = 100 * (0 + 0.1 + 0.1) / 3
        assert math.isclose(result, expected, rel_tol=1e-4)

    def test_mape_all_zeros_returns_zero(self, all_zeros_df):
        """Test that MAPE returns 0 when all true values are zero."""
        result = mape(all_zeros_df)
        assert result == 0.0

    def test_mape_missing_column_raises_error(self, regression_df):
        """Test that missing column raises ColumnNotFoundError."""
        with pytest.raises(ColumnNotFoundError, match="not_here"):
            mape(regression_df, label_col="not_here")

    def test_mape_custom_columns(self, custom_columns_df):
        """Test MAPE with custom column names.

        actual: [1, 2, 3]
        predicted: [1.5, 2.5, 3.5]
        APE: |1-1.5|/1 = 0.5, |2-2.5|/2 = 0.25, |3-3.5|/3 = 0.167
        MAPE = 100 * (0.5 + 0.25 + 0.167) / 3 = 30.56%
        """
        result = mape(custom_columns_df, label_col="actual", prediction_col="predicted")
        expected = 100 * (0.5 + 0.25 + 0.5 / 3) / 3
        assert math.isclose(result, expected, rel_tol=1e-4)


class TestColumnValidation:
    """Tests for column validation across all metrics."""

    def test_all_metrics_validate_label_col(self, spark_session):
        """Test that all metrics validate label column."""
        df = spark_session.createDataFrame([(1, 1.0)], ["id", "predict_label"])

        for metric_fn in [mse, rmse, mae, r2, mape]:
            with pytest.raises(ColumnNotFoundError, match="label"):
                metric_fn(df)

    def test_all_metrics_validate_prediction_col(self, spark_session):
        """Test that all metrics validate prediction column."""
        df = spark_session.createDataFrame([(1, 1.0)], ["id", "label"])

        for metric_fn in [mse, rmse, mae, r2, mape]:
            with pytest.raises(ColumnNotFoundError, match="predict_label"):
                metric_fn(df)

    def test_error_includes_available_columns(self, spark_session):
        """Test that ColumnNotFoundError includes available columns."""
        df = spark_session.createDataFrame([(1, 1.0, 2.0)], ["id", "col_a", "col_b"])

        try:
            mse(df, label_col="missing")
        except ColumnNotFoundError as e:
            assert "id" in str(e) or e.available_columns is not None
            assert e.column == "missing"


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_row(self, spark_session):
        """Test metrics with a single row DataFrame."""
        df = spark_session.createDataFrame([(1, 10.0, 12.0)], ["id", "label", "predict_label"])

        assert mse(df) == 4.0
        assert rmse(df) == 2.0
        assert mae(df) == 2.0
        # SS_tot = 0 (single point), SS_res = 4 (error^2) -> R2 = 0.0
        assert r2(df) == 0.0
        assert mape(df) == 20.0  # |10-12|/10 * 100 = 20%

    def test_negative_values(self, spark_session):
        """Test metrics with negative values."""
        data = [
            (1, -10.0, -8.0),
            (2, -5.0, -6.0),
            (3, 0.0, 1.0),
            (4, 5.0, 4.0),
            (5, 10.0, 11.0),
        ]
        df = spark_session.createDataFrame(data, ["id", "label", "predict_label"])

        # Should work without errors
        assert mse(df) > 0
        assert rmse(df) > 0
        assert mae(df) > 0
        # R2 should be valid
        r2_value = r2(df)
        assert r2_value <= 1.0

    def test_large_values(self, spark_session):
        """Test metrics with large values."""
        data = [
            (1, 1e9, 1.1e9),
            (2, 2e9, 2.1e9),
            (3, 3e9, 2.9e9),
        ]
        df = spark_session.createDataFrame(data, ["id", "label", "predict_label"])

        # Should work without overflow
        assert mse(df) > 0
        assert rmse(df) > 0
        assert mae(df) > 0
        assert r2(df) < 1.0
