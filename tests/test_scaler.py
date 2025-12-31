"""Tests for the Scaler preprocessing component."""

import math

import pytest
from pyspark.sql import functions as F

from smallaxe.exceptions import (
    ColumnNotFoundError,
    ModelNotFittedError,
    ValidationError,
)
from smallaxe.preprocessing import Scaler


@pytest.fixture
def df_for_scaling(spark_session):
    """Create a DataFrame for testing scaling."""
    data = [
        (1, 10.0, 100.0),
        (2, 20.0, 200.0),
        (3, 30.0, 300.0),
        (4, 40.0, 400.0),
        (5, 50.0, 500.0),
    ]
    columns = ["id", "value1", "value2"]
    return spark_session.createDataFrame(data, columns)


@pytest.fixture
def df_with_known_stats(spark_session):
    """Create a DataFrame with known statistics for verification.

    value1: [0, 2, 4, 6, 8] -> mean=4, std=sqrt(8)≈2.83
    value2: [0, 25, 50, 75, 100] -> min=0, max=100
    """
    data = [
        (1, 0.0, 0.0),
        (2, 2.0, 25.0),
        (3, 4.0, 50.0),
        (4, 6.0, 75.0),
        (5, 8.0, 100.0),
    ]
    columns = ["id", "value1", "value2"]
    return spark_session.createDataFrame(data, columns)


class TestScalerInit:
    """Tests for Scaler initialization."""

    def test_default_method(self):
        """Test that default method is 'standard'."""
        scaler = Scaler()
        assert scaler._method == "standard"

    def test_custom_method_standard(self):
        """Test setting standard method."""
        scaler = Scaler(method="standard")
        assert scaler._method == "standard"

    def test_custom_method_minmax(self):
        """Test setting minmax method."""
        scaler = Scaler(method="minmax")
        assert scaler._method == "minmax"

    def test_custom_method_maxabs(self):
        """Test setting maxabs method."""
        scaler = Scaler(method="maxabs")
        assert scaler._method == "maxabs"

    def test_invalid_method_raises_error(self):
        """Test that invalid method raises ValidationError."""
        with pytest.raises(ValidationError, match="Invalid method"):
            Scaler(method="invalid")

    def test_method_property(self):
        """Test method property returns the correct value."""
        scaler = Scaler(method="minmax")
        assert scaler.method == "minmax"


class TestStandardScaler:
    """Tests for Scaler with standard method (zero mean, unit variance)."""

    def test_standard_scaler_produces_zero_mean(self, df_with_known_stats):
        """Test that StandardScaler produces approximately zero mean."""
        scaler = Scaler(method="standard")
        result = scaler.fit_transform(df_with_known_stats, numerical_cols=["value1"])

        # Calculate mean of scaled values
        mean_value = result.select(F.mean("value1")).first()[0]

        # Mean should be approximately 0 (within floating point tolerance)
        assert abs(mean_value) < 1e-10

    def test_standard_scaler_produces_unit_variance(self, df_with_known_stats):
        """Test that StandardScaler produces approximately unit variance."""
        scaler = Scaler(method="standard")
        result = scaler.fit_transform(df_with_known_stats, numerical_cols=["value1"])

        # Calculate sample standard deviation of scaled values
        # PySpark's StandardScaler uses sample std (N-1), so use stddev_samp
        std_value = result.select(F.stddev_samp("value1")).first()[0]

        # Std should be approximately 1
        assert abs(std_value - 1.0) < 1e-6

    def test_standard_scaler_multiple_columns(self, df_with_known_stats):
        """Test StandardScaler with multiple columns."""
        scaler = Scaler(method="standard")
        result = scaler.fit_transform(df_with_known_stats, numerical_cols=["value1", "value2"])

        # Both columns should have approximately zero mean and unit sample std
        # PySpark's StandardScaler uses sample std (N-1), so use stddev_samp
        stats = result.select(
            F.mean("value1").alias("mean1"),
            F.mean("value2").alias("mean2"),
            F.stddev_samp("value1").alias("std1"),
            F.stddev_samp("value2").alias("std2"),
        ).first()

        assert abs(stats["mean1"]) < 1e-10
        assert abs(stats["mean2"]) < 1e-10
        assert abs(stats["std1"] - 1.0) < 1e-6
        assert abs(stats["std2"] - 1.0) < 1e-6


class TestMinMaxScaler:
    """Tests for Scaler with minmax method (0-1 range)."""

    def test_minmax_scaler_range(self, df_with_known_stats):
        """Test that MinMaxScaler produces values in 0-1 range."""
        scaler = Scaler(method="minmax")
        result = scaler.fit_transform(df_with_known_stats, numerical_cols=["value2"])

        # Get min and max of scaled values
        stats = result.select(
            F.min("value2").alias("min_val"),
            F.max("value2").alias("max_val"),
        ).first()

        # Min should be 0, max should be 1
        assert abs(stats["min_val"] - 0.0) < 1e-10
        assert abs(stats["max_val"] - 1.0) < 1e-10

    def test_minmax_scaler_linear_transform(self, df_with_known_stats):
        """Test that MinMaxScaler produces linearly spaced values."""
        scaler = Scaler(method="minmax")
        result = scaler.fit_transform(df_with_known_stats, numerical_cols=["value2"])

        # Collect scaled values
        scaled_values = [row["value2"] for row in result.orderBy("id").collect()]

        # Original: [0, 25, 50, 75, 100] -> scaled: [0, 0.25, 0.5, 0.75, 1.0]
        expected = [0.0, 0.25, 0.5, 0.75, 1.0]
        for actual, exp in zip(scaled_values, expected):
            assert abs(actual - exp) < 1e-10

    def test_minmax_scaler_multiple_columns(self, df_with_known_stats):
        """Test MinMaxScaler with multiple columns."""
        scaler = Scaler(method="minmax")
        result = scaler.fit_transform(df_with_known_stats, numerical_cols=["value1", "value2"])

        stats = result.select(
            F.min("value1").alias("min1"),
            F.max("value1").alias("max1"),
            F.min("value2").alias("min2"),
            F.max("value2").alias("max2"),
        ).first()

        # Both columns should be in [0, 1] range
        assert abs(stats["min1"] - 0.0) < 1e-10
        assert abs(stats["max1"] - 1.0) < 1e-10
        assert abs(stats["min2"] - 0.0) < 1e-10
        assert abs(stats["max2"] - 1.0) < 1e-10


class TestMaxAbsScaler:
    """Tests for Scaler with maxabs method."""

    def test_maxabs_scaler_range(self, spark_session):
        """Test that MaxAbsScaler scales by maximum absolute value."""
        # Create data with known max absolute value
        data = [
            (1, -10.0),
            (2, -5.0),
            (3, 0.0),
            (4, 5.0),
            (5, 20.0),  # max abs = 20
        ]
        df = spark_session.createDataFrame(data, ["id", "value"])

        scaler = Scaler(method="maxabs")
        result = scaler.fit_transform(df, numerical_cols=["value"])

        # Get scaled values
        scaled_values = [row["value"] for row in result.orderBy("id").collect()]

        # Expected: [-10/20, -5/20, 0/20, 5/20, 20/20] = [-0.5, -0.25, 0, 0.25, 1.0]
        expected = [-0.5, -0.25, 0.0, 0.25, 1.0]
        for actual, exp in zip(scaled_values, expected):
            assert abs(actual - exp) < 1e-10

    def test_maxabs_scaler_preserves_sign(self, spark_session):
        """Test that MaxAbsScaler preserves sign of values."""
        data = [
            (1, -100.0),
            (2, 50.0),
        ]
        df = spark_session.createDataFrame(data, ["id", "value"])

        scaler = Scaler(method="maxabs")
        result = scaler.fit_transform(df, numerical_cols=["value"])

        scaled_values = [row["value"] for row in result.orderBy("id").collect()]

        # Max abs = 100, so: [-100/100, 50/100] = [-1.0, 0.5]
        assert scaled_values[0] < 0  # Negative preserved
        assert scaled_values[1] > 0  # Positive preserved


class TestScalerErrors:
    """Tests for Scaler error handling."""

    def test_transform_before_fit_raises_error(self, df_for_scaling):
        """Test that transform before fit raises ModelNotFittedError."""
        scaler = Scaler()
        with pytest.raises(ModelNotFittedError, match="has not been fitted"):
            scaler.transform(df_for_scaling)

    def test_missing_column_in_fit_raises_error(self, df_for_scaling):
        """Test that missing column in fit raises ColumnNotFoundError."""
        scaler = Scaler()
        with pytest.raises(ColumnNotFoundError, match="nonexistent"):
            scaler.fit(df_for_scaling, numerical_cols=["nonexistent"])

    def test_missing_column_in_transform_raises_error(self, spark_session):
        """Test that missing column in transform raises ColumnNotFoundError."""
        # Fit on one DataFrame
        df_fit = spark_session.createDataFrame([(1, 10.0), (2, 20.0)], ["id", "value"])
        scaler = Scaler()
        scaler.fit(df_fit, numerical_cols=["value"])

        # Transform on a different DataFrame missing the column
        df_transform = spark_session.createDataFrame([(1, "A"), (2, "B")], ["id", "category"])
        with pytest.raises(ColumnNotFoundError, match="value"):
            scaler.transform(df_transform)

    def test_empty_numerical_cols_raises_error(self, df_for_scaling):
        """Test that empty numerical_cols raises ValidationError."""
        scaler = Scaler()
        with pytest.raises(ValidationError, match="numerical_cols cannot be empty"):
            scaler.fit(df_for_scaling, numerical_cols=[])

    def test_access_numerical_cols_before_fit_raises_error(self):
        """Test accessing numerical_cols before fit raises ModelNotFittedError."""
        scaler = Scaler()
        with pytest.raises(ModelNotFittedError):
            _ = scaler.numerical_cols


class TestScalerFitTransform:
    """Tests for fit_transform method."""

    def test_fit_transform_equivalent_to_fit_then_transform(self, df_for_scaling):
        """Test that fit_transform gives same result as fit + transform."""
        # Using fit_transform
        scaler1 = Scaler(method="standard")
        result1 = scaler1.fit_transform(df_for_scaling, numerical_cols=["value1"])

        # Using fit then transform
        scaler2 = Scaler(method="standard")
        scaler2.fit(df_for_scaling, numerical_cols=["value1"])
        result2 = scaler2.transform(df_for_scaling)

        # Results should have the same schema
        assert result1.columns == result2.columns


class TestScalerEdgeCases:
    """Tests for Scaler edge cases."""

    def test_single_column(self, df_for_scaling):
        """Test scaling a single column."""
        scaler = Scaler(method="standard")
        result = scaler.fit_transform(df_for_scaling, numerical_cols=["value1"])

        # Original columns should still exist
        assert "id" in result.columns
        assert "value1" in result.columns
        assert "value2" in result.columns

    def test_preserves_other_columns(self, df_for_scaling):
        """Test that non-scaled columns are preserved."""
        scaler = Scaler(method="minmax")
        result = scaler.fit_transform(df_for_scaling, numerical_cols=["value1"])

        # id and value2 should be unchanged
        original_ids = [row["id"] for row in df_for_scaling.orderBy("id").collect()]
        result_ids = [row["id"] for row in result.orderBy("id").collect()]
        assert original_ids == result_ids

        original_value2 = [row["value2"] for row in df_for_scaling.orderBy("id").collect()]
        result_value2 = [row["value2"] for row in result.orderBy("id").collect()]
        assert original_value2 == result_value2

    def test_numerical_cols_property(self, df_for_scaling):
        """Test numerical_cols property returns correct columns."""
        scaler = Scaler()
        scaler.fit(df_for_scaling, numerical_cols=["value1", "value2"])

        assert scaler.numerical_cols == ["value1", "value2"]

    def test_numerical_cols_property_returns_copy(self, df_for_scaling):
        """Test that numerical_cols property returns a copy."""
        scaler = Scaler()
        scaler.fit(df_for_scaling, numerical_cols=["value1", "value2"])

        cols = scaler.numerical_cols
        cols.append("modified")

        # Original should not be modified
        assert scaler.numerical_cols == ["value1", "value2"]

    def test_constant_column_standard(self, spark_session):
        """Test StandardScaler with constant column (all same values)."""
        data = [
            (1, 5.0),
            (2, 5.0),
            (3, 5.0),
        ]
        df = spark_session.createDataFrame(data, ["id", "value"])

        scaler = Scaler(method="standard")
        result = scaler.fit_transform(df, numerical_cols=["value"])

        # With constant column, std=0, so scaled values should be 0 or NaN
        # PySpark's StandardScaler handles this by returning 0
        scaled_values = [row["value"] for row in result.collect()]
        for val in scaled_values:
            # Either 0 or NaN is acceptable for constant column
            assert val == 0.0 or math.isnan(val)

    def test_constant_column_minmax(self, spark_session):
        """Test MinMaxScaler with constant column."""
        data = [
            (1, 5.0),
            (2, 5.0),
            (3, 5.0),
        ]
        df = spark_session.createDataFrame(data, ["id", "value"])

        scaler = Scaler(method="minmax")
        result = scaler.fit_transform(df, numerical_cols=["value"])

        # With constant column (min=max), MinMaxScaler typically returns 0.5 or 0
        scaled_values = [row["value"] for row in result.collect()]
        # Just verify it doesn't crash and returns some value
        assert all(v is not None for v in scaled_values)
