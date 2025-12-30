"""Tests for smallaxe sample datasets."""

import pytest
from pyspark.sql.types import DoubleType, IntegerType, StringType

from smallaxe.datasets import (
    dataset_info,
    load_sample_classification,
    load_sample_regression,
)


class TestRegressionDataset:
    """Tests for the sample regression dataset."""

    def test_load_sample_regression_returns_dataframe(self, spark_session):
        """Test that load_sample_regression returns a DataFrame."""
        df = load_sample_regression(spark_session)
        assert df is not None
        assert hasattr(df, "count")

    def test_load_sample_regression_default_row_count(self, spark_session):
        """Test that default row count is 10,000."""
        df = load_sample_regression(spark_session)
        assert df.count() == 10000

    def test_load_sample_regression_custom_row_count(self, spark_session):
        """Test that custom row count works."""
        df = load_sample_regression(spark_session, n_rows=100)
        assert df.count() == 100

    def test_load_sample_regression_schema_columns(self, spark_session):
        """Test that dataset has correct column names."""
        df = load_sample_regression(spark_session, n_rows=10)
        expected_columns = [
            "bedrooms",
            "bathrooms",
            "sqft",
            "age",
            "location",
            "condition",
            "price",
        ]
        assert df.columns == expected_columns

    def test_load_sample_regression_schema_types(self, spark_session):
        """Test that dataset has correct column types."""
        df = load_sample_regression(spark_session, n_rows=10)
        schema = {field.name: type(field.dataType) for field in df.schema.fields}

        assert schema["bedrooms"] == IntegerType
        assert schema["bathrooms"] == IntegerType
        assert schema["sqft"] == IntegerType
        assert schema["age"] == IntegerType
        assert schema["location"] == StringType
        assert schema["condition"] == StringType
        assert schema["price"] == DoubleType

    def test_load_sample_regression_no_nulls(self, spark_session):
        """Test that dataset has no null values."""
        df = load_sample_regression(spark_session, n_rows=1000)

        for col_name in df.columns:
            null_count = df.filter(df[col_name].isNull()).count()
            assert null_count == 0, f"Column {col_name} has {null_count} null values"

    def test_load_sample_regression_valid_values(self, spark_session):
        """Test that dataset values are within expected ranges."""
        df = load_sample_regression(spark_session, n_rows=1000)

        # Check bedrooms range
        bedrooms_range = (
            df.agg({"bedrooms": "min"}).collect()[0][0],
            df.agg({"bedrooms": "max"}).collect()[0][0],
        )
        assert bedrooms_range[0] >= 1
        assert bedrooms_range[1] <= 5

        # Check age range
        age_range = df.agg({"age": "min"}).collect()[0][0], df.agg({"age": "max"}).collect()[0][0]
        assert age_range[0] >= 0
        assert age_range[1] <= 50

        # Check location values
        locations = [row["location"] for row in df.select("location").distinct().collect()]
        assert set(locations).issubset({"urban", "suburban", "rural"})

        # Check condition values
        conditions = [row["condition"] for row in df.select("condition").distinct().collect()]
        assert set(conditions).issubset({"excellent", "good", "fair", "poor"})

        # Check price is positive
        min_price = df.agg({"price": "min"}).collect()[0][0]
        assert min_price > 0

    def test_load_sample_regression_reproducible(self, spark_session):
        """Test that same seed produces same data."""
        df1 = load_sample_regression(spark_session, n_rows=100, seed=123)
        df2 = load_sample_regression(spark_session, n_rows=100, seed=123)

        # Compare first few rows
        rows1 = df1.collect()
        rows2 = df2.collect()

        for r1, r2 in zip(rows1[:10], rows2[:10]):
            assert r1 == r2

    def test_load_sample_regression_different_seeds_differ(self, spark_session):
        """Test that different seeds produce different data."""
        df1 = load_sample_regression(spark_session, n_rows=100, seed=123)
        df2 = load_sample_regression(spark_session, n_rows=100, seed=456)

        rows1 = df1.collect()
        rows2 = df2.collect()

        # At least some rows should differ
        differences = sum(1 for r1, r2 in zip(rows1, rows2) if r1 != r2)
        assert differences > 0


class TestClassificationDataset:
    """Tests for the sample classification dataset."""

    def test_load_sample_classification_returns_dataframe(self, spark_session):
        """Test that load_sample_classification returns a DataFrame."""
        df = load_sample_classification(spark_session)
        assert df is not None
        assert hasattr(df, "count")

    def test_load_sample_classification_default_row_count(self, spark_session):
        """Test that default row count is 10,000."""
        df = load_sample_classification(spark_session)
        assert df.count() == 10000

    def test_load_sample_classification_custom_row_count(self, spark_session):
        """Test that custom row count works."""
        df = load_sample_classification(spark_session, n_rows=100)
        assert df.count() == 100

    def test_load_sample_classification_schema_columns(self, spark_session):
        """Test that dataset has correct column names."""
        df = load_sample_classification(spark_session, n_rows=10)
        expected_columns = [
            "tenure",
            "monthly_charges",
            "total_charges",
            "contract",
            "payment_method",
            "churn",
        ]
        assert df.columns == expected_columns

    def test_load_sample_classification_schema_types(self, spark_session):
        """Test that dataset has correct column types."""
        df = load_sample_classification(spark_session, n_rows=10)
        schema = {field.name: type(field.dataType) for field in df.schema.fields}

        assert schema["tenure"] == IntegerType
        assert schema["monthly_charges"] == DoubleType
        assert schema["total_charges"] == DoubleType
        assert schema["contract"] == StringType
        assert schema["payment_method"] == StringType
        assert schema["churn"] == IntegerType

    def test_load_sample_classification_no_nulls(self, spark_session):
        """Test that dataset has no null values."""
        df = load_sample_classification(spark_session, n_rows=1000)

        for col_name in df.columns:
            null_count = df.filter(df[col_name].isNull()).count()
            assert null_count == 0, f"Column {col_name} has {null_count} null values"

    def test_load_sample_classification_valid_values(self, spark_session):
        """Test that dataset values are within expected ranges."""
        df = load_sample_classification(spark_session, n_rows=1000)

        # Check tenure range
        tenure_range = (
            df.agg({"tenure": "min"}).collect()[0][0],
            df.agg({"tenure": "max"}).collect()[0][0],
        )
        assert tenure_range[0] >= 1
        assert tenure_range[1] <= 72

        # Check monthly_charges range
        charges_range = (
            df.agg({"monthly_charges": "min"}).collect()[0][0],
            df.agg({"monthly_charges": "max"}).collect()[0][0],
        )
        assert charges_range[0] >= 20
        assert charges_range[1] <= 120

        # Check contract values
        contracts = [row["contract"] for row in df.select("contract").distinct().collect()]
        assert set(contracts).issubset({"month-to-month", "one_year", "two_year"})

        # Check payment_method values
        methods = [
            row["payment_method"] for row in df.select("payment_method").distinct().collect()
        ]
        assert set(methods).issubset(
            {"credit_card", "bank_transfer", "electronic_check", "mailed_check"}
        )

        # Check churn is binary
        churn_values = [row["churn"] for row in df.select("churn").distinct().collect()]
        assert set(churn_values).issubset({0, 1})

    def test_load_sample_classification_class_balance(self, spark_session):
        """Test that churn class is approximately 30%."""
        df = load_sample_classification(spark_session, n_rows=10000)

        churn_count = df.filter(df["churn"] == 1).count()
        churn_rate = churn_count / 10000

        # Allow for some variance (20% - 40%)
        assert 0.20 <= churn_rate <= 0.40, f"Churn rate {churn_rate:.2%} outside expected range"

    def test_load_sample_classification_reproducible(self, spark_session):
        """Test that same seed produces same data."""
        df1 = load_sample_classification(spark_session, n_rows=100, seed=123)
        df2 = load_sample_classification(spark_session, n_rows=100, seed=123)

        rows1 = df1.collect()
        rows2 = df2.collect()

        for r1, r2 in zip(rows1[:10], rows2[:10]):
            assert r1 == r2


class TestDatasetInfo:
    """Tests for the dataset_info function."""

    def test_dataset_info_regression(self, capsys):
        """Test that dataset_info prints regression info."""
        dataset_info("regression")
        captured = capsys.readouterr()

        assert "Housing Prices" in captured.out
        assert "bedrooms" in captured.out
        assert "price" in captured.out
        assert "LABEL COLUMN" in captured.out

    def test_dataset_info_classification(self, capsys):
        """Test that dataset_info prints classification info."""
        dataset_info("classification")
        captured = capsys.readouterr()

        assert "Customer Churn" in captured.out
        assert "tenure" in captured.out
        assert "churn" in captured.out
        assert "LABEL COLUMN" in captured.out

    def test_dataset_info_invalid_raises_error(self):
        """Test that invalid dataset name raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            dataset_info("invalid")

        assert "Unknown dataset" in str(exc_info.value)
        assert "invalid" in str(exc_info.value)

    def test_dataset_info_regression_contains_usage(self, capsys):
        """Test that regression info contains usage example."""
        dataset_info("regression")
        captured = capsys.readouterr()

        assert "load_sample_regression" in captured.out

    def test_dataset_info_classification_contains_usage(self, capsys):
        """Test that classification info contains usage example."""
        dataset_info("classification")
        captured = capsys.readouterr()

        assert "load_sample_classification" in captured.out
