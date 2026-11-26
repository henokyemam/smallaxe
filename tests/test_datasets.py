"""Tests for smallaxe sample datasets."""

import hashlib
import random

import pytest
from pyspark.sql.types import DoubleType, IntegerType, StringType

import smallaxe
from smallaxe.datasets import (
    dataset_info,
    load_sample_classification,
    load_sample_regression,
)
from smallaxe.exceptions import ValidationError


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


class TestRandomState:
    """Generating data uses its own random state, never Python's global one."""

    # sha256 of repr(rows) for n_rows=200, recorded from the 0.10 generators
    # (which seeded the global ``random`` module) before they moved to a local Random.
    BASELINE = {
        ("regression", 42): "db745dec468768f29c5c74058a70f9e86c1d80982e66183850db5919445cb7f9",
        ("regression", 7): "0a6ca4474e05316b98d2a043e12496e464823bdccedbfb857113e06abe9a6cc5",
        ("classification", 42): (
            "4e79577cdf986965062b500b78edd5ad22332b659ba837837ddb4318320f2925"
        ),
        ("classification", 7): "477b4c96f3fa9620b3a553ee1a544bed55729566ebb09be4b1ee1a20cfd8beee",
    }

    @pytest.mark.parametrize("kind, seed", sorted(BASELINE))
    def test_generated_rows_are_unchanged(self, kind, seed):
        from smallaxe.datasets import _data

        generate = getattr(_data, f"_generate_{kind}_data")
        rows = generate(n_rows=200, seed=seed)
        assert hashlib.sha256(repr(rows).encode()).hexdigest() == self.BASELINE[(kind, seed)]

    @pytest.mark.parametrize("load", [load_sample_regression, load_sample_classification])
    def test_loading_leaves_the_global_random_state_alone(self, spark_session, load):
        random.seed(2024)
        expected = random.Random(2024).random()

        load(spark_session, n_rows=20, seed=1)

        assert random.random() == expected

    def test_loading_does_not_fix_the_unseeded_split_seed(self, spark_session):
        """The old global reseed made every later unseeded fit draw the same split."""
        from smallaxe.training.mixins.validation_mixin import _resolve_split_seed

        smallaxe.set_seed(None)
        drawn = set()
        for _ in range(3):
            load_sample_regression(spark_session, n_rows=5, seed=42)
            drawn.add(_resolve_split_seed())
        assert len(drawn) == 3


class TestDatasetInfo:
    """Tests for the dataset_info function."""

    def test_dataset_info_regression(self, capsys):
        """Test that dataset_info prints regression info."""
        dataset_info("regression", verbose=True)
        captured = capsys.readouterr()

        assert "Housing Prices" in captured.out
        assert "bedrooms" in captured.out
        assert "price" in captured.out
        assert "LABEL COLUMN" in captured.out

    def test_dataset_info_classification(self, capsys):
        """Test that dataset_info prints classification info."""
        dataset_info("classification", verbose=True)
        captured = capsys.readouterr()

        assert "Customer Churn" in captured.out
        assert "tenure" in captured.out
        assert "churn" in captured.out
        assert "LABEL COLUMN" in captured.out

    def test_dataset_info_invalid_raises_error(self):
        """Test that invalid dataset name raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            dataset_info("invalid", verbose=True)

        assert "Unknown dataset" in str(exc_info.value)
        assert "invalid" in str(exc_info.value)

    def test_dataset_info_regression_contains_usage(self, capsys):
        """Test that regression info contains usage example."""
        dataset_info("regression", verbose=True)
        captured = capsys.readouterr()

        assert "load_sample_regression" in captured.out

    def test_dataset_info_classification_contains_usage(self, capsys):
        """Test that classification info contains usage example."""
        dataset_info("classification", verbose=True)
        captured = capsys.readouterr()

        assert "load_sample_classification" in captured.out

    def test_dataset_info_returns_the_description(self):
        """dataset_info returns a fresh dict describing the dataset."""
        info = dataset_info("classification", verbose=False)

        assert info["title"] == "Customer Churn"
        assert info["task"] == "binary"
        assert list(info["columns"]) == [
            "tenure",
            "monthly_charges",
            "total_charges",
            "contract",
            "payment_method",
            "churn",
        ]
        assert info["columns"]["churn"]["type"] == "int"
        info["label"] = "changed"
        assert dataset_info("classification", verbose=False)["label"] == "churn"


class TestActiveSession:
    """``spark`` is optional: the loaders default to the active SparkSession."""

    @pytest.mark.parametrize("loader", [load_sample_regression, load_sample_classification])
    def test_spark_defaults_to_the_active_session(self, spark_session, loader):
        from pyspark.sql import SparkSession

        assert SparkSession.getActiveSession() is spark_session
        implicit = loader(n_rows=20, seed=3)
        explicit = loader(spark_session, n_rows=20, seed=3)

        assert implicit.sparkSession is spark_session
        assert implicit.collect() == explicit.collect()

    @pytest.mark.parametrize("loader", [load_sample_regression, load_sample_classification])
    def test_no_active_session_is_a_validation_error(self, monkeypatch, loader):
        from pyspark.sql import SparkSession

        monkeypatch.setattr(SparkSession, "getActiveSession", classmethod(lambda cls: None))
        with pytest.raises(ValidationError, match="no active SparkSession"):
            loader(n_rows=5)

    @pytest.mark.parametrize("loader", [load_sample_regression, load_sample_classification])
    def test_spark_must_be_a_session(self, loader):
        with pytest.raises(ValidationError, match="SparkSession"):
            loader("not a session", n_rows=5)

    @pytest.mark.parametrize("loader", [load_sample_regression, load_sample_classification])
    @pytest.mark.parametrize("n_rows", [0, -1, 2.5, True, "10"])
    def test_n_rows_must_be_a_positive_int(self, spark_session, loader, n_rows):
        with pytest.raises(ValidationError, match="n_rows"):
            loader(spark_session, n_rows=n_rows)

    @pytest.mark.parametrize("loader", [load_sample_regression, load_sample_classification])
    @pytest.mark.parametrize("seed", [1.5, True, "1", None])
    def test_seed_must_be_an_int(self, spark_session, loader, seed):
        with pytest.raises(ValidationError, match="seed"):
            loader(spark_session, n_rows=5, seed=seed)


class TestDatasetInfoContract:
    def test_default_returns_the_dict_without_printing(self, capsys):
        info = dataset_info("regression")

        assert info["label"] == "price"
        assert capsys.readouterr().out == ""

    def test_verbose_is_keyword_only(self):
        with pytest.raises(TypeError):
            dataset_info("regression", True)

    @pytest.mark.parametrize("name", [None, 3, ["regression"]])
    def test_non_string_name_is_a_validation_error(self, name):
        with pytest.raises(ValidationError, match="Unknown dataset"):
            dataset_info(name)
