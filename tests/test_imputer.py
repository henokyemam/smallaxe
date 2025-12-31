"""Tests for the Imputer preprocessing component."""

import pytest
from pyspark.sql import functions as F

from smallaxe.exceptions import (
    ColumnNotFoundError,
    ModelNotFittedError,
    ValidationError,
)
from smallaxe.preprocessing import Imputer


@pytest.fixture
def df_with_nulls(spark_session):
    """Create a DataFrame with null values for testing imputation."""
    data = [
        (1, 25.0, 50000.0, "A"),
        (2, 30.0, None, "B"),
        (3, None, 70000.0, "A"),
        (4, 40.0, 80000.0, None),
        (5, 45.0, 90000.0, "B"),
        (6, None, None, None),
        (7, 35.0, 60000.0, "C"),
    ]
    columns = ["id", "age", "income", "category"]
    return spark_session.createDataFrame(data, columns)


@pytest.fixture
def df_with_many_categories(spark_session):
    """Create a DataFrame with many categorical values for mode testing."""
    data = [
        (1, "A"),
        (2, "A"),
        (3, "A"),
        (4, "B"),
        (5, "B"),
        (6, "C"),
        (7, None),
        (8, None),
    ]
    columns = ["id", "category"]
    return spark_session.createDataFrame(data, columns)


class TestImputerInit:
    """Tests for Imputer initialization."""

    def test_default_strategies(self):
        """Test that default strategies are set correctly."""
        imputer = Imputer()
        assert imputer._numerical_strategy == "mean"
        assert imputer._categorical_strategy == "mode"

    def test_custom_numerical_strategy(self):
        """Test setting custom numerical strategy."""
        imputer = Imputer(numerical_strategy="median")
        assert imputer._numerical_strategy == "median"

    def test_custom_numerical_value(self):
        """Test setting custom numerical value for imputation."""
        imputer = Imputer(numerical_strategy=-999)
        assert imputer._numerical_strategy == -999

    def test_custom_categorical_value(self):
        """Test setting custom categorical value for imputation."""
        imputer = Imputer(categorical_strategy="UNKNOWN")
        assert imputer._categorical_strategy == "UNKNOWN"

    def test_invalid_numerical_strategy_raises_error(self):
        """Test that invalid numerical strategy raises ValidationError."""
        with pytest.raises(ValidationError, match="Invalid numerical_strategy"):
            Imputer(numerical_strategy="invalid")

    def test_invalid_categorical_strategy_type_raises_error(self):
        """Test that non-string categorical strategy raises ValidationError."""
        with pytest.raises(ValidationError, match="categorical_strategy must be a string"):
            Imputer(categorical_strategy=123)


class TestImputerMeanStrategy:
    """Tests for Imputer with mean strategy."""

    def test_fit_computes_mean(self, df_with_nulls):
        """Test that fit computes mean values correctly."""
        imputer = Imputer(numerical_strategy="mean")
        imputer.fit(df_with_nulls, numerical_cols=["age", "income"])

        # age: (25 + 30 + 40 + 45 + 35) / 5 = 35.0
        # income: (50000 + 70000 + 80000 + 90000 + 60000) / 5 = 70000.0
        assert imputer.numerical_fill_values["age"] == 35.0
        assert imputer.numerical_fill_values["income"] == 70000.0

    def test_transform_fills_nulls_with_mean(self, df_with_nulls):
        """Test that transform fills nulls with mean values."""
        imputer = Imputer(numerical_strategy="mean")
        imputer.fit(df_with_nulls, numerical_cols=["age", "income"])
        result = imputer.transform(df_with_nulls)

        # Check no nulls remain
        null_count = result.filter(F.col("age").isNull() | F.col("income").isNull()).count()
        assert null_count == 0

        # Check specific filled values
        row_3 = result.filter(F.col("id") == 3).first()
        assert row_3["age"] == 35.0  # mean

        row_2 = result.filter(F.col("id") == 2).first()
        assert row_2["income"] == 70000.0  # mean

    def test_fit_transform(self, df_with_nulls):
        """Test fit_transform convenience method."""
        imputer = Imputer(numerical_strategy="mean")
        result = imputer.fit_transform(df_with_nulls, numerical_cols=["age", "income"])

        null_count = result.filter(F.col("age").isNull() | F.col("income").isNull()).count()
        assert null_count == 0


class TestImputerMedianStrategy:
    """Tests for Imputer with median strategy."""

    def test_fit_computes_median(self, df_with_nulls):
        """Test that fit computes median values correctly."""
        imputer = Imputer(numerical_strategy="median")
        imputer.fit(df_with_nulls, numerical_cols=["age"])

        # age values without nulls: [25, 30, 35, 40, 45] -> median = 35
        assert imputer.numerical_fill_values["age"] == 35.0

    def test_transform_fills_nulls_with_median(self, df_with_nulls):
        """Test that transform fills nulls with median values."""
        imputer = Imputer(numerical_strategy="median")
        result = imputer.fit_transform(df_with_nulls, numerical_cols=["age"])

        row_3 = result.filter(F.col("id") == 3).first()
        assert row_3["age"] == 35.0  # median


class TestImputerModeStrategy:
    """Tests for Imputer with mode strategy."""

    def test_fit_computes_mode_categorical(self, df_with_many_categories):
        """Test that fit computes mode for categorical columns."""
        imputer = Imputer(categorical_strategy="mode")
        imputer.fit(df_with_many_categories, categorical_cols=["category"])

        # category: A appears 3 times, B appears 2 times, C appears 1 time
        assert imputer.categorical_fill_values["category"] == "A"

    def test_transform_fills_nulls_with_mode(self, df_with_many_categories):
        """Test that transform fills categorical nulls with mode."""
        imputer = Imputer(categorical_strategy="mode")
        result = imputer.fit_transform(df_with_many_categories, categorical_cols=["category"])

        # Check no nulls remain
        null_count = result.filter(F.col("category").isNull()).count()
        assert null_count == 0

        # Check nulls were filled with mode
        row_7 = result.filter(F.col("id") == 7).first()
        assert row_7["category"] == "A"

    def test_numerical_mode_strategy(self, spark_session):
        """Test mode strategy for numerical columns."""
        data = [
            (1, 10.0),
            (2, 10.0),
            (3, 10.0),
            (4, 20.0),
            (5, 20.0),
            (6, None),
        ]
        df = spark_session.createDataFrame(data, ["id", "value"])

        imputer = Imputer(numerical_strategy="mode")
        result = imputer.fit_transform(df, numerical_cols=["value"])

        row_6 = result.filter(F.col("id") == 6).first()
        assert row_6["value"] == 10.0  # mode


class TestImputerCustomValue:
    """Tests for Imputer with custom values."""

    def test_custom_numerical_value(self, df_with_nulls):
        """Test imputation with custom numerical value."""
        imputer = Imputer(numerical_strategy=-999)
        result = imputer.fit_transform(df_with_nulls, numerical_cols=["age", "income"])

        row_3 = result.filter(F.col("id") == 3).first()
        assert row_3["age"] == -999.0

        row_2 = result.filter(F.col("id") == 2).first()
        assert row_2["income"] == -999.0

    def test_custom_categorical_value(self, df_with_nulls):
        """Test imputation with custom categorical value."""
        imputer = Imputer(categorical_strategy="UNKNOWN")
        result = imputer.fit_transform(df_with_nulls, categorical_cols=["category"])

        row_4 = result.filter(F.col("id") == 4).first()
        assert row_4["category"] == "UNKNOWN"


class TestImputerErrors:
    """Tests for Imputer error handling."""

    def test_transform_before_fit_raises_error(self, df_with_nulls):
        """Test that transform before fit raises ModelNotFittedError."""
        imputer = Imputer()
        with pytest.raises(ModelNotFittedError, match="has not been fitted"):
            imputer.transform(df_with_nulls)

    def test_missing_column_in_fit_raises_error(self, df_with_nulls):
        """Test that missing column in fit raises ColumnNotFoundError."""
        imputer = Imputer()
        with pytest.raises(ColumnNotFoundError, match="nonexistent"):
            imputer.fit(df_with_nulls, numerical_cols=["nonexistent"])

    def test_missing_column_in_transform_raises_error(self, spark_session):
        """Test that missing column in transform raises ColumnNotFoundError."""
        # Fit on one DataFrame
        df_fit = spark_session.createDataFrame([(1, 25.0), (2, None)], ["id", "age"])
        imputer = Imputer()
        imputer.fit(df_fit, numerical_cols=["age"])

        # Transform on a different DataFrame missing the column
        df_transform = spark_session.createDataFrame([(1, "A"), (2, "B")], ["id", "category"])
        with pytest.raises(ColumnNotFoundError, match="age"):
            imputer.transform(df_transform)

    def test_access_fill_values_before_fit_raises_error(self):
        """Test accessing fill values before fit raises ModelNotFittedError."""
        imputer = Imputer()
        with pytest.raises(ModelNotFittedError):
            _ = imputer.numerical_fill_values
        with pytest.raises(ModelNotFittedError):
            _ = imputer.categorical_fill_values


class TestImputerEdgeCases:
    """Tests for Imputer edge cases."""

    def test_empty_column_lists(self, df_with_nulls):
        """Test that empty column lists work without error."""
        imputer = Imputer()
        result = imputer.fit_transform(df_with_nulls)

        # Should return unchanged DataFrame
        assert result.count() == df_with_nulls.count()

    def test_all_nulls_in_column(self, spark_session):
        """Test handling of column with all null values."""
        from pyspark.sql.types import DoubleType, IntegerType, StructField, StructType

        schema = StructType(
            [
                StructField("id", IntegerType(), False),
                StructField("value", DoubleType(), True),
            ]
        )
        data = [(1, None), (2, None), (3, None)]
        df = spark_session.createDataFrame(data, schema)

        imputer = Imputer(numerical_strategy="mean")
        imputer.fit(df, numerical_cols=["value"])

        # Should default to 0.0 when all values are null
        assert imputer.numerical_fill_values["value"] == 0.0

    def test_no_nulls_in_column(self, spark_session):
        """Test handling of column with no null values."""
        data = [(1, 10.0), (2, 20.0), (3, 30.0)]
        df = spark_session.createDataFrame(data, ["id", "value"])

        imputer = Imputer(numerical_strategy="mean")
        result = imputer.fit_transform(df, numerical_cols=["value"])

        # Should leave data unchanged
        values = [row["value"] for row in result.collect()]
        assert values == [10.0, 20.0, 30.0]

    def test_mixed_numerical_and_categorical(self, df_with_nulls):
        """Test imputing both numerical and categorical columns."""
        imputer = Imputer(numerical_strategy="mean", categorical_strategy="mode")
        result = imputer.fit_transform(
            df_with_nulls,
            numerical_cols=["age", "income"],
            categorical_cols=["category"],
        )

        # Check no nulls remain in any imputed column
        null_count = result.filter(
            F.col("age").isNull() | F.col("income").isNull() | F.col("category").isNull()
        ).count()
        assert null_count == 0
