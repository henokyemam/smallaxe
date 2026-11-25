"""Tests for the Imputer preprocessing component."""

from decimal import Decimal

import pytest
from pyspark.sql import DataFrame
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
        """Test that default strategies are None."""
        imputer = Imputer()
        assert imputer._numerical_strategy is None
        assert imputer._categorical_strategy is None

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

    def test_numerical_cols_without_strategy_raises_error(self, df_with_nulls):
        """Test that providing numerical_cols without numerical_strategy raises ValidationError."""
        imputer = Imputer()
        with pytest.raises(
            ValidationError, match="numerical_cols provided but numerical_strategy is None"
        ):
            imputer.fit(df_with_nulls, numerical_cols=["age"])

    def test_categorical_cols_without_strategy_raises_error(self, df_with_nulls):
        """Test that providing categorical_cols without categorical_strategy raises ValidationError."""
        imputer = Imputer()
        with pytest.raises(
            ValidationError, match="categorical_cols provided but categorical_strategy is None"
        ):
            imputer.fit(df_with_nulls, categorical_cols=["category"])

    def test_numerical_only_imputer(self, df_with_nulls):
        """Test that imputer with only numerical_strategy works without categorical_strategy."""
        imputer = Imputer(numerical_strategy="mean")
        imputer.fit(df_with_nulls, numerical_cols=["age", "income"])
        result = imputer.transform(df_with_nulls)
        assert result.filter(result.age.isNull()).count() == 0

    def test_categorical_only_imputer(self, df_with_nulls):
        """Test that imputer with only categorical_strategy works without numerical_strategy."""
        imputer = Imputer(categorical_strategy="mode")
        imputer.fit(df_with_nulls, categorical_cols=["category"])
        result = imputer.transform(df_with_nulls)
        assert result.filter(result.category.isNull()).count() == 0


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
        imputer = Imputer(categorical_strategy="most_frequent")
        imputer.fit(df_with_many_categories, categorical_cols=["category"])

        # category: A appears 3 times, B appears 2 times, C appears 1 time
        assert imputer.categorical_fill_values["category"] == "A"

    def test_transform_fills_nulls_with_mode(self, df_with_many_categories):
        """Test that transform fills categorical nulls with mode."""
        imputer = Imputer(categorical_strategy="most_frequent")
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
        imputer = Imputer(numerical_strategy="mean")
        with pytest.raises(ColumnNotFoundError, match="nonexistent"):
            imputer.fit(df_with_nulls, numerical_cols=["nonexistent"])

    def test_missing_column_in_transform_raises_error(self, spark_session):
        """Test that missing column in transform raises ColumnNotFoundError."""
        # Fit on one DataFrame
        df_fit = spark_session.createDataFrame([(1, 25.0), (2, None)], ["id", "age"])
        imputer = Imputer(numerical_strategy="mean")
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

    @pytest.mark.parametrize(
        "kwargs",
        [{}, {"numerical_cols": [], "categorical_cols": []}, {"numerical_cols": None}],
    )
    def test_empty_column_lists_raise(self, df_with_nulls, kwargs):
        """fit with nothing to impute raises, like Scaler and Encoder do."""
        imputer = Imputer(numerical_strategy="mean", categorical_strategy="most_frequent")
        with pytest.raises(ValidationError, match="cannot both be empty"):
            imputer.fit(df_with_nulls, **kwargs)
        assert not imputer._is_fitted

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
        imputer = Imputer(numerical_strategy="mean", categorical_strategy="most_frequent")
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


class TestImputerNonStringCategorical:
    """Every declared categorical column is really imputed, in its own type, or fit raises.

    0.10 filled with ``DataFrame.fillna(dict)``, which casts the fill to the column's type.
    A fill that does not cast (``'UNKNOWN'`` on an integer column, ``'-1'`` on a boolean)
    left the nulls in place without a message, and a lossy one (``'1.5'`` on an integer
    column) filled a value other than the configured one. The type-compatible cases below
    worked in 0.10 too; they pin the contract. The rejection cases are the regressions.
    """

    @pytest.fixture
    def df_int_category(self, spark_session):
        rows = [(1, 3), (2, 3), (3, 5), (4, None), (5, 5), (6, 3)]
        return spark_session.createDataFrame(rows, "id int, code int")

    def test_most_frequent_fills_integer_column(self, df_int_category):
        imputer = Imputer(categorical_strategy="most_frequent")
        result = imputer.fit_transform(df_int_category, categorical_cols=["code"])

        assert imputer.categorical_fill_values == {"code": "3"}
        assert dict(result.dtypes)["code"] == "int"
        assert result.filter(F.col("code").isNull()).count() == 0
        assert result.filter(F.col("id") == 4).first()["code"] == 3

    def test_custom_value_of_the_column_type_fills_integer_column(self, df_int_category):
        imputer = Imputer(categorical_strategy="-1")
        result = imputer.fit_transform(df_int_category, categorical_cols=["code"])

        assert dict(result.dtypes)["code"] == "int"
        assert result.filter(F.col("id") == 4).first()["code"] == -1
        assert result.filter(F.col("code").isNull()).count() == 0

    def test_custom_value_of_another_type_is_rejected_at_fit(self, df_int_category):
        imputer = Imputer(categorical_strategy="UNKNOWN")
        with pytest.raises(ValidationError, match="'code' of type int") as info:
            imputer.fit(df_int_category, categorical_cols=["code"])
        assert "'UNKNOWN'" in str(info.value)

    @pytest.mark.parametrize("fill", ["1.5", "3.000001", "99999999999"])
    def test_lossy_custom_value_is_rejected_at_fit(self, df_int_category, fill):
        imputer = Imputer(categorical_strategy=fill)
        with pytest.raises(ValidationError, match="'code' of type int") as info:
            imputer.fit(df_int_category, categorical_cols=["code"])
        assert repr(fill) in str(info.value)

    def test_exact_custom_value_in_another_spelling_is_accepted(self, df_int_category):
        result = Imputer(categorical_strategy="7.0").fit_transform(
            df_int_category, categorical_cols=["code"]
        )
        assert result.filter(F.col("id") == 4).first()["code"] == 7

    def test_lossy_custom_value_on_decimal_column_is_rejected(self, spark_session):
        df = spark_session.createDataFrame(
            [(1, Decimal("1.25")), (2, None)], "id int, price decimal(5,2)"
        )
        with pytest.raises(ValidationError, match="'price' of type decimal"):
            Imputer(categorical_strategy="1.255").fit(df, categorical_cols=["price"])
        result = Imputer(categorical_strategy="1.5").fit_transform(df, categorical_cols=["price"])
        assert result.filter(F.col("id") == 2).first()["price"] == Decimal("1.50")

    def test_most_frequent_on_double_column_ignores_nan(self, spark_session):
        nan = float("nan")
        df = spark_session.createDataFrame(
            [(1, nan), (2, nan), (3, 1.0), (4, None)], "id int, d double"
        )
        imputer = Imputer(categorical_strategy="most_frequent")
        result = imputer.fit_transform(df, categorical_cols=["d"])

        assert imputer.categorical_fill_values == {"d": "1.0"}
        assert [row["d"] for row in result.orderBy("id").collect()] == [1.0, 1.0, 1.0, 1.0]

    def test_nan_in_double_column_is_filled_like_null(self, spark_session):
        """As 0.10's fillna did, and as the numerical path does."""
        df = spark_session.createDataFrame(
            [(1, float("nan")), (2, None), (3, 1.0)], "id int, d double"
        )
        result = Imputer(categorical_strategy="2.5").fit_transform(df, categorical_cols=["d"])
        assert [row["d"] for row in result.orderBy("id").collect()] == [2.5, 2.5, 1.0]

    def test_unexpected_spark_failure_is_not_reported_as_a_bad_value(
        self, df_int_category, monkeypatch
    ):
        """Only a failed cast means the value does not fit; other errors propagate."""

        def broken_first(self):
            raise RuntimeError("gateway down")

        monkeypatch.setattr(DataFrame, "first", broken_first)
        with pytest.raises(RuntimeError, match="gateway down"):
            Imputer(categorical_strategy="3").fit(df_int_category, categorical_cols=["code"])

    def test_most_frequent_on_all_null_integer_column_is_rejected(self, spark_session):
        df = spark_session.createDataFrame([(1, None), (2, None)], "id int, code int")
        with pytest.raises(ValidationError, match="'code' of type int"):
            Imputer(categorical_strategy="most_frequent").fit(df, categorical_cols=["code"])

    def test_most_frequent_fills_boolean_column(self, spark_session):
        df = spark_session.createDataFrame(
            [(1, True), (2, None), (3, True), (4, False)], "id int, flag boolean"
        )
        imputer = Imputer(categorical_strategy="most_frequent")
        result = imputer.fit_transform(df, categorical_cols=["flag"])

        assert imputer.categorical_fill_values == {"flag": "true"}
        assert result.filter(F.col("id") == 2).first()["flag"] is True

    def test_string_column_is_unchanged(self, df_with_nulls):
        imputer = Imputer(categorical_strategy="most_frequent")
        result = imputer.fit_transform(df_with_nulls, categorical_cols=["category"])

        assert dict(result.dtypes)["category"] == "string"
        assert result.columns == df_with_nulls.columns

    def test_most_frequent_ties_go_to_the_smallest_value(self, spark_session):
        df = spark_session.createDataFrame(
            [(1, "b"), (2, "a"), (3, "b"), (4, "a"), (5, None)], "id int, c string"
        )
        for partitions in (1, 3):
            imputer = Imputer(categorical_strategy="most_frequent")
            imputer.fit(df.repartition(partitions), categorical_cols=["c"])
            assert imputer.categorical_fill_values == {"c": "a"}
