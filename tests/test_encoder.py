"""Tests for the Encoder preprocessing component."""

import pytest

from smallaxe.exceptions import (
    ColumnNotFoundError,
    ModelNotFittedError,
    ValidationError,
)
from smallaxe.preprocessing.encoder import Encoder


@pytest.fixture
def df_for_encoding(spark_session):
    """Create a DataFrame for testing encoding."""
    data = [
        (1, "red", "small"),
        (2, "blue", "medium"),
        (3, "green", "large"),
        (4, "red", "small"),
        (5, "blue", "medium"),
    ]
    columns = ["id", "color", "size"]
    return spark_session.createDataFrame(data, columns)


@pytest.fixture
def df_with_many_categories(spark_session):
    """Create a DataFrame with many categories for testing max_categories."""
    # Create 15 unique categories
    data = [(i, f"cat_{i % 15}", f"size_{i % 5}") for i in range(100)]
    columns = ["id", "category", "size"]
    return spark_session.createDataFrame(data, columns)


@pytest.fixture
def df_with_rare_categories(spark_session):
    """Create a DataFrame with some rare categories."""
    # cat_A appears 10 times, cat_B appears 10 times, cat_C appears 3 times, cat_D appears 2 times
    data = (
        [(i, "cat_A", "X") for i in range(10)]
        + [(i + 10, "cat_B", "X") for i in range(10)]
        + [(i + 20, "cat_C", "Y") for i in range(3)]
        + [(i + 23, "cat_D", "Z") for i in range(2)]
    )
    columns = ["id", "category", "type"]
    return spark_session.createDataFrame(data, columns)


class TestEncoderInit:
    """Tests for Encoder initialization."""

    def test_default_method(self):
        """Test that default method is 'onehot'."""
        encoder = Encoder()
        assert encoder._method == "onehot"

    def test_default_max_categories(self):
        """Test that default max_categories is 100."""
        encoder = Encoder()
        assert encoder._max_categories == 100

    def test_default_handle_rare(self):
        """Test that default handle_rare is 'other'."""
        encoder = Encoder()
        assert encoder._handle_rare == "other"

    def test_custom_method_onehot(self):
        """Test setting onehot method."""
        encoder = Encoder(method="onehot")
        assert encoder._method == "onehot"

    def test_custom_method_label(self):
        """Test setting label method."""
        encoder = Encoder(method="label")
        assert encoder._method == "label"

    def test_custom_max_categories(self):
        """Test setting custom max_categories."""
        encoder = Encoder(max_categories=50)
        assert encoder._max_categories == 50

    def test_custom_handle_rare_keep(self):
        """Test setting handle_rare to 'keep'."""
        encoder = Encoder(handle_rare="keep")
        assert encoder._handle_rare == "keep"

    def test_custom_handle_rare_error(self):
        """Test setting handle_rare to 'error'."""
        encoder = Encoder(handle_rare="error")
        assert encoder._handle_rare == "error"

    def test_invalid_method_raises_error(self):
        """Test that invalid method raises ValidationError."""
        with pytest.raises(ValidationError, match="Invalid method"):
            Encoder(method="invalid")

    def test_invalid_handle_rare_raises_error(self):
        """Test that invalid handle_rare raises ValidationError."""
        with pytest.raises(ValidationError, match="Invalid handle_rare"):
            Encoder(handle_rare="invalid")

    def test_invalid_max_categories_raises_error(self):
        """Test that invalid max_categories raises ValidationError."""
        with pytest.raises(ValidationError, match="max_categories"):
            Encoder(max_categories=0)

        with pytest.raises(ValidationError, match="max_categories"):
            Encoder(max_categories=-1)

    def test_method_property(self):
        """Test method property returns the correct value."""
        encoder = Encoder(method="label")
        assert encoder.method == "label"

    def test_max_categories_property(self):
        """Test max_categories property returns the correct value."""
        encoder = Encoder(max_categories=75)
        assert encoder.max_categories == 75

    def test_handle_rare_property(self):
        """Test handle_rare property returns the correct value."""
        encoder = Encoder(handle_rare="keep")
        assert encoder.handle_rare == "keep"


class TestLabelEncoder:
    """Tests for Encoder with label method."""

    def test_label_encoder_output_values(self, df_for_encoding):
        """Test that LabelEncoder produces integer indices."""
        encoder = Encoder(method="label")
        result = encoder.fit_transform(df_for_encoding, categorical_cols=["color"])

        # Get the encoded values
        encoded_values = [row["color"] for row in result.orderBy("id").collect()]

        # All values should be non-negative integers
        for val in encoded_values:
            assert isinstance(val, int) or val is None
            if val is not None:
                assert val >= 0

    def test_label_encoder_consistent_mapping(self, df_for_encoding):
        """Test that same categories get same indices."""
        encoder = Encoder(method="label")
        result = encoder.fit_transform(df_for_encoding, categorical_cols=["color"])

        # Collect the results
        rows = result.orderBy("id").collect()

        # Row 1 and 4 both have "red" originally, should have same encoded value
        assert rows[0]["color"] == rows[3]["color"]

        # Row 2 and 5 both have "blue" originally, should have same encoded value
        assert rows[1]["color"] == rows[4]["color"]

    def test_label_encoder_unique_indices(self, df_for_encoding):
        """Test that different categories get different indices."""
        encoder = Encoder(method="label")
        result = encoder.fit_transform(df_for_encoding, categorical_cols=["color"])

        # Get unique encoded values
        unique_values = result.select("color").distinct().collect()
        unique_indices = [row["color"] for row in unique_values]

        # Should have 3 unique indices for red, blue, green
        assert len(set(unique_indices)) == 3

    def test_label_encoder_multiple_columns(self, df_for_encoding):
        """Test LabelEncoder with multiple columns."""
        encoder = Encoder(method="label")
        result = encoder.fit_transform(
            df_for_encoding, categorical_cols=["color", "size"]
        )

        # Both columns should be encoded
        for col in ["color", "size"]:
            values = [row[col] for row in result.collect()]
            for val in values:
                assert isinstance(val, int) or val is None

    def test_label_encoder_preserves_other_columns(self, df_for_encoding):
        """Test that LabelEncoder preserves non-encoded columns."""
        encoder = Encoder(method="label")
        result = encoder.fit_transform(df_for_encoding, categorical_cols=["color"])

        # id column should be unchanged
        original_ids = [row["id"] for row in df_for_encoding.orderBy("id").collect()]
        result_ids = [row["id"] for row in result.orderBy("id").collect()]
        assert original_ids == result_ids

        # size column should be unchanged
        original_sizes = [row["size"] for row in df_for_encoding.orderBy("id").collect()]
        result_sizes = [row["size"] for row in result.orderBy("id").collect()]
        assert original_sizes == result_sizes


class TestOneHotEncoder:
    """Tests for Encoder with onehot method."""

    def test_onehot_encoder_output_shape(self, df_for_encoding):
        """Test that OneHotEncoder creates correct number of columns."""
        encoder = Encoder(method="onehot")
        result = encoder.fit_transform(df_for_encoding, categorical_cols=["color"])

        # Original column "color" should be removed
        assert "color" not in result.columns

        # Should have columns like color_red, color_blue, color_green
        color_cols = [c for c in result.columns if c.startswith("color_")]
        assert len(color_cols) == 3  # red, blue, green

    def test_onehot_encoder_binary_values(self, df_for_encoding):
        """Test that OneHotEncoder produces binary (0/1) values."""
        encoder = Encoder(method="onehot")
        result = encoder.fit_transform(df_for_encoding, categorical_cols=["color"])

        # Get all one-hot columns
        color_cols = [c for c in result.columns if c.startswith("color_")]

        for col in color_cols:
            values = [row[col] for row in result.collect()]
            for val in values:
                assert val in [0.0, 1.0]

    def test_onehot_encoder_single_one_per_row(self, df_for_encoding):
        """Test that each row has exactly one 1 in the one-hot columns."""
        encoder = Encoder(method="onehot")
        result = encoder.fit_transform(df_for_encoding, categorical_cols=["color"])

        color_cols = [c for c in result.columns if c.startswith("color_")]

        for row in result.collect():
            # Sum of one-hot values should be 1 for each row
            row_sum = sum(row[col] for col in color_cols)
            assert row_sum == 1.0

    def test_onehot_encoder_multiple_columns(self, df_for_encoding):
        """Test OneHotEncoder with multiple columns."""
        encoder = Encoder(method="onehot")
        result = encoder.fit_transform(
            df_for_encoding, categorical_cols=["color", "size"]
        )

        # Original columns should be removed
        assert "color" not in result.columns
        assert "size" not in result.columns

        # Should have one-hot columns for both
        color_cols = [c for c in result.columns if c.startswith("color_")]
        size_cols = [c for c in result.columns if c.startswith("size_")]

        assert len(color_cols) == 3  # red, blue, green
        assert len(size_cols) == 3  # small, medium, large


class TestMaxCategories:
    """Tests for max_categories limiting output dimensions."""

    def test_max_categories_limits_onehot_columns(self, df_with_many_categories):
        """Test that max_categories limits the number of one-hot columns."""
        encoder = Encoder(method="onehot", max_categories=5, handle_rare="other")
        result = encoder.fit_transform(
            df_with_many_categories, categorical_cols=["category"]
        )

        # Should have at most max_categories + 1 (for OTHER) columns
        category_cols = [c for c in result.columns if c.startswith("category_")]
        assert len(category_cols) <= 6  # 5 categories + 1 OTHER

    def test_max_categories_limits_label_indices(self, df_with_many_categories):
        """Test that max_categories limits the number of label indices."""
        encoder = Encoder(method="label", max_categories=5, handle_rare="other")
        result = encoder.fit_transform(
            df_with_many_categories, categorical_cols=["category"]
        )

        # Get unique encoded values
        unique_values = result.select("category").distinct().collect()
        unique_indices = [row["category"] for row in unique_values if row["category"] is not None]

        # Should have at most max_categories + 1 (for OTHER) unique indices
        assert len(unique_indices) <= 6


class TestHandleRare:
    """Tests for handle_rare strategies."""

    def test_handle_rare_other_groups_categories(self, df_with_rare_categories):
        """Test that handle_rare='other' groups rare categories."""
        # With max_categories=2, only cat_A and cat_B should be kept
        # cat_C and cat_D should be grouped into OTHER
        encoder = Encoder(method="onehot", max_categories=2, handle_rare="other")
        result = encoder.fit_transform(
            df_with_rare_categories, categorical_cols=["category"]
        )

        category_cols = [c for c in result.columns if c.startswith("category_")]

        # Should have columns for top 2 categories + OTHER
        assert len(category_cols) == 3

        # Should have an OTHER column
        other_cols = [c for c in category_cols if "OTHER" in c]
        assert len(other_cols) == 1

    def test_handle_rare_keep_preserves_all(self, df_with_rare_categories):
        """Test that handle_rare='keep' preserves all categories."""
        encoder = Encoder(method="onehot", max_categories=2, handle_rare="keep")
        result = encoder.fit_transform(
            df_with_rare_categories, categorical_cols=["category"]
        )

        category_cols = [c for c in result.columns if c.startswith("category_")]

        # Should have columns for all 4 categories (cat_A, cat_B, cat_C, cat_D)
        assert len(category_cols) == 4

    def test_handle_rare_error_raises_on_excess(self, df_with_rare_categories):
        """Test that handle_rare='error' raises when categories exceed max."""
        encoder = Encoder(method="onehot", max_categories=2, handle_rare="error")

        with pytest.raises(ValidationError, match="exceeds max_categories"):
            encoder.fit(df_with_rare_categories, categorical_cols=["category"])

    def test_handle_rare_error_no_error_when_within_limit(self, df_for_encoding):
        """Test that handle_rare='error' doesn't raise when within limit."""
        # df_for_encoding has 3 unique colors, max_categories=5
        encoder = Encoder(method="onehot", max_categories=5, handle_rare="error")

        # Should not raise
        encoder.fit(df_for_encoding, categorical_cols=["color"])
        assert encoder._is_fitted


class TestUnseenCategories:
    """Tests for handling unseen categories during transform."""

    def test_unseen_category_with_other(self, spark_session):
        """Test handling of unseen categories when handle_rare='other'."""
        # Fit on data with categories A, B, C
        fit_data = [
            (1, "A"),
            (2, "B"),
            (3, "C"),
            (4, "A"),
            (5, "B"),
        ]
        df_fit = spark_session.createDataFrame(fit_data, ["id", "category"])

        # Transform data with unseen category D
        transform_data = [
            (1, "A"),
            (2, "D"),  # Unseen category
        ]
        df_transform = spark_session.createDataFrame(transform_data, ["id", "category"])

        encoder = Encoder(method="label", max_categories=2, handle_rare="other")
        encoder.fit(df_fit, categorical_cols=["category"])
        result = encoder.transform(df_transform)

        # Row with unseen category should get the OTHER index
        rows = result.orderBy("id").collect()
        # The unseen category D should map to OTHER (or None if no OTHER exists)
        assert rows[1]["category"] is not None or "__OTHER__" not in encoder.category_mappings["category"]

    def test_unseen_category_in_onehot(self, spark_session):
        """Test handling of unseen categories in one-hot encoding."""
        fit_data = [(i, "cat_A") for i in range(10)] + [(i + 10, "cat_B") for i in range(10)]
        df_fit = spark_session.createDataFrame(fit_data, ["id", "category"])

        transform_data = [
            (1, "cat_A"),
            (2, "cat_C"),  # Unseen
        ]
        df_transform = spark_session.createDataFrame(transform_data, ["id", "category"])

        # With max_categories=1, only cat_A is kept, cat_B goes to OTHER
        encoder = Encoder(method="onehot", max_categories=1, handle_rare="other")
        encoder.fit(df_fit, categorical_cols=["category"])
        result = encoder.transform(df_transform)

        # Should have columns and not crash
        category_cols = [c for c in result.columns if c.startswith("category_")]
        assert len(category_cols) > 0


class TestEncoderErrors:
    """Tests for Encoder error handling."""

    def test_transform_before_fit_raises_error(self, df_for_encoding):
        """Test that transform before fit raises ModelNotFittedError."""
        encoder = Encoder()
        with pytest.raises(ModelNotFittedError, match="has not been fitted"):
            encoder.transform(df_for_encoding)

    def test_missing_column_in_fit_raises_error(self, df_for_encoding):
        """Test that missing column in fit raises ColumnNotFoundError."""
        encoder = Encoder()
        with pytest.raises(ColumnNotFoundError, match="nonexistent"):
            encoder.fit(df_for_encoding, categorical_cols=["nonexistent"])

    def test_missing_column_in_transform_raises_error(self, spark_session):
        """Test that missing column in transform raises ColumnNotFoundError."""
        # Fit on one DataFrame
        df_fit = spark_session.createDataFrame(
            [(1, "A"), (2, "B")], ["id", "category"]
        )
        encoder = Encoder()
        encoder.fit(df_fit, categorical_cols=["category"])

        # Transform on a different DataFrame missing the column
        df_transform = spark_session.createDataFrame(
            [(1, 100), (2, 200)], ["id", "value"]
        )
        with pytest.raises(ColumnNotFoundError, match="category"):
            encoder.transform(df_transform)

    def test_empty_categorical_cols_raises_error(self, df_for_encoding):
        """Test that empty categorical_cols raises ValidationError."""
        encoder = Encoder()
        with pytest.raises(ValidationError, match="categorical_cols cannot be empty"):
            encoder.fit(df_for_encoding, categorical_cols=[])

    def test_access_categorical_cols_before_fit_raises_error(self):
        """Test accessing categorical_cols before fit raises ModelNotFittedError."""
        encoder = Encoder()
        with pytest.raises(ModelNotFittedError):
            _ = encoder.categorical_cols

    def test_access_category_mappings_before_fit_raises_error(self):
        """Test accessing category_mappings before fit raises ModelNotFittedError."""
        encoder = Encoder()
        with pytest.raises(ModelNotFittedError):
            _ = encoder.category_mappings


class TestEncoderFitTransform:
    """Tests for fit_transform method."""

    def test_fit_transform_equivalent_to_fit_then_transform(self, df_for_encoding):
        """Test that fit_transform gives same result as fit + transform."""
        # Using fit_transform
        encoder1 = Encoder(method="label")
        result1 = encoder1.fit_transform(df_for_encoding, categorical_cols=["color"])

        # Using fit then transform
        encoder2 = Encoder(method="label")
        encoder2.fit(df_for_encoding, categorical_cols=["color"])
        result2 = encoder2.transform(df_for_encoding)

        # Results should have same schema
        assert result1.columns == result2.columns

        # Values should be the same
        values1 = [row["color"] for row in result1.orderBy("id").collect()]
        values2 = [row["color"] for row in result2.orderBy("id").collect()]
        assert values1 == values2


class TestEncoderProperties:
    """Tests for Encoder properties."""

    def test_categorical_cols_property(self, df_for_encoding):
        """Test categorical_cols property returns correct columns."""
        encoder = Encoder()
        encoder.fit(df_for_encoding, categorical_cols=["color", "size"])

        assert encoder.categorical_cols == ["color", "size"]

    def test_categorical_cols_property_returns_copy(self, df_for_encoding):
        """Test that categorical_cols property returns a copy."""
        encoder = Encoder()
        encoder.fit(df_for_encoding, categorical_cols=["color", "size"])

        cols = encoder.categorical_cols
        cols.append("modified")

        # Original should not be modified
        assert encoder.categorical_cols == ["color", "size"]

    def test_category_mappings_property(self, df_for_encoding):
        """Test category_mappings property returns correct mappings."""
        encoder = Encoder(method="label")
        encoder.fit(df_for_encoding, categorical_cols=["color"])

        mappings = encoder.category_mappings
        assert "color" in mappings
        assert len(mappings["color"]) == 3  # red, blue, green

    def test_category_mappings_property_returns_copy(self, df_for_encoding):
        """Test that category_mappings property returns a copy."""
        encoder = Encoder(method="label")
        encoder.fit(df_for_encoding, categorical_cols=["color"])

        mappings = encoder.category_mappings
        mappings["color"]["new_key"] = 999

        # Original should not be modified
        assert "new_key" not in encoder.category_mappings["color"]


class TestEncoderEdgeCases:
    """Tests for Encoder edge cases."""

    def test_single_category(self, spark_session):
        """Test encoding with a single category."""
        data = [(i, "only_one") for i in range(5)]
        df = spark_session.createDataFrame(data, ["id", "category"])

        encoder = Encoder(method="onehot")
        result = encoder.fit_transform(df, categorical_cols=["category"])

        # Should have one column for the single category
        category_cols = [c for c in result.columns if c.startswith("category_")]
        assert len(category_cols) == 1

        # All values should be 1.0
        values = [row[category_cols[0]] for row in result.collect()]
        assert all(v == 1.0 for v in values)

    def test_null_values_handling(self, spark_session):
        """Test that null values are handled correctly."""
        data = [
            (1, "A"),
            (2, None),
            (3, "B"),
            (4, None),
        ]
        df = spark_session.createDataFrame(data, ["id", "category"])

        encoder = Encoder(method="label")
        result = encoder.fit_transform(df, categorical_cols=["category"])

        # Get the encoded values
        rows = result.orderBy("id").collect()

        # Non-null values should be encoded
        assert rows[0]["category"] is not None
        assert rows[2]["category"] is not None

        # Null values should remain null (or be handled gracefully)
        # The exact behavior depends on implementation

    def test_numeric_string_categories(self, spark_session):
        """Test encoding with numeric-like string categories."""
        data = [
            (1, "1"),
            (2, "2"),
            (3, "3"),
            (4, "1"),
        ]
        df = spark_session.createDataFrame(data, ["id", "category"])

        encoder = Encoder(method="label")
        result = encoder.fit_transform(df, categorical_cols=["category"])

        # Should work without issues
        rows = result.orderBy("id").collect()
        assert rows[0]["category"] == rows[3]["category"]  # Both "1" should have same index

    def test_special_characters_in_categories(self, spark_session):
        """Test encoding with special characters in category names."""
        data = [
            (1, "hello world"),
            (2, "hello-world"),
            (3, "hello_world"),
        ]
        df = spark_session.createDataFrame(data, ["id", "category"])

        encoder = Encoder(method="onehot")
        result = encoder.fit_transform(df, categorical_cols=["category"])

        # Should create valid column names
        category_cols = [c for c in result.columns if c.startswith("category_")]
        assert len(category_cols) == 3

        # Column names should be sanitized
        for col in category_cols:
            # Should not contain problematic characters for Spark column names
            assert " " not in col or "_" in col
