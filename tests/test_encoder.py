"""Tests for the Encoder preprocessing component."""

import pickle

import pytest
from pyspark.sql import functions as F

from smallaxe.exceptions import (
    ColumnNotFoundError,
    ModelNotFittedError,
    ValidationError,
)
from smallaxe.preprocessing import Encoder


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
        result = encoder.fit_transform(df_for_encoding, categorical_cols=["color", "size"])

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

        # color_red, color_blue, color_green, plus the bucket for unseen values
        color_cols = [c for c in result.columns if c.startswith("color_")]
        assert sorted(color_cols) == [
            "color_UNKNOWN",
            "color_blue",
            "color_green",
            "color_red",
        ]

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
        result = encoder.fit_transform(df_for_encoding, categorical_cols=["color", "size"])

        # Original columns should be removed
        assert "color" not in result.columns
        assert "size" not in result.columns

        # Should have one-hot columns for both
        color_cols = [c for c in result.columns if c.startswith("color_")]
        size_cols = [c for c in result.columns if c.startswith("size_")]

        assert len(color_cols) == 4  # red, blue, green, UNKNOWN
        assert len(size_cols) == 4  # small, medium, large, UNKNOWN


class TestMaxCategories:
    """Tests for max_categories limiting output dimensions."""

    def test_max_categories_limits_onehot_columns(self, df_with_many_categories):
        """Test that max_categories limits the number of one-hot columns."""
        encoder = Encoder(method="onehot", max_categories=5, handle_rare="other")
        result = encoder.fit_transform(df_with_many_categories, categorical_cols=["category"])

        # Should have at most max_categories + 1 (for OTHER) columns
        category_cols = [c for c in result.columns if c.startswith("category_")]
        assert len(category_cols) <= 6  # 5 categories + 1 OTHER

    def test_max_categories_limits_label_indices(self, df_with_many_categories):
        """Test that max_categories limits the number of label indices."""
        encoder = Encoder(method="label", max_categories=5, handle_rare="other")
        result = encoder.fit_transform(df_with_many_categories, categorical_cols=["category"])

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
        result = encoder.fit_transform(df_with_rare_categories, categorical_cols=["category"])

        category_cols = [c for c in result.columns if c.startswith("category_")]

        # Should have columns for top 2 categories + OTHER
        assert len(category_cols) == 3

        # Should have an OTHER column
        other_cols = [c for c in category_cols if "OTHER" in c]
        assert len(other_cols) == 1

    def test_handle_rare_keep_preserves_all(self, df_with_rare_categories):
        """Test that handle_rare='keep' preserves all categories."""
        encoder = Encoder(method="onehot", max_categories=2, handle_rare="keep")
        result = encoder.fit_transform(df_with_rare_categories, categorical_cols=["category"])

        category_cols = [c for c in result.columns if c.startswith("category_")]

        # All 4 categories (cat_A, cat_B, cat_C, cat_D), plus the UNKNOWN bucket
        assert len(category_cols) == 5
        assert "category_UNKNOWN" in category_cols

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

    # A appears 3 times, B twice, C once.
    FIT = [(1, "A"), (2, "B"), (3, "C"), (4, "A"), (5, "B"), (6, "A")]
    # id 2 is unseen, id 3 is null.
    PROBE = [(1, "A"), (2, "Z"), (3, None), (4, "B"), (5, "C")]

    # (handle_rare, max_categories, bucket the unseen value lands in)
    SETTINGS = [
        ("other", 100, "__UNKNOWN__"),
        ("other", 2, "__OTHER__"),
        ("keep", 2, "__UNKNOWN__"),
        ("error", 5, "__UNKNOWN__"),
    ]

    def _fit(self, spark_session, method, handle_rare, max_categories):
        df_fit = spark_session.createDataFrame(self.FIT, "id int, category string")
        df_probe = spark_session.createDataFrame(self.PROBE, "id int, category string")
        encoder = Encoder(method=method, max_categories=max_categories, handle_rare=handle_rare)
        encoder.fit(df_fit, categorical_cols=["category"])
        return encoder, df_probe

    @pytest.mark.parametrize("handle_rare,max_categories,bucket", SETTINGS)
    def test_label_unseen_maps_to_bucket_and_null_stays_null(
        self, spark_session, handle_rare, max_categories, bucket
    ):
        encoder, df_probe = self._fit(spark_session, "label", handle_rare, max_categories)
        mapping = encoder.category_mappings["category"]
        assert bucket in mapping
        assert mapping[bucket] == len(mapping) - 1  # the bucket is the last index

        result = encoder.transform(df_probe)

        assert result.count() == len(self.PROBE)
        rows = {row["id"]: row["category"] for row in result.collect()}
        assert rows[2] == mapping[bucket]  # unseen: never null
        assert rows[3] is None  # null input stays null
        assert rows[1] == mapping["A"]

    @pytest.mark.parametrize("handle_rare,max_categories,bucket", SETTINGS)
    def test_onehot_unseen_sets_bucket_and_null_is_all_zero(
        self, spark_session, handle_rare, max_categories, bucket
    ):
        encoder, df_probe = self._fit(spark_session, "onehot", handle_rare, max_categories)
        bucket_col = "category_OTHER" if bucket == "__OTHER__" else "category_UNKNOWN"

        result = encoder.transform(df_probe)

        assert result.count() == len(self.PROBE)
        category_cols = [c for c in result.columns if c.startswith("category_")]
        assert category_cols[-1] == bucket_col
        rows = {row["id"]: row for row in result.collect()}
        unseen = rows[2]
        assert unseen[bucket_col] == 1.0
        assert sum(unseen[c] for c in category_cols) == 1.0
        null_row = rows[3]
        assert all(null_row[c] == 0.0 for c in category_cols)
        assert rows[1]["category_A"] == 1.0

    def test_onehot_has_no_unknown_column_when_other_exists(self, spark_session):
        encoder, df_probe = self._fit(spark_session, "onehot", "other", 1)
        result = encoder.transform(df_probe)

        assert "category_UNKNOWN" not in result.columns
        assert "__UNKNOWN__" not in encoder.category_mappings["category"]
        rows = {row["id"]: row for row in result.collect()}
        # B and C were grouped into OTHER at fit; Z was never seen. All three land there.
        for row_id in (2, 4, 5):
            assert rows[row_id]["category_OTHER"] == 1.0
            assert rows[row_id]["category_A"] == 0.0

    def test_unseen_in_several_columns(self, spark_session):
        df_fit = spark_session.createDataFrame(
            [(1, "a", "x"), (2, "b", "y")], "id int, c1 string, c2 string"
        )
        df_probe = spark_session.createDataFrame(
            [(1, "new", "x"), (2, "b", "new")], "id int, c1 string, c2 string"
        )
        encoder = Encoder(method="label").fit(df_fit, categorical_cols=["c1", "c2"])
        rows = {row["id"]: row for row in encoder.transform(df_probe).collect()}

        unknown_c1 = encoder.category_mappings["c1"]["__UNKNOWN__"]
        unknown_c2 = encoder.category_mappings["c2"]["__UNKNOWN__"]
        assert rows[1]["c1"] == unknown_c1
        assert rows[2]["c2"] == unknown_c2
        assert rows[2]["c1"] == encoder.category_mappings["c1"]["b"]

    def test_reserved_value_in_data_raises_at_fit(self, spark_session):
        df = spark_session.createDataFrame([(1, "__UNKNOWN__"), (2, "a")], "id int, c string")
        with pytest.raises(ValidationError, match="reserves"):
            Encoder().fit(df, categorical_cols=["c"])


class TestNonStringCategories:
    """Category keys come from Spark's cast to string, so every type matches itself."""

    @pytest.fixture
    def df_typed(self, spark_session):
        rows = [
            (1, True, 1.5),
            (2, False, 2.0),
            (3, True, 1.5),
            (4, None, None),
            (5, True, 2.5e7),  # Spark: "2.5E7"; Python str(): "25000000.0"
        ]
        return spark_session.createDataFrame(rows, "id int, flag boolean, score double")

    def test_mapping_keys_are_spark_strings(self, df_typed):
        encoder = Encoder(method="label").fit(df_typed, categorical_cols=["flag", "score"])
        mappings = encoder.category_mappings

        assert mappings["flag"] == {"true": 0, "false": 1, "__UNKNOWN__": 2}
        assert mappings["score"] == {"1.5": 0, "2.0": 1, "2.5E7": 2, "__UNKNOWN__": 3}

    def test_label_matches_every_training_category(self, df_typed):
        encoder = Encoder(method="label")
        result = encoder.fit_transform(df_typed, categorical_cols=["flag", "score"])
        rows = {row["id"]: row for row in result.collect()}
        mappings = encoder.category_mappings

        assert rows[1]["flag"] == mappings["flag"]["true"]
        assert rows[2]["flag"] == mappings["flag"]["false"]
        assert rows[2]["score"] == mappings["score"]["2.0"]
        assert rows[5]["score"] == mappings["score"]["2.5E7"]
        assert rows[4]["flag"] is None
        assert rows[4]["score"] is None
        for col in ("flag", "score"):
            unknown = mappings[col]["__UNKNOWN__"]
            assert all(row[col] != unknown for row in rows.values())

    def test_onehot_matches_every_training_category(self, df_typed):
        encoder = Encoder(method="onehot")
        result = encoder.fit_transform(df_typed, categorical_cols=["flag", "score"])

        assert "flag_true" in result.columns
        assert "flag_false" in result.columns
        assert "score_1.5" in result.columns
        totals = result.agg(
            F.sum("flag_UNKNOWN").alias("flag_unknown"),
            F.sum("score_UNKNOWN").alias("score_unknown"),
        ).first()
        assert totals["flag_unknown"] == 0.0
        assert totals["score_unknown"] == 0.0
        rows = {row["id"]: row for row in result.collect()}
        assert rows[1]["flag_true"] == 1.0
        assert rows[2]["flag_false"] == 1.0
        assert rows[5]["score_2.5E7"] == 1.0

    def test_integer_categories(self, spark_session):
        df = spark_session.createDataFrame([(1, 10), (2, 20), (3, 10)], "id int, code int")
        encoder = Encoder(method="label")
        result = encoder.fit_transform(df, categorical_cols=["code"])

        assert encoder.category_mappings["code"] == {"10": 0, "20": 1, "__UNKNOWN__": 2}
        assert [row["code"] for row in result.orderBy("id").collect()] == [0, 1, 0]


class TestDeterministicOrder:
    """Ties are broken by the category string, so partitioning does not matter."""

    ROWS = [(i, cat) for i, cat in enumerate(["d", "b", "c", "a", "b", "d", "a", "c", "e"])]

    def test_ties_are_ordered_by_category(self, spark_session):
        df = spark_session.createDataFrame(self.ROWS, "id int, c string")
        mapping = Encoder(method="label").fit(df, categorical_cols=["c"]).category_mappings["c"]

        # a, b, c, d appear twice each; e once.
        assert mapping == {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "__UNKNOWN__": 5}

    @pytest.mark.parametrize("partitions", [1, 3, 7])
    def test_same_mapping_under_repartitioning(self, spark_session, partitions):
        df = spark_session.createDataFrame(self.ROWS, "id int, c string")
        reference = Encoder().fit(df, categorical_cols=["c"]).category_mappings

        shuffled = spark_session.createDataFrame(list(reversed(self.ROWS)), "id int, c string")
        shuffled = shuffled.repartition(partitions)
        assert Encoder().fit(shuffled, categorical_cols=["c"]).category_mappings == reference

    def test_rare_grouping_is_deterministic(self, spark_session):
        df = spark_session.createDataFrame(self.ROWS, "id int, c string")
        mapping = (
            Encoder(max_categories=2, handle_rare="other")
            .fit(df.repartition(5), categorical_cols=["c"])
            .category_mappings["c"]
        )
        assert mapping == {"a": 0, "b": 1, "__OTHER__": 2}


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
        df_fit = spark_session.createDataFrame([(1, "A"), (2, "B")], ["id", "category"])
        encoder = Encoder()
        encoder.fit(df_fit, categorical_cols=["category"])

        # Transform on a different DataFrame missing the column
        df_transform = spark_session.createDataFrame([(1, 100), (2, 200)], ["id", "value"])
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
        # red, blue, green, and the reserved last index for unseen values
        assert len(mappings["color"]) == 4
        assert mappings["color"]["__UNKNOWN__"] == 3

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

        # One column for the single category, plus the UNKNOWN bucket
        category_cols = [c for c in result.columns if c.startswith("category_")]
        assert category_cols == ["category_only_one", "category_UNKNOWN"]

        rows = result.collect()
        assert all(row["category_only_one"] == 1.0 for row in rows)
        assert all(row["category_UNKNOWN"] == 0.0 for row in rows)

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

        # Null input stays null: missing values are the Imputer's and the model's business
        assert rows[1]["category"] is None
        assert rows[3]["category"] is None
        assert "None" not in encoder.category_mappings["category"]

    def test_null_values_onehot_all_zero(self, spark_session):
        """A null yields 0.0 in every one-hot column of the feature, UNKNOWN included."""
        df = spark_session.createDataFrame(
            [(1, "A"), (2, None), (3, "B")], "id int, category string"
        )
        result = Encoder(method="onehot").fit_transform(df, categorical_cols=["category"])

        null_row = result.filter(F.col("id") == 2).first()
        category_cols = [c for c in result.columns if c.startswith("category_")]
        assert category_cols == ["category_A", "category_B", "category_UNKNOWN"]
        assert [null_row[c] for c in category_cols] == [0.0, 0.0, 0.0]

    def test_all_null_label_encoding(self, spark_session):
        """Test label encoding when a fitted categorical column has no categories."""
        df = spark_session.createDataFrame(
            [(1, None), (2, None)],
            "id int, category string",
        )

        encoder = Encoder(method="label")
        result = encoder.fit_transform(df, categorical_cols=["category"])

        rows = result.orderBy("id").collect()
        assert [row["category"] for row in rows] == [None, None]
        assert encoder.category_mappings["category"] == {"__UNKNOWN__": 0}

        # Any non-null value at transform is unseen and maps to UNKNOWN
        probe = df.sparkSession.createDataFrame([(1, "x")], "id int, category string")
        assert encoder.transform(probe).first()["category"] == 0

    def test_all_null_onehot_encoding(self, spark_session):
        """Test one-hot encoding when a fitted categorical column has no categories."""
        df = spark_session.createDataFrame(
            [(1, None), (2, None)],
            "id int, category string",
        )

        encoder = Encoder(method="onehot")
        result = encoder.fit_transform(df, categorical_cols=["category"])

        assert "category" not in result.columns
        assert [col for col in result.columns if col.startswith("category_")] == [
            "category_UNKNOWN"
        ]
        assert [row["category_UNKNOWN"] for row in result.collect()] == [0.0, 0.0]
        assert result.count() == 2

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
        # Ties are ordered by the category string: " " < "-" < "_". The second and third
        # sanitise to a name already used, so they get their index appended.
        assert category_cols == [
            "category_hello_world",
            "category_hello_world_1",
            "category_hello_world_2",
            "category_UNKNOWN",
        ]

        # Column names should be sanitized
        for col in category_cols:
            assert " " not in col and "-" not in col

    def test_category_named_like_a_bucket_keeps_the_bucket_name(self, spark_session):
        """A real category 'UNKNOWN' does not take over the {col}_UNKNOWN column."""
        df = spark_session.createDataFrame(
            [(1, "UNKNOWN"), (2, "UNKNOWN"), (3, "a")], "id int, category string"
        )
        encoder = Encoder(method="onehot")
        result = encoder.fit_transform(df, categorical_cols=["category"])

        category_cols = [c for c in result.columns if c.startswith("category_")]
        assert category_cols == ["category_UNKNOWN_0", "category_a", "category_UNKNOWN"]
        rows = {row["id"]: row for row in result.collect()}
        assert rows[1]["category_UNKNOWN_0"] == 1.0
        assert rows[1]["category_UNKNOWN"] == 0.0

    def test_categories_differing_only_in_case_get_distinct_columns(self, spark_session):
        """Spark resolves names case-insensitively, so 'A' and 'a' need distinct names."""
        df = spark_session.createDataFrame(
            [(1, "A"), (2, "A"), (3, "a")], "id int, category string"
        )
        result = Encoder(method="onehot").fit_transform(df, categorical_cols=["category"])

        category_cols = [c for c in result.columns if c.startswith("category_")]
        assert category_cols == ["category_A", "category_a_1", "category_UNKNOWN"]
        rows = {row["id"]: row for row in result.collect()}
        assert rows[3]["category_a_1"] == 1.0
        assert rows[3]["category_A"] == 0.0


class TestColumnOrder:
    """The transform's column order is deterministic (see the Encoder docstring)."""

    @pytest.fixture
    def df_mixed(self, spark_session):
        return spark_session.createDataFrame(
            [(1, "red", 1.0, "s"), (2, "blue", 2.0, "m"), (3, "red", 3.0, "s")],
            "id int, color string, x double, size string",
        )

    def test_label_keeps_input_order(self, df_mixed):
        result = Encoder(method="label").fit_transform(df_mixed, categorical_cols=["size", "color"])
        assert result.columns == ["id", "color", "x", "size"]
        assert dict(result.dtypes)["color"] == "int"

    def test_onehot_appends_blocks_in_categorical_cols_order(self, df_mixed):
        result = Encoder(method="onehot").fit_transform(
            df_mixed, categorical_cols=["size", "color"]
        )
        assert result.columns == [
            "id",
            "x",
            "size_s",
            "size_m",
            "size_UNKNOWN",
            "color_red",
            "color_blue",
            "color_UNKNOWN",
        ]

    def test_user_column_named_like_the_working_index_is_untouched(self, spark_session):
        df = spark_session.createDataFrame(
            [(1, "a", 7.0), (2, "b", 8.0)], "id int, c string, `__c_index` double"
        )
        result = Encoder(method="onehot").fit_transform(df, categorical_cols=["c"])

        assert "__c_index" in result.columns
        assert [row["__c_index"] for row in result.orderBy("id").collect()] == [7.0, 8.0]

    def test_output_named_like_another_encoded_input_survives(self, spark_session):
        """c's category 'a' gives c_a, which is also an encoded input column's name."""
        df = spark_session.createDataFrame(
            [(1, "a", "x"), (2, "b", "y"), (3, None, "x")], "id int, c string, c_a string"
        )
        result = Encoder(method="onehot").fit_transform(df, categorical_cols=["c", "c_a"])

        assert result.columns == [
            "id",
            "c_a",
            "c_b",
            "c_UNKNOWN",
            "c_a_x",
            "c_a_y",
            "c_a_UNKNOWN",
        ]
        rows = result.orderBy("id").collect()
        assert [row["c_a"] for row in rows] == [1.0, 0.0, 0.0]
        assert [row["c_a_x"] for row in rows] == [1.0, 0.0, 1.0]

    def test_output_named_like_a_pass_through_column_raises(self, spark_session):
        fit = spark_session.createDataFrame([(1, "a"), (2, "b")], "id int, c string")
        encoder = Encoder(method="onehot").fit(fit, categorical_cols=["c"])
        df = spark_session.createDataFrame([(1, "a", 5.0)], "id int, c string, C_A double")

        with pytest.raises(ValidationError, match="C_A") as info:
            encoder.transform(df)
        assert "c_a" in str(info.value)


class TestEncoderHoldsPlainData:
    """A fitted Encoder holds no Spark objects, so it pickles."""

    @pytest.mark.parametrize("method", ["onehot", "label"])
    def test_pickle_round_trip(self, df_for_encoding, method):
        encoder = Encoder(method=method).fit(df_for_encoding, categorical_cols=["color", "size"])

        restored = pickle.loads(pickle.dumps(encoder))

        assert restored.category_mappings == encoder.category_mappings
        expected = encoder.transform(df_for_encoding).orderBy("id").collect()
        assert restored.transform(df_for_encoding).orderBy("id").collect() == expected

    def test_no_spark_model_attributes(self, df_for_encoding):
        encoder = Encoder().fit(df_for_encoding, categorical_cols=["color"])
        assert not hasattr(encoder, "_onehot_models")
        assert not hasattr(encoder, "_indexer_models")
        for value in vars(encoder).values():
            assert isinstance(value, (str, int, bool, list, dict))
