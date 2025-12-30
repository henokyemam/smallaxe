"""Encoder for categorical feature encoding in PySpark DataFrames."""

from typing import Dict, List

from pyspark.ml.feature import OneHotEncoder as SparkOneHotEncoder
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType

from smallaxe.exceptions import (
    ColumnNotFoundError,
    ModelNotFittedError,
    ValidationError,
)

VALID_METHODS = {"onehot", "label"}
VALID_HANDLE_RARE = {"other", "keep", "error"}


class Encoder:
    """Encode categorical features using various encoding methods.

    Supports OneHot encoding (creates binary columns for each category) and
    Label encoding (converts categories to numeric indices).

    Args:
        method: Encoding method. One of 'onehot', 'label'. Default is 'onehot'.
        max_categories: Maximum number of categories to keep per column.
            Categories beyond this limit are grouped based on handle_rare.
            Default is 100.
        handle_rare: How to handle rare/excess categories. One of:
            - 'other': Group rare categories into an 'OTHER' category.
            - 'keep': Keep all categories (ignore max_categories).
            - 'error': Raise error if categories exceed max_categories.
            Default is 'other'.

    Example:
        >>> encoder = Encoder(method='onehot', max_categories=50)
        >>> encoder.fit(df, categorical_cols=['category', 'color'])
        >>> encoded_df = encoder.transform(df)
    """

    def __init__(
        self,
        method: str = "onehot",
        max_categories: int = 100,
        handle_rare: str = "other",
    ):
        self._validate_method(method)
        self._validate_handle_rare(handle_rare)
        self._validate_max_categories(max_categories)

        self._method = method
        self._max_categories = max_categories
        self._handle_rare = handle_rare
        self._categorical_cols: List[str] = []
        self._category_mappings: Dict[str, Dict[str, int]] = {}
        self._indexer_models: Dict[str, any] = {}
        self._onehot_models: Dict[str, any] = {}
        self._is_fitted = False

    def _validate_method(self, method: str) -> None:
        """Validate the encoding method."""
        if method not in VALID_METHODS:
            raise ValidationError(
                f"Invalid method '{method}'. "
                f"Must be one of {sorted(VALID_METHODS)}."
            )

    def _validate_handle_rare(self, handle_rare: str) -> None:
        """Validate the handle_rare option."""
        if handle_rare not in VALID_HANDLE_RARE:
            raise ValidationError(
                f"Invalid handle_rare '{handle_rare}'. "
                f"Must be one of {sorted(VALID_HANDLE_RARE)}."
            )

    def _validate_max_categories(self, max_categories: int) -> None:
        """Validate the max_categories parameter."""
        if not isinstance(max_categories, int) or max_categories < 1:
            raise ValidationError(
                f"max_categories must be a positive integer, got {max_categories}."
            )

    def _validate_columns(
        self,
        df: DataFrame,
        categorical_cols: List[str],
    ) -> None:
        """Validate that all specified columns exist in the DataFrame."""
        available_columns = df.columns
        for col in categorical_cols:
            if col not in available_columns:
                raise ColumnNotFoundError(column=col, available_columns=available_columns)

    def _get_top_categories(
        self,
        df: DataFrame,
        col: str,
    ) -> List[str]:
        """Get the top N categories by frequency for a column."""
        # Count categories and get top max_categories
        category_counts = (
            df.filter(F.col(col).isNotNull())
            .groupBy(col)
            .count()
            .orderBy(F.col("count").desc())
            .collect()
        )

        categories = [row[col] for row in category_counts]
        return categories

    def _build_category_mapping(
        self,
        df: DataFrame,
        col: str,
    ) -> Dict[str, int]:
        """Build category to index mapping for a column."""
        categories = self._get_top_categories(df, col)
        num_categories = len(categories)

        if self._handle_rare == "error" and num_categories > self._max_categories:
            raise ValidationError(
                f"Column '{col}' has {num_categories} categories, "
                f"which exceeds max_categories={self._max_categories}. "
                f"Set handle_rare='other' to group rare categories, or increase max_categories."
            )

        if self._handle_rare == "keep":
            # Keep all categories
            mapping = {str(cat): idx for idx, cat in enumerate(categories)}
        else:
            # Use 'other' strategy - limit to max_categories
            top_categories = categories[: self._max_categories]
            mapping = {str(cat): idx for idx, cat in enumerate(top_categories)}

            # Add OTHER category for rare values if there are more categories
            if num_categories > self._max_categories:
                mapping["__OTHER__"] = len(top_categories)

        return mapping

    def fit(
        self,
        df: DataFrame,
        categorical_cols: List[str],
    ) -> "Encoder":
        """Fit the encoder by learning category mappings from the DataFrame.

        Args:
            df: PySpark DataFrame to fit on.
            categorical_cols: List of categorical column names to encode.

        Returns:
            self: The fitted Encoder instance.

        Raises:
            ValidationError: If categorical_cols is empty or max_categories exceeded with handle_rare='error'.
            ColumnNotFoundError: If any specified column is not in the DataFrame.
        """
        if not categorical_cols:
            raise ValidationError("categorical_cols cannot be empty for Encoder.")

        self._validate_columns(df, categorical_cols)
        self._categorical_cols = categorical_cols
        self._category_mappings = {}
        self._indexer_models = {}
        self._onehot_models = {}

        # Build category mappings for each column
        for col in categorical_cols:
            self._category_mappings[col] = self._build_category_mapping(df, col)

        # For label encoding, we're done - we use the mappings directly
        # For onehot encoding, we also fit the OneHotEncoder models
        if self._method == "onehot":
            self._fit_onehot_encoders(df)

        self._is_fitted = True
        return self

    def _fit_onehot_encoders(self, df: DataFrame) -> None:
        """Fit OneHotEncoder models for each categorical column."""
        # First apply the category mappings to get indexed columns
        indexed_df = self._apply_label_encoding(df)

        for col in self._categorical_cols:
            indexed_col = f"__{col}_indexed"
            onehot_col = f"__{col}_onehot"

            # Create and fit OneHotEncoder
            encoder = SparkOneHotEncoder(
                inputCol=indexed_col,
                outputCol=onehot_col,
                dropLast=False,
            )
            self._onehot_models[col] = encoder.fit(indexed_df)

    def _apply_label_encoding(self, df: DataFrame) -> DataFrame:
        """Apply label encoding using the fitted category mappings."""
        result_df = df

        for col in self._categorical_cols:
            mapping = self._category_mappings[col]
            indexed_col = f"__{col}_indexed"

            # Broadcast the mapping for efficiency
            spark = df.sparkSession
            broadcast_mapping = spark.sparkContext.broadcast(mapping)

            # Check if we have an OTHER category
            has_other = "__OTHER__" in mapping
            other_idx = mapping.get("__OTHER__", -1)

            # Create a UDF to map categories to indices
            def create_map_udf(bc_mapping, has_other_flag, other_index):
                def map_category(value):
                    if value is None:
                        return None
                    m = bc_mapping.value
                    str_val = str(value)
                    if str_val in m:
                        return float(m[str_val])
                    elif has_other_flag:
                        return float(other_index)
                    else:
                        # Handle unseen categories - map to OTHER if available, else None
                        return None

                return F.udf(map_category, DoubleType())

            map_udf = create_map_udf(broadcast_mapping, has_other, other_idx)
            result_df = result_df.withColumn(indexed_col, map_udf(F.col(col)))

        return result_df

    def transform(self, df: DataFrame) -> DataFrame:
        """Transform the DataFrame by encoding categorical columns.

        Args:
            df: PySpark DataFrame to transform.

        Returns:
            DataFrame with encoded categorical columns.
            - For 'label' method: Original columns replaced with integer indices.
            - For 'onehot' method: Original columns replaced with binary columns
              named '{col}_{category}'.

        Raises:
            ModelNotFittedError: If transform is called before fit.
            ColumnNotFoundError: If any fitted column is not in the DataFrame.
        """
        if not self._is_fitted:
            raise ModelNotFittedError(
                "Encoder has not been fitted. Call fit() before transform()."
            )

        # Validate that all fitted columns exist
        self._validate_columns(df, self._categorical_cols)

        if self._method == "label":
            return self._transform_label(df)
        else:
            return self._transform_onehot(df)

    def _transform_label(self, df: DataFrame) -> DataFrame:
        """Transform using label encoding."""
        result_df = self._apply_label_encoding(df)

        # Replace original columns with indexed columns
        for col in self._categorical_cols:
            indexed_col = f"__{col}_indexed"
            result_df = result_df.withColumn(col, F.col(indexed_col).cast(IntegerType()))
            result_df = result_df.drop(indexed_col)

        return result_df

    def _transform_onehot(self, df: DataFrame) -> DataFrame:
        """Transform using one-hot encoding."""
        # First apply label encoding
        result_df = self._apply_label_encoding(df)

        # Apply one-hot encoding for each column
        for col in self._categorical_cols:
            # Apply the fitted OneHotEncoder
            result_df = self._onehot_models[col].transform(result_df)

        # Extract one-hot encoded values to individual columns
        result_df = self._extract_onehot_columns(result_df)

        return result_df

    def _extract_onehot_columns(self, df: DataFrame) -> DataFrame:
        """Extract one-hot encoded vectors to individual columns."""
        result_df = df

        for col in self._categorical_cols:
            mapping = self._category_mappings[col]
            onehot_col = f"__{col}_onehot"
            indexed_col = f"__{col}_indexed"

            # Create a reverse mapping (index -> category name)
            reverse_mapping = {v: k for k, v in mapping.items()}

            # Extract each position from the one-hot vector as a separate column
            num_categories = len(mapping)

            # Track used column names to avoid collisions
            used_col_names = set()

            for idx in range(num_categories):
                category_name = reverse_mapping.get(idx, f"unknown_{idx}")
                # Clean up category name for column naming
                if category_name == "__OTHER__":
                    new_col_name = f"{col}_OTHER"
                else:
                    # Sanitize category name for column naming
                    safe_name = str(category_name).replace(" ", "_").replace("-", "_")
                    new_col_name = f"{col}_{safe_name}"

                    # Handle potential collisions by appending index
                    if new_col_name in used_col_names:
                        new_col_name = f"{new_col_name}_{idx}"

                used_col_names.add(new_col_name)

                # Extract the idx-th element from the one-hot vector
                extract_element = F.udf(
                    lambda v, i=idx: float(v[i]) if v is not None and len(v) > i else 0.0,
                    DoubleType(),
                )
                result_df = result_df.withColumn(new_col_name, extract_element(F.col(onehot_col)))

            # Drop the original column, indexed column, and onehot vector column
            result_df = result_df.drop(col, indexed_col, onehot_col)

        return result_df

    def fit_transform(
        self,
        df: DataFrame,
        categorical_cols: List[str],
    ) -> DataFrame:
        """Fit and transform in a single step.

        Args:
            df: PySpark DataFrame to fit and transform.
            categorical_cols: List of categorical column names to encode.

        Returns:
            DataFrame with encoded categorical columns.
        """
        self.fit(df, categorical_cols)
        return self.transform(df)

    @property
    def method(self) -> str:
        """Get the encoding method.

        Returns:
            The encoding method ('onehot' or 'label').
        """
        return self._method

    @property
    def max_categories(self) -> int:
        """Get the max_categories setting.

        Returns:
            The maximum number of categories per column.
        """
        return self._max_categories

    @property
    def handle_rare(self) -> str:
        """Get the handle_rare setting.

        Returns:
            The handle_rare strategy ('other', 'keep', or 'error').
        """
        return self._handle_rare

    @property
    def categorical_cols(self) -> List[str]:
        """Get the fitted categorical columns.

        Returns:
            List of column names that were fitted.

        Raises:
            ModelNotFittedError: If accessed before fit.
        """
        if not self._is_fitted:
            raise ModelNotFittedError(
                "Encoder has not been fitted. Call fit() to access categorical_cols."
            )
        return self._categorical_cols.copy()

    @property
    def category_mappings(self) -> Dict[str, Dict[str, int]]:
        """Get the category mappings for each column.

        Returns:
            Dictionary mapping column names to their category-to-index mappings.

        Raises:
            ModelNotFittedError: If accessed before fit.
        """
        if not self._is_fitted:
            raise ModelNotFittedError(
                "Encoder has not been fitted. Call fit() to access category_mappings."
            )
        return {col: mapping.copy() for col, mapping in self._category_mappings.items()}
