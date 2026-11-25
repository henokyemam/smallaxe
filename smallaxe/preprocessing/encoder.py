"""Encoder for categorical feature encoding in PySpark DataFrames."""

from typing import Any, Dict, List, Set, Tuple

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

from smallaxe import _artifact
from smallaxe.exceptions import (
    ColumnNotFoundError,
    ModelNotFittedError,
    ValidationError,
)
from smallaxe.preprocessing._common import (
    column_list,
    column_ref,
    read_step_metadata,
    require_keys,
    unique_column_name,
)

VALID_METHODS = {"onehot", "label"}
VALID_HANDLE_RARE = {"other", "keep", "error"}

OTHER = "__OTHER__"
UNKNOWN = "__UNKNOWN__"
_RESERVED = (OTHER, UNKNOWN)
_RESERVED_SUFFIX = {OTHER: "OTHER", UNKNOWN: "UNKNOWN"}

ARTIFACT_KIND = "encoder"


class Encoder:
    """Encode categorical features using various encoding methods.

    Supports OneHot encoding (creates binary columns for each category) and
    Label encoding (converts categories to numeric indices).

    Categories are the column's values cast to string by Spark (so a boolean column has
    the categories ``"true"`` and ``"false"``), ranked by count descending and then by
    the category string ascending. Their rank is their index.

    Every fitted column has an index for values not in its mapping: the ``"__OTHER__"``
    bucket when ``handle_rare='other'`` created one, otherwise a reserved
    ``"__UNKNOWN__"`` entry appended as the last index. At transform time:

    - an unseen category maps to that index (label) or sets ``{col}_OTHER`` /
      ``{col}_UNKNOWN`` to 1.0 (one-hot), whatever ``handle_rare`` is;
    - a null stays null (label) or yields 0.0 in every column of that feature (one-hot);
    - no row is dropped, and no data value raises.

    Output columns:

    - ``'label'``: the input columns in their order, each encoded column replaced in
      place by an integer index.
    - ``'onehot'``: the input's other columns in their order, then, for each column in
      ``categorical_cols`` order, one double column per index in index order, named
      ``{col}_{category}`` (spaces and hyphens become underscores; a name already
      used gets ``_{index}`` appended), and ``{col}_OTHER`` or ``{col}_UNKNOWN`` last.
      An output name may equal an encoded input column's name (that input is replaced);
      if it equals a column the Encoder does not encode, ``transform`` raises
      ``ValidationError`` rather than overwrite it.

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
        self._is_fitted = False

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore a pickled Encoder, upgrading one pickled by smallaxe 0.10.

        0.10 had no ``save()``, so ``Pipeline.save`` pickled the Encoder. Its state held
        ``_indexer_models`` / ``_onehot_models`` and mappings with no bucket for unseen
        values. Those are dropped, and ``"__UNKNOWN__"`` is appended to every mapping
        that has no ``"__OTHER__"``, so the current unseen-category rule applies.
        """
        state = dict(state)
        state.pop("_indexer_models", None)
        state.pop("_onehot_models", None)
        mappings = state.get("_category_mappings") or {}
        upgraded = {}
        for col, mapping in mappings.items():
            mapping = dict(mapping)
            if OTHER not in mapping and UNKNOWN not in mapping:
                mapping[UNKNOWN] = len(mapping)
            upgraded[col] = mapping
        state["_category_mappings"] = upgraded
        self.__dict__.update(state)

    def _validate_method(self, method: str) -> None:
        """Validate the encoding method."""
        if method not in VALID_METHODS:
            raise ValidationError(
                f"Invalid method '{method}'. Must be one of {sorted(VALID_METHODS)}."
            )

    def _validate_handle_rare(self, handle_rare: str) -> None:
        """Validate the handle_rare option."""
        if handle_rare not in VALID_HANDLE_RARE:
            raise ValidationError(
                f"Invalid handle_rare '{handle_rare}'. Must be one of "
                f"{sorted(VALID_HANDLE_RARE)}."
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
        """Return the column's categories, most frequent first.

        The key is Spark's ``cast("string")`` of the value, the same expression transform
        looks up, so booleans, doubles and dates match. Ties are broken by the key, so the
        order does not depend on partitioning.
        """
        category_counts = (
            df.select(column_ref(col).cast("string").alias("category"))
            .filter(F.col("category").isNotNull())
            .groupBy("category")
            .count()
            .orderBy(F.col("count").desc(), F.col("category").asc())
            .collect()
        )
        return [row["category"] for row in category_counts]

    def _build_category_mapping(
        self,
        df: DataFrame,
        col: str,
    ) -> Dict[str, int]:
        """Build category to index mapping for a column."""
        categories = self._get_top_categories(df, col)
        reserved = [cat for cat in categories if cat in _RESERVED]
        if reserved:
            raise ValidationError(
                f"Column '{col}' contains the value {reserved[0]!r}, which the Encoder "
                f"reserves for its {_RESERVED_SUFFIX[reserved[0]]} bucket. Rename that "
                "value before encoding."
            )
        num_categories = len(categories)

        if self._handle_rare == "error" and num_categories > self._max_categories:
            raise ValidationError(
                f"Column '{col}' has {num_categories} categories, "
                f"which exceeds max_categories={self._max_categories}. "
                f"Set handle_rare='other' to group rare categories, or increase "
                f"max_categories."
            )

        if self._handle_rare == "keep":
            kept = categories
        else:
            kept = categories[: self._max_categories]
        mapping = {cat: idx for idx, cat in enumerate(kept)}

        if len(kept) < num_categories:
            mapping[OTHER] = len(kept)
        if OTHER not in mapping:
            mapping[UNKNOWN] = len(mapping)

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
            ValidationError: If categorical_cols is empty, max_categories is exceeded
                with handle_rare='error', or a column holds a reserved value
                ('__OTHER__' or '__UNKNOWN__').
            ColumnNotFoundError: If any specified column is not in the DataFrame.
        """
        if not categorical_cols:
            raise ValidationError("categorical_cols cannot be empty for Encoder.")

        self._validate_columns(df, categorical_cols)
        mappings = {col: self._build_category_mapping(df, col) for col in categorical_cols}

        self._categorical_cols = list(categorical_cols)
        self._category_mappings = mappings
        self._is_fitted = True
        return self

    @staticmethod
    def _fallback_index(mapping: Dict[str, int]) -> int:
        """The index an unseen category maps to: OTHER if present, else UNKNOWN."""
        return mapping[OTHER] if OTHER in mapping else mapping[UNKNOWN]

    def _index_expr(self, col: str) -> Column:
        """Integer index of ``col``: null for null, the fallback index for unseen values."""
        mapping = self._category_mappings[col]
        source = column_ref(col)

        map_args = []
        for category, idx in mapping.items():
            if category in _RESERVED:
                continue
            map_args.append(F.lit(category))
            map_args.append(F.lit(int(idx)))

        if map_args:
            lookup = F.create_map(*map_args)[source.cast("string")]
        else:
            lookup = F.lit(None).cast("int")

        fallback = F.lit(int(self._fallback_index(mapping)))
        index = F.when(source.isNull(), F.lit(None)).otherwise(F.coalesce(lookup, fallback))
        return index.cast("int")

    def _onehot_layout(self) -> Dict[str, List[Tuple[str, int]]]:
        """``(output column name, index)`` per fitted column, each in index order.

        The OTHER / UNKNOWN buckets always get ``{col}_OTHER`` / ``{col}_UNKNOWN``. A
        category whose sanitised name is already taken (compared case-insensitively, as
        Spark resolves names) gets ``_{index}`` appended.
        """
        used: Set[str] = set()
        for col in self._categorical_cols:
            for key in _RESERVED:
                if key in self._category_mappings[col]:
                    used.add(f"{col}_{_RESERVED_SUFFIX[key]}".lower())

        layout: Dict[str, List[Tuple[str, int]]] = {}
        for col in self._categorical_cols:
            mapping = self._category_mappings[col]
            by_index = {idx: category for category, idx in mapping.items()}
            columns = []
            for idx in range(len(mapping)):
                category = by_index[idx]
                if category in _RESERVED:
                    name = f"{col}_{_RESERVED_SUFFIX[category]}"
                else:
                    safe_name = category.replace(" ", "_").replace("-", "_")
                    name = f"{col}_{safe_name}"
                    if name.lower() in used:
                        name = unique_column_name(f"{name}_{idx}", used)
                    used.add(name.lower())
                columns.append((name, idx))
            layout[col] = columns
        return layout

    def transform(self, df: DataFrame) -> DataFrame:
        """Transform the DataFrame by encoding categorical columns.

        Args:
            df: PySpark DataFrame to transform.

        Returns:
            DataFrame with encoded categorical columns (see the class docstring for the
            column order and the unseen / null rules).
            - For 'label' method: Original columns replaced with integer indices.
            - For 'onehot' method: Original columns replaced with binary columns
              named '{col}_{category}', plus '{col}_OTHER' or '{col}_UNKNOWN'.

        Raises:
            ModelNotFittedError: If transform is called before fit.
            ColumnNotFoundError: If any fitted column is not in the DataFrame.
            ValidationError: For 'onehot', if an output column name is already used by a
                column the Encoder does not encode (it would be overwritten).
        """
        if not self._is_fitted:
            raise ModelNotFittedError("Encoder has not been fitted. Call fit() before transform().")

        self._validate_columns(df, self._categorical_cols)

        if self._method == "label":
            return self._transform_label(df)

        return self._transform_onehot(df)

    def _transform_label(self, df: DataFrame) -> DataFrame:
        """Transform using label encoding."""
        return df.withColumns({col: self._index_expr(col) for col in self._categorical_cols})

    def _transform_onehot(self, df: DataFrame) -> DataFrame:
        """Transform using one-hot encoding built from the Encoder's own mapping."""
        layout = self._onehot_layout()
        taken = list(df.columns) + [name for columns in layout.values() for name, _ in columns]
        index_cols: Dict[str, str] = {}
        for col in self._categorical_cols:
            name = unique_column_name(f"__{col}_index", taken)
            taken.append(name)
            index_cols[col] = name

        encoded = {col.lower() for col in self._categorical_cols}
        kept = {col.lower(): col for col in df.columns if col.lower() not in encoded}
        clashes = [
            (name, kept[name.lower()])
            for columns in layout.values()
            for name, _ in columns
            if name.lower() in kept
        ]
        if clashes:
            name, existing = clashes[0]
            raise ValidationError(
                f"One-hot output column '{name}' would overwrite the existing column "
                f"'{existing}', which the Encoder does not encode. Rename or drop that "
                "column before transform."
            )

        result_df = df.withColumns(
            {index_cols[col]: self._index_expr(col) for col in self._categorical_cols}
        )
        # Drop the inputs before adding the outputs, so an output named like an encoded
        # input column (c's category 'a' gives 'c_a') is not dropped with it.
        result_df = result_df.drop(*self._categorical_cols)

        onehot: Dict[str, Column] = {}
        for col, columns in layout.items():
            index = column_ref(index_cols[col])
            for name, idx in columns:
                onehot[name] = F.coalesce((index == F.lit(idx)).cast("double"), F.lit(0.0))
        result_df = result_df.withColumns(onehot)

        return result_df.drop(*index_cols.values())

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

    def save(self, path: str) -> None:
        """Save the fitted encoder as a directory holding ``metadata.json``.

        The file records the constructor arguments, the fitted columns and their
        category mappings. IO goes through the Hadoop FileSystem layer, so local paths,
        ``dbfs:/``, ``s3://`` and ``hdfs://`` all work.

        Args:
            path: The directory path to save the encoder to.

        Raises:
            ModelNotFittedError: If save is called before fit.
            ValidationError: If path is empty.
        """
        if not self._is_fitted:
            raise ModelNotFittedError("Encoder has not been fitted. Call fit() before saving.")

        _artifact.write_metadata(
            path,
            ARTIFACT_KIND,
            {
                "method": self._method,
                "max_categories": self._max_categories,
                "handle_rare": self._handle_rare,
                "categorical_cols": list(self._categorical_cols),
                "category_mappings": self._category_mappings,
            },
        )

    @classmethod
    def load(cls, path: str) -> "Encoder":
        """Load an encoder saved by :meth:`save`.

        Args:
            path: The directory path to load the encoder from.

        Returns:
            The loaded, fitted Encoder.

        Raises:
            ValidationError: If the path does not exist, is not an encoder artifact,
                was written by a newer smallaxe, or its metadata is incomplete.
        """
        metadata = read_step_metadata(path, ARTIFACT_KIND)
        keys = ["method", "max_categories", "handle_rare", "categorical_cols"]
        require_keys(metadata, keys + ["category_mappings"], ARTIFACT_KIND, path)

        encoder = cls(
            method=metadata["method"],
            max_categories=metadata["max_categories"],
            handle_rare=metadata["handle_rare"],
        )
        categorical_cols = column_list(metadata, "categorical_cols", ARTIFACT_KIND, path)
        raw_mappings = metadata["category_mappings"]
        mappings: Dict[str, Dict[str, int]] = {}
        try:
            for col in categorical_cols:
                mapping = {str(cat): int(idx) for cat, idx in raw_mappings[col].items()}
                if sorted(mapping.values()) != list(range(len(mapping))) or not any(
                    key in mapping for key in _RESERVED
                ):
                    raise ValueError(f"invalid mapping for column '{col}'")
                mappings[col] = mapping
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise ValidationError(
                f"Cannot load {ARTIFACT_KIND}: the category mappings at {path!r} are "
                f"corrupted ({exc})."
            ) from exc

        encoder._categorical_cols = categorical_cols
        encoder._category_mappings = mappings
        encoder._is_fitted = True
        return encoder

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

        Each mapping includes the bucket unseen values go to: ``"__OTHER__"`` or
        ``"__UNKNOWN__"``.

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

    @property
    def output_cols(self) -> List[str]:
        """The columns ``transform`` writes for the encoded features, in order.

        For ``'label'`` these are the encoded input columns (replaced in place), in
        ``categorical_cols`` order. For ``'onehot'`` they are the generated columns: for
        each column in ``categorical_cols`` order, one column per index in index order,
        the ``{col}_OTHER`` / ``{col}_UNKNOWN`` bucket last (see the class docstring for
        the names). A ``Pipeline`` hands exactly these columns to its model.

        Returns:
            List of output column names.

        Raises:
            ModelNotFittedError: If accessed before fit.
        """
        if not self._is_fitted:
            raise ModelNotFittedError(
                "Encoder has not been fitted. Call fit() to access output_cols."
            )
        if self._method == "label":
            return list(self._categorical_cols)
        layout = self._onehot_layout()
        return [name for col in self._categorical_cols for name, _ in layout[col]]
