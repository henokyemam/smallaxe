"""ValidationMixin for data splitting and cross-validation.

Every split in smallaxe comes from one rule: a row's fold is a hash of the row's
content and the seed. The assignment therefore depends only on the rows and the
seed, never on partitioning, caching, or how many times Spark reads the source.
Exact duplicate rows land in the same fold.
"""

import random
from typing import Iterator, List, Optional, Tuple

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import MapType, UserDefinedType

import smallaxe
from smallaxe.exceptions import ColumnNotFoundError, ValidationError

# ``F.hash`` yields a signed 32-bit int; masking the sign bit maps it onto [0, 2**31).
_HASH_RANGE = 2**31


def hashable_columns(df: DataFrame) -> List[str]:
    """Columns of ``df`` that Spark's ``hash`` accepts (no maps or ML vectors)."""
    return [
        field.name
        for field in df.schema.fields
        if not isinstance(field.dataType, (MapType, UserDefinedType))
    ]


def split_hash(df: DataFrame, seed: int, fold_key: Optional[str] = None) -> Column:
    """A non-negative hash of each row's content (or of ``fold_key``) and ``seed``.

    Args:
        df: The DataFrame being split.
        seed: Seed mixed into the hash.
        fold_key: If given, hash only this column, so all rows sharing a key land in
            the same fold or split.

    Raises:
        ColumnNotFoundError: If ``fold_key`` is not a column of ``df``.
        ValidationError: If no column of ``df`` can be hashed.
    """
    if fold_key is not None:
        if fold_key not in df.columns:
            raise ColumnNotFoundError(column=fold_key, available_columns=list(df.columns))
        columns = [fold_key]
    else:
        columns = hashable_columns(df)
        if not columns:
            raise ValidationError(
                "No column can be hashed to assign folds: every column is a map or an ML "
                "vector. Pass fold_key= naming a column to hash."
            )
    return F.hash(*[F.col(c) for c in columns], F.lit(seed)).bitwiseAND(F.lit(_HASH_RANGE - 1))


def resolve_split_seed(seed: Optional[int] = None) -> int:
    """The seed splits use: the argument, else the global seed, else a fresh random one."""
    if seed is not None:
        return seed
    global_seed = smallaxe.get_seed()
    if global_seed is not None:
        return global_seed
    return random.randrange(_HASH_RANGE)


class ValidationMixin:
    """Mixin providing data splitting and cross-validation functionality.

    Splits and folds are assigned by :func:`split_hash`. Stratification is implicit:
    a content hash is independent of the label, so every class is split in the same
    proportion in expectation. The ``stratified`` arguments are accepted for API
    compatibility and validated, but select no different rule.
    """

    #: The seed the most recent split or fold assignment used.
    _split_seed: Optional[int] = None

    def _train_test_split(
        self,
        df: DataFrame,
        test_size: float = 0.2,
        stratified: bool = False,
        label_col: Optional[str] = None,
        seed: Optional[int] = None,
        fold_key: Optional[str] = None,
    ) -> Tuple[DataFrame, DataFrame]:
        """Split DataFrame into training and test sets.

        Args:
            df: PySpark DataFrame to split.
            test_size: Proportion of data for the test set. Default is 0.2.
            stratified: Accepted for compatibility; see the class docstring.
                Requires label_col to be specified.
            label_col: Column name for stratification. Required if stratified=True.
            seed: Seed for the assignment. Defaults to the global seed, else random.
            fold_key: Column to hash instead of the whole row.

        Returns:
            Tuple of (train_df, test_df).

        Raises:
            ValidationError: If test_size is not between 0 and 1.
            ValidationError: If stratified=True but label_col is not specified.
        """
        if not 0 < test_size < 1:
            raise ValidationError(f"test_size must be between 0 and 1, got {test_size}.")

        if stratified and not label_col:
            raise ValidationError("label_col must be specified when stratified=True.")

        self._split_seed = resolve_split_seed(seed)
        bucket = split_hash(df, self._split_seed, fold_key) / F.lit(float(_HASH_RANGE))
        is_test = bucket < F.lit(test_size)
        return df.filter(~is_test), df.filter(is_test)

    def _stratified_train_test_split(
        self,
        df: DataFrame,
        test_size: float,
        label_col: str,
        seed: Optional[int] = None,
    ) -> Tuple[DataFrame, DataFrame]:
        """Same rule as :meth:`_train_test_split`; kept for compatibility."""
        return self._train_test_split(df, test_size, True, label_col, seed)

    def _kfold_split(
        self,
        df: DataFrame,
        n_folds: int = 5,
        stratified: bool = False,
        label_col: Optional[str] = None,
        seed: Optional[int] = None,
        fold_key: Optional[str] = None,
    ) -> Iterator[Tuple[DataFrame, DataFrame]]:
        """Generate k-fold cross-validation splits.

        Args:
            df: PySpark DataFrame to split.
            n_folds: Number of folds. Default is 5.
            stratified: Accepted for compatibility; see the class docstring.
                Requires label_col to be specified.
            label_col: Column name for stratification. Required if stratified=True.
            seed: Seed for the assignment. Defaults to the global seed, else random.
            fold_key: Column to hash instead of the whole row.

        Yields:
            Tuple of (train_df, val_df) for each fold.

        Raises:
            ValidationError: If n_folds < 2.
            ValidationError: If stratified=True but label_col is not specified.
        """
        if n_folds < 2:
            raise ValidationError(f"n_folds must be at least 2, got {n_folds}.")

        if stratified and not label_col:
            raise ValidationError("label_col must be specified when stratified=True.")

        self._split_seed = resolve_split_seed(seed)
        fold = split_hash(df, self._split_seed, fold_key) % n_folds

        for fold_idx in range(n_folds):
            yield df.filter(fold != fold_idx), df.filter(fold == fold_idx)

    def _simple_kfold_split(
        self,
        df: DataFrame,
        n_folds: int,
        seed: Optional[int] = None,
    ) -> Iterator[Tuple[DataFrame, DataFrame]]:
        """Same rule as :meth:`_kfold_split`; kept for compatibility."""
        yield from self._kfold_split(df, n_folds, False, None, seed)

    def _stratified_kfold_split(
        self,
        df: DataFrame,
        n_folds: int,
        label_col: str,
        seed: Optional[int] = None,
    ) -> Iterator[Tuple[DataFrame, DataFrame]]:
        """Same rule as :meth:`_kfold_split`; kept for compatibility."""
        yield from self._kfold_split(df, n_folds, True, label_col, seed)
