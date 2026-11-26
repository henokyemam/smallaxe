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

__all__ = ["ValidationMixin"]

# ``F.hash`` yields a signed 32-bit int; masking the sign bit maps it onto [0, 2**31).
_HASH_RANGE = 2**31


def _hashable_columns(dataframe: DataFrame) -> List[str]:
    """Columns of ``dataframe`` that Spark's ``hash`` accepts (no maps or ML vectors)."""
    return [
        field.name
        for field in dataframe.schema.fields
        if not isinstance(field.dataType, (MapType, UserDefinedType))
    ]


def _split_hash(dataframe: DataFrame, seed: int, fold_key: Optional[str] = None) -> Column:
    """A non-negative hash of each row's content (or of ``fold_key``) and ``seed``.

    Args:
        dataframe: The DataFrame being split.
        seed: Seed mixed into the hash.
        fold_key: If given, hash only this column, so all rows sharing a key land in
            the same fold or split.

    Raises:
        ColumnNotFoundError: If ``fold_key`` is not a column of ``dataframe``.
        ValidationError: If no column of ``dataframe`` can be hashed.
    """
    if fold_key is not None:
        if fold_key not in dataframe.columns:
            raise ColumnNotFoundError(column=fold_key, available_columns=list(dataframe.columns))
        columns = [fold_key]
    else:
        columns = _hashable_columns(dataframe)
        if not columns:
            raise ValidationError(
                "No column can be hashed to assign folds: every column is a map or an ML "
                "vector. Pass fold_key= naming a column to hash."
            )
    return F.hash(*[F.col(c) for c in columns], F.lit(seed)).bitwiseAND(F.lit(_HASH_RANGE - 1))


def _resolve_split_seed(seed: Optional[int] = None, fallback: Optional[int] = None) -> int:
    """The seed splits use: ``seed``, else the global seed, else ``fallback``, else random.

    ``fallback`` lets a caller that fits many models (``optimize.run``) give them all one
    split when nothing else seeds it.
    """
    if seed is not None:
        return seed
    global_seed = smallaxe.get_seed()
    if global_seed is not None:
        return global_seed
    if fallback is not None:
        return fallback
    return random.randrange(_HASH_RANGE)


class ValidationMixin:
    """Mixin providing data splitting and cross-validation functionality.

    Splits and folds are assigned by :func:`_split_hash`. A content hash is independent
    of the label, so every class is split in the same proportion in expectation.
    """

    #: The seed the most recent split or fold assignment used.
    _split_seed: Optional[int] = None

    def _train_test_split(
        self,
        dataframe: DataFrame,
        test_size: float = 0.2,
        seed: Optional[int] = None,
        fold_key: Optional[str] = None,
    ) -> Tuple[DataFrame, DataFrame]:
        """Split DataFrame into training and test sets.

        Args:
            dataframe: PySpark DataFrame to split.
            test_size: Proportion of data for the test set. Default is 0.2.
            seed: Seed for the assignment. Defaults to the global seed, else random.
            fold_key: Column to hash instead of the whole row.

        Returns:
            Tuple of (train_df, test_df).

        Raises:
            ValidationError: If test_size is not a number strictly between 0 and 1.
        """
        if (
            isinstance(test_size, bool)
            or not isinstance(test_size, (int, float))
            or not 0 < test_size < 1
        ):
            raise ValidationError(f"test_size must be between 0 and 1, got {test_size!r}.")

        self._split_seed = _resolve_split_seed(seed)
        bucket = _split_hash(dataframe, self._split_seed, fold_key) / F.lit(float(_HASH_RANGE))
        is_test = bucket < F.lit(test_size)
        return dataframe.filter(~is_test), dataframe.filter(is_test)

    def _kfold_split(
        self,
        dataframe: DataFrame,
        n_folds: int = 5,
        seed: Optional[int] = None,
        fold_key: Optional[str] = None,
    ) -> Iterator[Tuple[DataFrame, DataFrame]]:
        """Generate k-fold cross-validation splits.

        Args:
            dataframe: PySpark DataFrame to split.
            n_folds: Number of folds. Default is 5.
            seed: Seed for the assignment. Defaults to the global seed, else random.
            fold_key: Column to hash instead of the whole row.

        Yields:
            Tuple of (train_df, val_df) for each fold.

        Raises:
            ValidationError: If n_folds is not an integer of at least 2.
        """
        if isinstance(n_folds, bool) or not isinstance(n_folds, int) or n_folds < 2:
            raise ValidationError(f"n_folds must be an integer of at least 2, got {n_folds!r}.")

        self._split_seed = _resolve_split_seed(seed)
        fold = _split_hash(dataframe, self._split_seed, fold_key) % n_folds

        for fold_idx in range(n_folds):
            yield dataframe.filter(fold != fold_idx), dataframe.filter(fold == fold_idx)
