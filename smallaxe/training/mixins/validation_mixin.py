"""ValidationMixin for data splitting and cross-validation."""

from typing import Iterator, List, Optional, Tuple

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

import smallaxe
from smallaxe.exceptions import ValidationError


class ValidationMixin:
    """Mixin providing data splitting and cross-validation functionality.

    This mixin provides methods for splitting data into train/test sets
    and performing k-fold cross-validation, with optional stratification
    for classification tasks.
    """

    def _train_test_split(
        self,
        df: DataFrame,
        test_size: float = 0.2,
        stratified: bool = False,
        label_col: Optional[str] = None,
    ) -> Tuple[DataFrame, DataFrame]:
        """Split DataFrame into training and test sets.

        Args:
            df: PySpark DataFrame to split.
            test_size: Proportion of data for the test set. Default is 0.2.
            stratified: If True, preserve class distribution in both sets.
                Requires label_col to be specified. Default is False.
            label_col: Column name for stratification. Required if stratified=True.

        Returns:
            Tuple of (train_df, test_df).

        Raises:
            ValidationError: If test_size is not between 0 and 1.
            ValidationError: If stratified=True but label_col is not specified.
        """
        if not 0 < test_size < 1:
            raise ValidationError(
                f"test_size must be between 0 and 1, got {test_size}."
            )

        if stratified and not label_col:
            raise ValidationError(
                "label_col must be specified when stratified=True."
            )

        seed = smallaxe.get_seed()

        if stratified:
            return self._stratified_train_test_split(df, test_size, label_col, seed)
        else:
            train_ratio = 1.0 - test_size
            if seed is not None:
                train_df, test_df = df.randomSplit([train_ratio, test_size], seed=seed)
            else:
                train_df, test_df = df.randomSplit([train_ratio, test_size])
            return train_df, test_df

    def _stratified_train_test_split(
        self,
        df: DataFrame,
        test_size: float,
        label_col: str,
        seed: Optional[int] = None,
    ) -> Tuple[DataFrame, DataFrame]:
        """Perform stratified train/test split preserving class distribution.

        Args:
            df: PySpark DataFrame to split.
            test_size: Proportion of data for the test set.
            label_col: Column name for stratification.
            seed: Random seed for reproducibility.

        Returns:
            Tuple of (train_df, test_df).
        """
        train_ratio = 1.0 - test_size

        # Get unique labels
        labels = [row[label_col] for row in df.select(label_col).distinct().collect()]

        train_dfs: List[DataFrame] = []
        test_dfs: List[DataFrame] = []

        for label in labels:
            label_df = df.filter(F.col(label_col) == label)
            if seed is not None:
                train_part, test_part = label_df.randomSplit(
                    [train_ratio, test_size], seed=seed
                )
            else:
                train_part, test_part = label_df.randomSplit([train_ratio, test_size])
            train_dfs.append(train_part)
            test_dfs.append(test_part)

        # Union all parts
        train_df = train_dfs[0]
        test_df = test_dfs[0]
        for i in range(1, len(labels)):
            train_df = train_df.union(train_dfs[i])
            test_df = test_df.union(test_dfs[i])

        return train_df, test_df

    def _kfold_split(
        self,
        df: DataFrame,
        n_folds: int = 5,
        stratified: bool = False,
        label_col: Optional[str] = None,
    ) -> Iterator[Tuple[DataFrame, DataFrame]]:
        """Generate k-fold cross-validation splits.

        Args:
            df: PySpark DataFrame to split.
            n_folds: Number of folds. Default is 5.
            stratified: If True, preserve class distribution in each fold.
                Requires label_col to be specified. Default is False.
            label_col: Column name for stratification. Required if stratified=True.

        Yields:
            Tuple of (train_df, val_df) for each fold.

        Raises:
            ValidationError: If n_folds < 2.
            ValidationError: If stratified=True but label_col is not specified.
        """
        if n_folds < 2:
            raise ValidationError(f"n_folds must be at least 2, got {n_folds}.")

        if stratified and not label_col:
            raise ValidationError(
                "label_col must be specified when stratified=True."
            )

        seed = smallaxe.get_seed()

        if stratified:
            yield from self._stratified_kfold_split(df, n_folds, label_col, seed)
        else:
            yield from self._simple_kfold_split(df, n_folds, seed)

    def _simple_kfold_split(
        self,
        df: DataFrame,
        n_folds: int,
        seed: Optional[int] = None,
    ) -> Iterator[Tuple[DataFrame, DataFrame]]:
        """Generate simple k-fold splits without stratification.

        Args:
            df: PySpark DataFrame to split.
            n_folds: Number of folds.
            seed: Random seed for reproducibility.

        Yields:
            Tuple of (train_df, val_df) for each fold.
        """
        # Add fold assignment column using row_number with random ordering
        # This ensures even distribution of rows across folds
        fold_col = "__fold__"
        rand_col = "__rand__"

        if seed is not None:
            df_with_rand = df.withColumn(rand_col, F.rand(seed))
        else:
            df_with_rand = df.withColumn(rand_col, F.rand())

        # Use row_number with random ordering, then modulo for fold assignment
        window = Window.orderBy(rand_col)
        df_with_fold = df_with_rand.withColumn(
            fold_col,
            (F.row_number().over(window) - 1) % n_folds
        ).drop(rand_col)

        for fold_idx in range(n_folds):
            val_df = df_with_fold.filter(F.col(fold_col) == fold_idx).drop(fold_col)
            train_df = df_with_fold.filter(F.col(fold_col) != fold_idx).drop(fold_col)
            yield train_df, val_df

    def _stratified_kfold_split(
        self,
        df: DataFrame,
        n_folds: int,
        label_col: str,
        seed: Optional[int] = None,
    ) -> Iterator[Tuple[DataFrame, DataFrame]]:
        """Generate stratified k-fold splits preserving class distribution.

        Args:
            df: PySpark DataFrame to split.
            n_folds: Number of folds.
            label_col: Column name for stratification.
            seed: Random seed for reproducibility.

        Yields:
            Tuple of (train_df, val_df) for each fold.
        """
        fold_col = "__fold__"
        rand_col = "__rand__"

        # Get unique labels
        labels = [row[label_col] for row in df.select(label_col).distinct().collect()]

        # Assign folds within each class using row_number with random ordering
        df_parts: List[DataFrame] = []
        for label in labels:
            label_df = df.filter(F.col(label_col) == label)
            if seed is not None:
                label_df = label_df.withColumn(rand_col, F.rand(seed))
            else:
                label_df = label_df.withColumn(rand_col, F.rand())

            # Use row_number with random ordering, then modulo for fold assignment
            window = Window.orderBy(rand_col)
            label_df = label_df.withColumn(
                fold_col,
                (F.row_number().over(window) - 1) % n_folds
            ).drop(rand_col)
            df_parts.append(label_df)

        # Union all parts
        df_with_fold = df_parts[0]
        for i in range(1, len(labels)):
            df_with_fold = df_with_fold.union(df_parts[i])

        # Generate folds
        for fold_idx in range(n_folds):
            val_df = df_with_fold.filter(F.col(fold_col) == fold_idx).drop(fold_col)
            train_df = df_with_fold.filter(F.col(fold_col) != fold_idx).drop(fold_col)
            yield train_df, val_df
