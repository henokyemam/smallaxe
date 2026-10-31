"""Splits and folds depend only on the rows and the seed.

See ``GOTCHAS.md`` item 1. The old ``monotonically_increasing_id`` assignment changed
with partitioning and with every read of an uncached, non-deterministic source.
"""

import pytest
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

import smallaxe
from smallaxe.exceptions import ColumnNotFoundError
from smallaxe.training.mixins.validation_mixin import ValidationMixin, split_hash

N_ROWS = 2000


class _Splitter(ValidationMixin):
    pass


@pytest.fixture
def rows(spark_session):
    """Uncached rows whose lineage includes a step Spark treats as non-deterministic."""
    doubled = F.udf(lambda x: float(x) * 2.0, DoubleType()).asNondeterministic()
    return (
        spark_session.range(N_ROWS)
        .withColumn("x", doubled(F.col("id")))
        .withColumn("group", F.col("id") % 100)
        .withColumn("label", (F.col("id") % 3 == 0).cast("int"))
    )


def _ids(df):
    return {row["id"] for row in df.select("id").collect()}


def test_kfold_sets_are_disjoint_and_cover_every_row(rows):
    folds = list(_Splitter()._kfold_split(rows, n_folds=5, seed=7))
    val_sets = [_ids(val) for _, val in folds]
    for i, left in enumerate(val_sets):
        for right in val_sets[i + 1 :]:
            assert not (left & right)
    assert set().union(*val_sets) == set(range(N_ROWS))
    for (train, _), val_ids in zip(folds, val_sets):
        assert not (_ids(train) & val_ids)


def test_train_test_split_is_disjoint_and_complete(rows):
    train, test = _Splitter()._train_test_split(rows, test_size=0.3, seed=7)
    train_ids, test_ids = _ids(train), _ids(test)
    assert not (train_ids & test_ids)
    assert train_ids | test_ids == set(range(N_ROWS))
    assert abs(len(test_ids) / N_ROWS - 0.3) < 0.05


def test_folds_do_not_depend_on_partitioning(rows):
    def assignment(df):
        return {
            idx: _ids(val)
            for idx, (_, val) in enumerate(_Splitter()._kfold_split(df, n_folds=4, seed=11))
        }

    assert assignment(rows.repartition(2)) == assignment(rows.repartition(8))


def test_split_does_not_depend_on_partitioning(rows):
    def test_ids(df):
        return _ids(_Splitter()._train_test_split(df, test_size=0.2, seed=11)[1])

    assert test_ids(rows.repartition(2)) == test_ids(rows.repartition(8))


def test_seed_changes_the_assignment(rows):
    first = _ids(_Splitter()._train_test_split(rows, test_size=0.2, seed=1)[1])
    second = _ids(_Splitter()._train_test_split(rows, test_size=0.2, seed=2)[1])
    assert first != second


def test_global_seed_is_used_when_no_seed_is_given(rows):
    smallaxe.set_seed(5)
    try:
        from_global = _ids(_Splitter()._train_test_split(rows, test_size=0.2)[1])
    finally:
        smallaxe.set_seed(None)
    explicit = _ids(_Splitter()._train_test_split(rows, test_size=0.2, seed=5)[1])
    assert from_global == explicit


def test_unseeded_split_records_the_seed_it_drew(rows):
    smallaxe.set_seed(None)
    splitter = _Splitter()
    splitter._train_test_split(rows, test_size=0.2)
    assert isinstance(splitter._split_seed, int)


def test_fold_key_keeps_an_entity_in_one_fold(rows):
    for train, val in _Splitter()._kfold_split(rows, n_folds=3, seed=3, fold_key="group"):
        train_groups = {r["group"] for r in train.select("group").distinct().collect()}
        val_groups = {r["group"] for r in val.select("group").distinct().collect()}
        assert not (train_groups & val_groups)


def test_fold_key_must_exist(rows):
    with pytest.raises(ColumnNotFoundError):
        list(_Splitter()._kfold_split(rows, n_folds=3, fold_key="missing"))


def test_duplicate_rows_share_a_fold(spark_session):
    df = spark_session.createDataFrame([(1, "a")] * 50 + [(2, "b")] * 50, ["k", "v"])
    for _, val in _Splitter()._kfold_split(df, n_folds=2, seed=1):
        counts = {r["k"]: r["count"] for r in val.groupBy("k").count().collect()}
        assert all(c == 50 for c in counts.values())


def test_vector_columns_are_left_out_of_the_hash(spark_session):
    from pyspark.ml.linalg import Vectors

    df = spark_session.createDataFrame(
        [(i, Vectors.dense([float(i)])) for i in range(10)], ["id", "vec"]
    )
    assert df.select(split_hash(df, 1).alias("h")).count() == 10


def test_stratified_arguments_still_validate(rows):
    with pytest.raises(Exception, match="label_col must be specified"):
        _Splitter()._train_test_split(rows, stratified=True)
    with pytest.raises(Exception, match="label_col must be specified"):
        list(_Splitter()._kfold_split(rows, stratified=True))


def test_fit_accepts_fold_key(rows):
    from smallaxe.training import Regressors

    model = Regressors.random_forest(n_estimators=2, seed=1)
    model.fit(
        rows,
        label_col="x",
        feature_cols=["id"],
        validation="kfold",
        n_folds=2,
        fold_key="group",
    )
    assert model.validation_scores["n_folds"] == 2
