"""Smoke tests for smallaxe package."""


def test_import_smallaxe():
    """Test that smallaxe can be imported."""
    import smallaxe

    assert smallaxe is not None


def test_version():
    """Test that version is defined."""
    import smallaxe

    assert hasattr(smallaxe, "__version__")
    assert isinstance(smallaxe.__version__, str)
    assert smallaxe.__version__ == "0.1.0"


def test_import_submodules():
    """Test that all submodules can be imported."""
    from smallaxe import (
        auto,
        datasets,
        exceptions,
        metrics,
        pipeline,
        preprocessing,
        search,
        training,
        viz,
    )

    assert training is not None
    assert preprocessing is not None
    assert pipeline is not None
    assert metrics is not None
    assert search is not None
    assert auto is not None
    assert viz is not None
    assert exceptions is not None
    assert datasets is not None


def test_spark_session_fixture(spark_session):
    """Test that the Spark session fixture works."""
    assert spark_session is not None
    assert spark_session.version is not None


def test_sample_df_fixture(sample_df):
    """Test that the sample DataFrame fixture works."""
    assert sample_df is not None
    assert sample_df.count() == 5
    assert set(sample_df.columns) == {"id", "age", "income", "category", "target"}


def test_sample_classification_df_fixture(sample_classification_df):
    """Test that the classification DataFrame fixture works."""
    assert sample_classification_df is not None
    assert sample_classification_df.count() == 8
    assert "label" in sample_classification_df.columns
