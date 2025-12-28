"""Pytest configuration and fixtures for smallaxe tests."""

import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark_session():
    """Create a SparkSession for testing.

    Session-scoped to avoid the overhead of creating a new session for each test.
    """
    spark = (
        SparkSession.builder.master("local[2]")
        .appName("smallaxe-tests")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.default.parallelism", "2")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.memory", "1g")
        .getOrCreate()
    )

    # Set log level to reduce noise during tests
    spark.sparkContext.setLogLevel("WARN")

    yield spark

    spark.stop()


@pytest.fixture
def sample_df(spark_session):
    """Create a sample DataFrame for basic testing."""
    data = [
        (1, 25, 50000.0, "A", 100.0),
        (2, 30, 60000.0, "B", 150.0),
        (3, 35, 70000.0, "A", 200.0),
        (4, 40, 80000.0, "C", 250.0),
        (5, 45, 90000.0, "B", 300.0),
    ]
    columns = ["id", "age", "income", "category", "target"]
    return spark_session.createDataFrame(data, columns)


@pytest.fixture
def sample_classification_df(spark_session):
    """Create a sample DataFrame for classification testing."""
    data = [
        (1, 25, 50000.0, "A", 0),
        (2, 30, 60000.0, "B", 1),
        (3, 35, 70000.0, "A", 0),
        (4, 40, 80000.0, "C", 1),
        (5, 45, 90000.0, "B", 1),
        (6, 28, 55000.0, "A", 0),
        (7, 33, 65000.0, "B", 1),
        (8, 38, 75000.0, "C", 0),
    ]
    columns = ["id", "age", "income", "category", "label"]
    return spark_session.createDataFrame(data, columns)
