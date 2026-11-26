"""Sample datasets: synthetic housing prices (regression) and customer churn (binary)."""

import copy
import random
from typing import Any, Dict, List, Optional, Tuple

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from smallaxe.exceptions import ValidationError

__all__ = ["load_sample_regression", "load_sample_classification", "dataset_info"]

# Regression dataset constants
_LOCATIONS = ["urban", "suburban", "rural"]
_CONDITIONS = ["excellent", "good", "fair", "poor"]

# Classification dataset constants
_CONTRACTS = ["month-to-month", "one_year", "two_year"]
_PAYMENT_METHODS = ["credit_card", "bank_transfer", "electronic_check", "mailed_check"]


def _generate_regression_data(
    n_rows: int = 10000, seed: int = 42
) -> List[Tuple[int, int, int, int, str, str, float]]:
    """Generate synthetic housing data with realistic distributions.

    The price is correlated with features:
    - More bedrooms/bathrooms → higher price
    - More sqft → higher price
    - Newer homes (lower age) → higher price
    - Urban > suburban > rural
    - Excellent > good > fair > poor condition
    """
    # A local generator: seeding the global ``random`` module would also fix the
    # random split seed of every later unseeded fit.
    rng = random.Random(seed)

    location_multipliers = {"urban": 1.3, "suburban": 1.0, "rural": 0.7}
    condition_multipliers = {"excellent": 1.2, "good": 1.0, "fair": 0.85, "poor": 0.7}

    data = []
    for _ in range(n_rows):
        # Generate correlated features
        bedrooms = rng.choices([1, 2, 3, 4, 5], weights=[10, 25, 35, 20, 10])[0]
        bathrooms = max(1, bedrooms - rng.randint(0, 1))
        sqft = int(500 + bedrooms * 400 + rng.gauss(0, 200))
        sqft = max(400, sqft)  # minimum sqft
        age = rng.choices(list(range(0, 51)), weights=[max(1, 50 - i) for i in range(51)])[0]
        location = rng.choices(_LOCATIONS, weights=[30, 50, 20])[0]
        condition = rng.choices(_CONDITIONS, weights=[15, 45, 30, 10])[0]

        # Calculate price with realistic correlation
        base_price = 50000 + sqft * 150 + bedrooms * 10000 + bathrooms * 8000
        age_discount = age * 1000
        location_factor = location_multipliers[location]
        condition_factor = condition_multipliers[condition]

        price = (base_price - age_discount) * location_factor * condition_factor
        price = price + rng.gauss(0, price * 0.1)  # Add noise
        price = max(50000, round(price, 2))  # Minimum price

        data.append((bedrooms, bathrooms, sqft, age, location, condition, price))

    return data


def _generate_classification_data(
    n_rows: int = 10000, seed: int = 42
) -> List[Tuple[int, float, float, str, str, int]]:
    """Generate synthetic customer churn data with realistic distributions.

    Churn probability is correlated with features:
    - Lower tenure → higher churn
    - Higher monthly charges → higher churn
    - Month-to-month contract → higher churn
    - Electronic check payment → higher churn
    """
    # A local generator: seeding the global ``random`` module would also fix the
    # random split seed of every later unseeded fit.
    rng = random.Random(seed)

    contract_churn_base = {"month-to-month": 0.4, "one_year": 0.15, "two_year": 0.05}
    payment_churn_modifier = {
        "credit_card": -0.05,
        "bank_transfer": -0.05,
        "electronic_check": 0.1,
        "mailed_check": 0.0,
    }

    data = []
    for _ in range(n_rows):
        # Generate features
        tenure = rng.choices(list(range(1, 73)), weights=[max(1, 72 - i) for i in range(72)])[0]
        monthly_charges = round(rng.uniform(20, 120), 2)
        total_charges = round(tenure * monthly_charges * rng.uniform(0.9, 1.1), 2)
        contract = rng.choices(_CONTRACTS, weights=[55, 25, 20])[0]
        payment_method = rng.choices(_PAYMENT_METHODS, weights=[25, 25, 30, 20])[0]

        # Calculate churn probability
        base_churn = contract_churn_base[contract]
        tenure_modifier = max(0, (24 - tenure) / 100)  # Higher churn for low tenure
        charge_modifier = (monthly_charges - 70) / 500  # Higher charges → more churn
        payment_modifier = payment_churn_modifier[payment_method]

        churn_prob = base_churn + tenure_modifier + charge_modifier + payment_modifier
        churn_prob = max(0.02, min(0.8, churn_prob))  # Clamp probability

        churn = 1 if rng.random() < churn_prob else 0

        data.append((tenure, monthly_charges, total_charges, contract, payment_method, churn))

    return data


def _session_and_checks(spark: Optional[SparkSession], n_rows: Any, seed: Any) -> SparkSession:
    """The session to build the frame in, after checking the loader's arguments.

    Raises:
        ValidationError: If ``spark`` is None and there is no active SparkSession, or is
            not a SparkSession; or ``n_rows`` is not a positive int or ``seed`` not an int.
    """
    if isinstance(n_rows, bool) or not isinstance(n_rows, int) or n_rows < 1:
        raise ValidationError(f"n_rows must be a positive integer, got {n_rows!r}.")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValidationError(f"seed must be an integer, got {seed!r}.")
    if spark is None:
        spark = SparkSession.getActiveSession()
        if spark is None:
            raise ValidationError(
                "There is no active SparkSession. Pass one as spark=, or create one first "
                "(SparkSession.builder.getOrCreate())."
            )
        return spark
    if not hasattr(spark, "createDataFrame"):
        raise ValidationError(f"spark must be a SparkSession, got {type(spark).__name__}.")
    return spark


def load_sample_regression(
    spark: Optional[SparkSession] = None, n_rows: int = 10000, seed: int = 42
) -> DataFrame:
    """Load a sample regression dataset (housing prices).

    The rows depend only on ``n_rows`` and ``seed``, never on Python's global ``random``
    state, which loading leaves untouched.

    Args:
        spark: The SparkSession to create the DataFrame in. Defaults to the active
            session.
        n_rows: Number of rows to generate. Default is 10,000.
        seed: Random seed for reproducibility. Default is 42.

    Returns:
        PySpark DataFrame with columns:
            - bedrooms (int): Number of bedrooms (1-5)
            - bathrooms (int): Number of bathrooms (1-5)
            - sqft (int): Square footage (400+)
            - age (int): Age of home in years (0-50)
            - location (str): 'urban', 'suburban', or 'rural'
            - condition (str): 'excellent', 'good', 'fair', or 'poor'
            - price (float): House price in dollars (label column)

    Raises:
        ValidationError: If ``spark`` is None and there is no active SparkSession, or
            ``n_rows`` or ``seed`` is invalid.

    Example:
        >>> houses = load_sample_regression(n_rows=1000)
    """
    spark = _session_and_checks(spark, n_rows, seed)
    schema = StructType(
        [
            StructField("bedrooms", IntegerType(), False),
            StructField("bathrooms", IntegerType(), False),
            StructField("sqft", IntegerType(), False),
            StructField("age", IntegerType(), False),
            StructField("location", StringType(), False),
            StructField("condition", StringType(), False),
            StructField("price", DoubleType(), False),
        ]
    )

    data = _generate_regression_data(n_rows=n_rows, seed=seed)
    return spark.createDataFrame(data, schema)


def load_sample_classification(
    spark: Optional[SparkSession] = None, n_rows: int = 10000, seed: int = 42
) -> DataFrame:
    """Load a sample binary classification dataset (customer churn).

    The rows depend only on ``n_rows`` and ``seed``, never on Python's global ``random``
    state, which loading leaves untouched.

    Args:
        spark: The SparkSession to create the DataFrame in. Defaults to the active
            session.
        n_rows: Number of rows to generate. Default is 10,000.
        seed: Random seed for reproducibility. Default is 42.

    Returns:
        PySpark DataFrame with columns:
            - tenure (int): Months as customer (1-72)
            - monthly_charges (float): Monthly bill amount (20-120)
            - total_charges (float): Total amount charged
            - contract (str): 'month-to-month', 'one_year', or 'two_year'
            - payment_method (str): Payment method used
            - churn (int): 1 if churned, 0 otherwise (label column)

    Raises:
        ValidationError: If ``spark`` is None and there is no active SparkSession, or
            ``n_rows`` or ``seed`` is invalid.

    Example:
        >>> churn = load_sample_classification(n_rows=1000)
    """
    spark = _session_and_checks(spark, n_rows, seed)
    schema = StructType(
        [
            StructField("tenure", IntegerType(), False),
            StructField("monthly_charges", DoubleType(), False),
            StructField("total_charges", DoubleType(), False),
            StructField("contract", StringType(), False),
            StructField("payment_method", StringType(), False),
            StructField("churn", IntegerType(), False),
        ]
    )

    data = _generate_classification_data(n_rows=n_rows, seed=seed)
    return spark.createDataFrame(data, schema)


_INFO: Dict[str, Dict[str, Any]] = {
    "regression": {
        "name": "regression",
        "title": "Housing Prices",
        "task": "simple_regression",
        "columns": {
            "bedrooms": {"type": "int", "description": "Number of bedrooms (1-5)"},
            "bathrooms": {"type": "int", "description": "Number of bathrooms (1-5)"},
            "sqft": {"type": "int", "description": "Square footage of the home (400+)"},
            "age": {"type": "int", "description": "Age of the home in years (0-50)"},
            "location": {"type": "str", "description": "'urban', 'suburban', or 'rural'"},
            "condition": {
                "type": "str",
                "description": "'excellent', 'good', 'fair', or 'poor'",
            },
            "price": {"type": "float", "description": "House price in dollars (label)"},
        },
        "numerical_features": ["bedrooms", "bathrooms", "sqft", "age"],
        "categorical_features": ["location", "condition"],
        "label": "price",
        "loader": "load_sample_regression",
    },
    "classification": {
        "name": "classification",
        "title": "Customer Churn",
        "task": "binary",
        "columns": {
            "tenure": {"type": "int", "description": "Number of months as a customer (1-72)"},
            "monthly_charges": {"type": "float", "description": "Monthly bill amount (20-120)"},
            "total_charges": {
                "type": "float",
                "description": "Total amount charged over tenure",
            },
            "contract": {
                "type": "str",
                "description": "'month-to-month', 'one_year', or 'two_year'",
            },
            "payment_method": {
                "type": "str",
                "description": (
                    "'credit_card', 'bank_transfer', 'electronic_check', or 'mailed_check'"
                ),
            },
            "churn": {
                "type": "int",
                "description": "1 if customer churned, 0 otherwise (label)",
            },
        },
        "numerical_features": ["tenure", "monthly_charges", "total_charges"],
        "categorical_features": ["contract", "payment_method"],
        "label": "churn",
        "loader": "load_sample_classification",
        "class_distribution": {"1": 0.3, "0": 0.7},
    },
}

# What dataset_info(verbose=True) prints.
_TEXT = {
    "regression": """
Sample Regression Dataset: Housing Prices
==========================================

Columns:
  - bedrooms (int): Number of bedrooms (1-5)
  - bathrooms (int): Number of bathrooms (1-5)
  - sqft (int): Square footage of the home (400+)
  - age (int): Age of the home in years (0-50)
  - location (str): Location type - 'urban', 'suburban', or 'rural'
  - condition (str): Home condition - 'excellent', 'good', 'fair', or 'poor'
  - price (float): House price in dollars (LABEL COLUMN)

Numerical features: bedrooms, bathrooms, sqft, age
Categorical features: location, condition
Label: price

Usage:
  from smallaxe.datasets import load_sample_regression
  dataframe = load_sample_regression()
""",
    "classification": """
Sample Classification Dataset: Customer Churn
==============================================

Columns:
  - tenure (int): Number of months as a customer (1-72)
  - monthly_charges (float): Monthly bill amount (20-120)
  - total_charges (float): Total amount charged over tenure
  - contract (str): Contract type - 'month-to-month', 'one_year', or 'two_year'
  - payment_method (str): 'credit_card', 'bank_transfer', 'electronic_check', or 'mailed_check'
  - churn (int): 1 if customer churned, 0 otherwise (LABEL COLUMN)

Numerical features: tenure, monthly_charges, total_charges
Categorical features: contract, payment_method
Label: churn (binary: 0 or 1)

Class distribution: ~30% churn (1), ~70% no churn (0)

Usage:
  from smallaxe.datasets import load_sample_classification
  dataframe = load_sample_classification()
""",
}


def dataset_info(dataset_name: str, *, verbose: bool = False) -> Dict[str, Any]:
    """Describe a sample dataset: its columns, features, label and loader.

    Args:
        dataset_name: Either 'regression' or 'classification'.
        verbose: Also print a readable description. Default False.

    Returns:
        A new dict with ``name``, ``title``, ``task``, ``columns`` (column name ->
        ``{"type", "description"}``), ``numerical_features``, ``categorical_features``,
        ``label`` and ``loader`` (the function in :mod:`smallaxe.datasets` that loads
        it); the classification dataset adds ``class_distribution``.

    Raises:
        ValidationError: If dataset_name is not recognized.

    Example:
        >>> info = dataset_info('regression')
        >>> info['label']
        'price'
    """
    if not isinstance(dataset_name, str) or dataset_name not in _INFO:
        raise ValidationError(
            f"Unknown dataset: {dataset_name!r}. Use 'regression' or 'classification'."
        )
    if verbose:
        print(_TEXT[dataset_name])
    return copy.deepcopy(_INFO[dataset_name])
