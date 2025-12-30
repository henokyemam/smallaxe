"""Raw data generators for sample datasets."""

import random
from typing import List, Tuple

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

# Regression dataset constants
LOCATIONS = ["urban", "suburban", "rural"]
CONDITIONS = ["excellent", "good", "fair", "poor"]

# Classification dataset constants
CONTRACTS = ["month-to-month", "one_year", "two_year"]
PAYMENT_METHODS = ["credit_card", "bank_transfer", "electronic_check", "mailed_check"]


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
    random.seed(seed)

    location_multipliers = {"urban": 1.3, "suburban": 1.0, "rural": 0.7}
    condition_multipliers = {"excellent": 1.2, "good": 1.0, "fair": 0.85, "poor": 0.7}

    data = []
    for _ in range(n_rows):
        # Generate correlated features
        bedrooms = random.choices([1, 2, 3, 4, 5], weights=[10, 25, 35, 20, 10])[0]
        bathrooms = max(1, bedrooms - random.randint(0, 1))
        sqft = int(500 + bedrooms * 400 + random.gauss(0, 200))
        sqft = max(400, sqft)  # minimum sqft
        age = random.choices(list(range(0, 51)), weights=[max(1, 50 - i) for i in range(51)])[0]
        location = random.choices(LOCATIONS, weights=[30, 50, 20])[0]
        condition = random.choices(CONDITIONS, weights=[15, 45, 30, 10])[0]

        # Calculate price with realistic correlation
        base_price = 50000 + sqft * 150 + bedrooms * 10000 + bathrooms * 8000
        age_discount = age * 1000
        location_factor = location_multipliers[location]
        condition_factor = condition_multipliers[condition]

        price = (base_price - age_discount) * location_factor * condition_factor
        price = price + random.gauss(0, price * 0.1)  # Add noise
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
    random.seed(seed)

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
        tenure = random.choices(list(range(1, 73)), weights=[max(1, 72 - i) for i in range(72)])[0]
        monthly_charges = round(random.uniform(20, 120), 2)
        total_charges = round(tenure * monthly_charges * random.uniform(0.9, 1.1), 2)
        contract = random.choices(CONTRACTS, weights=[55, 25, 20])[0]
        payment_method = random.choices(PAYMENT_METHODS, weights=[25, 25, 30, 20])[0]

        # Calculate churn probability
        base_churn = contract_churn_base[contract]
        tenure_modifier = max(0, (24 - tenure) / 100)  # Higher churn for low tenure
        charge_modifier = (monthly_charges - 70) / 500  # Higher charges → more churn
        payment_modifier = payment_churn_modifier[payment_method]

        churn_prob = base_churn + tenure_modifier + charge_modifier + payment_modifier
        churn_prob = max(0.02, min(0.8, churn_prob))  # Clamp probability

        churn = 1 if random.random() < churn_prob else 0

        data.append((tenure, monthly_charges, total_charges, contract, payment_method, churn))

    return data


def load_sample_regression(spark: SparkSession, n_rows: int = 10000, seed: int = 42) -> DataFrame:
    """Load a sample regression dataset (housing prices).

    Args:
        spark: SparkSession instance.
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
    """
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
    spark: SparkSession, n_rows: int = 10000, seed: int = 42
) -> DataFrame:
    """Load a sample classification dataset (customer churn).

    Args:
        spark: SparkSession instance.
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
    """
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


def dataset_info(dataset_name: str) -> None:
    """Print information about a sample dataset.

    Args:
        dataset_name: Either 'regression' or 'classification'.

    Raises:
        ValueError: If dataset_name is not recognized.
    """
    if dataset_name == "regression":
        info = """
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
  df = load_sample_regression(spark)
"""
    elif dataset_name == "classification":
        info = """
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
  df = load_sample_classification(spark)
"""
    else:
        raise ValueError(
            f"Unknown dataset: '{dataset_name}'. Use 'regression' or 'classification'."
        )

    print(info)
