"""Shared test data: the sample employees, validated once."""

import pandas as pd
import pytest

from src.pension import DATA_DIR, validate_employees


@pytest.fixture(scope="session")
def raw_employees():
    return pd.read_csv(DATA_DIR / "employees_raw.csv")


@pytest.fixture(scope="session")
def validated(raw_employees):
    return validate_employees(raw_employees)


@pytest.fixture(scope="session")
def valid_employees(validated):
    return validated[0]
