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


@pytest.fixture(scope="session")
def pipeline_config():
    from src.pension_lab.config import load_config
    return load_config()


@pytest.fixture(scope="session")
def pipeline_results(raw_employees, pipeline_config):
    from src.pension_lab.pipeline import run_pipeline
    # Test status is given here, because a test run can't report on its own outcome
    tests = {"status": "passed", "detail": "Given for this test.", "recorded": None}
    return run_pipeline(raw_employees, pipeline_config, tests=tests)
