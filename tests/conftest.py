"""Shared test data: the sample employees, validated once."""

import hashlib
import re

import pandas as pd
import pytest

from src.pension import DATA_DIR, validate_employees


# Fingerprints of names that must never appear on the page. Stored this way so that a public
# copy of the tests doesn't contain the names it is checking for
HIDDEN_WORD_HASHES = {
    "e2ca6aec52558fcc04311983bad03e38ca5e799c239d202ae7a4f58aff8a6970",
    "c8f434fa70cb5423ff3110a7cedfa9fa1e90fbd5e0cc3c8b4b65cf684c54de1e",
    "9975df775a995cdc875708b4d635fb60bf8f4ef99ad02085952b37dd9e5bb79a",
    "4a2e9726d2d48b0ead5dc53afe460e8f863a050a8841a1810dec02804c34c715",
}


def hidden_words_in(text):
    """Return any word in the text whose fingerprint is on the hidden list."""
    words = set(re.findall(r"[a-z]+", text.lower()))
    return [word for word in words
            if hashlib.sha256(word.encode()).hexdigest() in HIDDEN_WORD_HASHES]


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
