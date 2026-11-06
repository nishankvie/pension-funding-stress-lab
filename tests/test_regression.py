"""The Part 6 values from the earlier build. If one fails, stop and explain before continuing."""

import hashlib

import pytest

from src.pension import ASSUMPTIONS, DATA_DIR, VALUATION_DATE, dbo_portfolio, total_dbo

RAW_DATA_SHA256 = "e76d0f0c80c9d323b5b3843e24e92e6930dfe5500565b52eb916658f403c93a0"


def test_sample_data_is_unchanged():
    # Running src/pension.py directly rewrites this file, so the values below depend on this check
    content = (DATA_DIR / "employees_raw.csv").read_bytes()
    assert hashlib.sha256(content).hexdigest() == RAW_DATA_SHA256


def test_row_and_query_counts(raw_employees, validated):
    valid, log, queries = validated
    assert len(raw_employees) == 43
    assert len(valid) == 33
    assert len(queries) == 8
    assert len(log) == 6


def test_total_dbo(valid_employees):
    assert dbo_portfolio(valid_employees)[1] == pytest.approx(2_821_995.55, abs=0.01)


def test_employee_e017_dbo(valid_employees):
    results = dbo_portfolio(valid_employees)[0].set_index("employee_id")
    assert results.loc["E017", "dbo"] == pytest.approx(63_742.43, abs=0.01)


def test_lower_discount_rate_raises_dbo_by_about_12_percent(valid_employees):
    base = total_dbo(valid_employees, VALUATION_DATE, ASSUMPTIONS)
    lower = total_dbo(valid_employees, VALUATION_DATE, dict(ASSUMPTIONS, discount_rate=0.03))
    assert lower / base - 1 == pytest.approx(0.1211, abs=0.0001)
