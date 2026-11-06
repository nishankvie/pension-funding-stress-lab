"""Bad records must become client queries, never silent changes to the liability."""

import pandas as pd
import pytest

from src.pension import DATA_DIR, validate_employees

GOOD_ROW = {"employee_id": "T001", "birth_date": "1980-01-15", "entry_date": "2010-03-01",
            "monthly_salary_eur": 4000.0}


def run_with(**changes):
    """Validate one good employee plus one changed employee."""
    changed = {**GOOD_ROW, "employee_id": "T002", **changes}
    return validate_employees(pd.DataFrame([GOOD_ROW, changed]))


def test_clean_file_does_not_crash():
    truth = pd.read_csv(DATA_DIR / "employees_truth.csv")
    valid, log, queries = validate_employees(truth)
    assert len(valid) == 40
    assert log.empty
    assert queries.empty


@pytest.mark.parametrize("changes, issue", [
    ({"entry_date": "2030-01-01"}, "entry after valuation date"),
    ({"monthly_salary_eur": -5000.0}, "salary out of range"),
    ({"monthly_salary_eur": 0.0}, "salary out of range"),
    ({"monthly_salary_eur": 99999.0}, "salary out of range"),
    ({"entry_date": None}, "missing date"),
    ({"birth_date": None}, "missing date"),
])
def test_bad_record_is_queried_and_excluded(changes, issue):
    valid, _, queries = run_with(**changes)
    assert valid["employee_id"].tolist() == ["T001"]
    assert queries["employee_id"].tolist() == ["T002"]
    assert queries["issue"].tolist() == [issue]


def test_salary_at_the_limits_is_accepted():
    for salary in (2500.0, 8000.0):
        valid, _, queries = run_with(monthly_salary_eur=salary)
        assert len(valid) == 2
        assert queries.empty


def test_same_id_with_different_salary_is_queried_once():
    conflict = {**GOOD_ROW, "monthly_salary_eur": 7000.0}
    valid, _, queries = validate_employees(pd.DataFrame([GOOD_ROW, conflict]))
    assert valid.empty
    assert queries["issue"].tolist() == ["conflicting records"]


def test_exact_duplicate_is_dropped_not_queried():
    valid, log, queries = validate_employees(pd.DataFrame([GOOD_ROW, GOOD_ROW]))
    assert len(valid) == 1
    assert log["issue"].tolist() == ["exact duplicate"]
    assert queries.empty
