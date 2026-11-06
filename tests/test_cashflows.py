"""The payment table must rebuild the DBO exactly, and match simple hand calculations."""

import pandas as pd
import pytest

from src.pension import ASSUMPTIONS, VALUATION_DATE, expected_cash_flows, total_dbo
from src.pension_lab.cashflows import (
    RECONCILIATION_TOLERANCE_EUR,
    benefit_kept_factor,
    employee_payment_table,
    present_value,
    reconcile_to_dbo,
    yearly_payment_table,
)

# Aged exactly 60 with exactly 20 years of service at the valuation date. With no deaths, no
# salary growth and a last age of 67, the promise is three payments of 0.01 x 20 x 14,000 = 2,800
HAND_EMPLOYEE = pd.DataFrame([{
    "employee_id": "H001",
    "birth_date": pd.Timestamp("1966-12-31"),
    "entry_date": pd.Timestamp("2006-12-31"),
    "monthly_salary_eur": 1000.0,
}])
HAND_ASSUMPTIONS = dict(ASSUMPTIONS, discount_rate=0.10, salary_growth=0.0, makeham_A=0.0,
                        makeham_B=0.0, max_age=67)


@pytest.fixture(scope="module")
def payments(valid_employees):
    return employee_payment_table(valid_employees)


def test_every_employee_has_a_payment_for_every_age_to_110(payments, valid_employees):
    assert len(payments) == 33 * 46
    assert payments["employee_id"].nunique() == len(valid_employees)
    assert payments["age_at_payment"].min() == 65
    assert payments["age_at_payment"].max() == 110
    assert (payments["expected_payment"] > 0).all()


def test_present_value_reconciles_to_dbo(payments, valid_employees):
    check = reconcile_to_dbo(payments, valid_employees)
    assert check["passed"]
    assert abs(check["total_difference"]) <= RECONCILIATION_TOLERANCE_EUR
    assert check["largest_employee_difference"] <= RECONCILIATION_TOLERANCE_EUR


def test_losing_the_smallest_payment_fails_reconciliation(payments, valid_employees):
    # Proves the tolerance is tight enough to catch the smallest possible mistake
    smallest = payments["present_value"].idxmin()
    check = reconcile_to_dbo(payments.drop(index=smallest), valid_employees)
    assert not check["passed"]


@pytest.mark.parametrize("rate", [0.02, 0.03, 0.05])
def test_reconciles_at_other_discount_rates(valid_employees, rate):
    changed = dict(ASSUMPTIONS, discount_rate=rate)
    payments = employee_payment_table(valid_employees, a=changed)
    assert reconcile_to_dbo(payments, valid_employees, a=changed)["passed"]


def test_yearly_table_equals_sum_of_employee_rows(payments):
    yearly = yearly_payment_table(payments)
    assert yearly["expected_payment"].sum() == pytest.approx(payments["expected_payment"].sum())
    one_year = payments[payments["calendar_year"] == 2040]["expected_payment"].sum()
    from_table = yearly.loc[yearly["calendar_year"] == 2040, "expected_payment"].iloc[0]
    assert from_table == pytest.approx(one_year)


def test_yearly_table_matches_existing_cash_flows(payments, valid_employees):
    yearly = yearly_payment_table(payments).set_index("calendar_year")
    existing = expected_cash_flows(valid_employees).set_index("year")
    assert yearly.index.tolist() == existing.index.tolist()
    assert yearly["expected_payment"].to_numpy() == pytest.approx(
        existing["expected_payment"].to_numpy(), abs=1e-6)


def test_hand_calculated_example():
    payments = employee_payment_table(HAND_EMPLOYEE, a=HAND_ASSUMPTIONS)
    assert payments["expected_payment"].tolist() == pytest.approx([2800, 2800, 2800])
    assert payments["years_from_now"].tolist() == pytest.approx([5, 6, 7])
    # 2,800 x (1.1^-5 + 1.1^-6 + 1.1^-7), worked out by hand
    assert present_value(payments, 0.10) == pytest.approx(4755.95, abs=0.01)


def test_payment_exactly_at_year_end_stays_in_that_year():
    payments = employee_payment_table(HAND_EMPLOYEE, a=HAND_ASSUMPTIONS)
    assert payments["simulation_year"].tolist() == [5, 6, 7]
    assert payments["calendar_year"].tolist() == [2031, 2032, 2033]


def test_indexation_by_hand():
    payments = employee_payment_table(HAND_EMPLOYEE, a=HAND_ASSUMPTIONS, indexation_rate=0.02)
    assert payments["expected_payment"].tolist() == pytest.approx([2800, 2856, 2913.12])


def test_turnover_by_hand():
    # 10% a year for 5 years: 0.9^5 = 0.59049 still there at retirement
    assert benefit_kept_factor(5, 0.10, 0.0) == pytest.approx(0.59049)
    # Leavers keep half: 0.59049 + 0.40951 x 0.5
    assert benefit_kept_factor(5, 0.10, 0.5) == pytest.approx(0.795245)


def test_options_move_the_value_the_right_way(payments, valid_employees):
    base = payments["present_value"].sum()
    indexed = employee_payment_table(valid_employees, indexation_rate=0.01)
    forfeit = employee_payment_table(valid_employees, turnover_rate=0.03)
    half = employee_payment_table(valid_employees, turnover_rate=0.03, leaver_benefit_share=0.5)
    kept = employee_payment_table(valid_employees, turnover_rate=0.03, leaver_benefit_share=1.0)
    assert indexed["present_value"].sum() > base
    assert forfeit["present_value"].sum() < half["present_value"].sum() < base
    assert kept["present_value"].sum() == pytest.approx(base)


def test_higher_discount_rate_lowers_present_value(payments):
    assert present_value(payments, 0.04) < present_value(payments, 0.035)


def test_longer_lives_match_existing_dbo(valid_employees):
    changed = dict(ASSUMPTIONS, mortality_factor=0.9)
    payments = employee_payment_table(valid_employees, a=changed)
    expected = total_dbo(valid_employees, VALUATION_DATE, changed)
    assert payments["present_value"].sum() == pytest.approx(expected, abs=1e-6)
    assert expected == pytest.approx(2_936_460, abs=1)


@pytest.mark.parametrize("options", [
    {"turnover_rate": 1.0},
    {"turnover_rate": -0.1},
    {"leaver_benefit_share": 1.5},
    {"indexation_rate": -1.0},
    {"valuation_date": "2026-06-30"},
])
def test_bad_settings_are_refused(valid_employees, options):
    with pytest.raises(ValueError):
        employee_payment_table(valid_employees, **options)
