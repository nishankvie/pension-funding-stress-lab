"""Run every step from raw employee data to results, and check the run as it goes."""

import copy

import numpy as np

from src.pension import duration, sensitivity_table, validate_employees
from src.pension_lab.attention import build_attention
from src.pension_lab.cashflows import (
    employee_payment_table,
    present_value,
    reconcile_to_dbo,
    yearly_payment_table,
)
from src.pension_lab.config import values
from src.pension_lab.drivers import largest_drivers, risk_driver_table
from src.pension_lab.economy import check_correlation_matrix, correlation_matrix
from src.pension_lab.risk_metrics import funding_bands, risk_summary, stability_table
from src.pension_lab.simulation import build_payment_book, simulate
from src.pension_lab.stress import run_stress_tests
from src.pension_lab.test_status import recorded_test_status

# Simulated correlations can differ from their targets by chance, but not by more than this
CORRELATION_CHECK_LIMIT = 0.02
# Differences above this are real, not computer rounding
ROUNDING_LIMIT_EUR = 0.000001


def check_result(name, status, detail):
    """Return one model check as a plain record."""
    return {"name": name, "status": status, "detail": detail}


def passed_or_failed(name, passed, detail):
    """Return a check that can only pass or fail."""
    return check_result(name, "passed" if passed else "failed", detail)


def reconciliation_check(reconciliation, plan):
    """Check the payments rebuild the DBO, which only applies with both options off."""
    name = "Payments reconcile to the DBO"
    if plan["turnover_rate"] != 0 or plan["indexation_rate"] != 0:
        return check_result(name, "incomplete", "Not applicable while turnover or indexation is on.")
    return passed_or_failed(name, reconciliation["passed"],
                            f"Difference EUR {reconciliation['total_difference']:.9f}")


def zero_volatility_check(book, config, payments, rate):
    """Check that with no risk the simulation matches a direct calculation of the final year."""
    horizon = values(config, "simulation")["horizon_years"]
    calm = simulate(book, config, paths=2, active_risks=())
    remaining = payments[payments["simulation_year"] > horizon]
    expected = (remaining["expected_payment"]
                * (1 + rate) ** -(remaining["years_from_now"] - horizon)).sum()
    difference = abs(calm["obligation"][0, -1] - expected)
    return passed_or_failed("No risk matches the direct projection",
                            difference <= ROUNDING_LIMIT_EUR, f"Difference EUR {difference:.9f}")


def seed_check(book, config):
    """Check that the same seed repeats exactly."""
    first = simulate(book, config, paths=200)
    second = simulate(book, config, paths=200)
    same = np.array_equal(first["obligation"], second["obligation"])
    return passed_or_failed("Same seed gives the same result", same, "Two runs compared")


def correlation_check(run, matrix):
    """Check the simulated shocks have the target correlations."""
    measured = np.corrcoef(run["shocks"].reshape(-1, matrix.shape[0]), rowvar=False)
    largest = float(np.abs(measured - matrix).max())
    return passed_or_failed("Simulated correlations match targets",
                            largest <= CORRELATION_CHECK_LIMIT, f"Largest gap {largest:.4f}")


def starting_funding_check(run, plan):
    """Check that every path of every portfolio starts at the funding input."""
    target = plan["starting_funding_level"]
    largest = max(float(np.abs(result["funding_level"][:, 0] - target).max())
                  for result in run["portfolios"].values())
    return passed_or_failed("Starting funding equals the input", largest < 1e-9,
                            f"Largest gap {largest:.12f}")


def direction_check(payments, stress, rate):
    """Check a higher rate lowers the obligation, and longer lives raise it."""
    rate_falls = present_value(payments, rate + 0.01) < present_value(payments, rate)
    base = present_value(payments, rate)
    longer = stress.loc[stress["stress_id"] == "longer_lives", "obligation"].iloc[0]
    return passed_or_failed("Obligation moves the right way", bool(rate_falls and longer > base),
                            "Higher rate lowers it, longer lives raise it")


def payment_table_check(payments, yearly):
    """Check the yearly table adds up to the employee rows."""
    difference = abs(yearly["expected_payment"].sum() - payments["expected_payment"].sum())
    return passed_or_failed("Yearly table equals employee rows", difference <= ROUNDING_LIMIT_EUR,
                            f"Difference EUR {difference:.9f}")


def stability_check(stability, limit):
    """Check results hold between the default and the larger run."""
    largest = float(stability["difference"].max())
    return passed_or_failed("Results stable between path counts", largest <= limit,
                            f"Largest difference {largest:.4f}, limit {limit:.4f}")


def bond_formula_check(run):
    """Check the bond formula stayed inside the range where it moves the right way."""
    largest = max(run["checks"]["wrong_way_share"].values())
    return passed_or_failed("Bond formula stayed in its safe range", largest == 0,
                            f"Share of path-years outside it {largest:.6f}")


def automated_tests_check(status):
    """Report whether the tests have passed on exactly this code and data."""
    return check_result("Automated tests", status["status"], status["detail"])


def comparison_summary(book, config):
    """Return the risk measures with the brief's lower inflation volatility."""
    changed = copy.deepcopy(config)
    economy = changed["economy"]
    economy["inflation_volatility"]["value"] = economy["inflation_volatility_comparison"]["value"]
    return risk_summary(simulate(book, changed), changed)


def run_pipeline(raw_df, config, source_name="sample employees", tests=None):
    """Return every result of one run, from validation to the model checks."""
    pension = values(config, "pension")
    plan = values(config, "plan")
    rate = pension["discount_rate"]

    valid, log, queries = validate_employees(raw_df)
    payments = employee_payment_table(
        valid, a=pension, turnover_rate=plan["turnover_rate"],
        leaver_benefit_share=plan["leaver_benefit_share"],
        indexation_rate=plan["indexation_rate"],
    )
    yearly = yearly_payment_table(payments)
    reconciliation = reconcile_to_dbo(payments, valid, a=pension)
    sensitivity = sensitivity_table(valid, a=pension)
    stress = run_stress_tests(valid, config)

    matrix = correlation_matrix(values(config, "correlations"))
    check_correlation_matrix(matrix)
    book = build_payment_book(valid, config)
    run = simulate(book, config)
    summary = risk_summary(run, config)
    drivers = risk_driver_table(book, config)
    stability = stability_table(book, config)
    tests = recorded_test_status() if tests is None else tests

    checks = [
        reconciliation_check(reconciliation, plan),
        payment_table_check(payments, yearly),
        zero_volatility_check(book, config, payments, rate),
        seed_check(book, config),
        correlation_check(run, matrix),
        starting_funding_check(run, plan),
        direction_check(payments, stress, rate),
        stability_check(stability, values(config, "simulation")["stability_limit"]),
        bond_formula_check(run),
        automated_tests_check(tests),
    ]

    results = {
        "source_name": source_name,
        "raw": raw_df,
        "valid": valid,
        "log": log,
        "queries": queries,
        "payments": payments,
        "yearly": yearly,
        "reconciliation": reconciliation,
        "sensitivity": sensitivity,
        "obligation_duration": duration(sensitivity),
        "stress": stress,
        "run": run,
        "bands": funding_bands(run),
        "summary": summary,
        "inflation_comparison": comparison_summary(book, config),
        "drivers": drivers,
        "largest_drivers": largest_drivers(drivers),
        "stability": stability,
        "checks": checks,
        "tests": tests,
    }
    results["attention"] = build_attention(results, config)
    return results
