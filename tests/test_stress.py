"""Stress results must match hand calculations and move in the right direction."""

import copy

import numpy as np
import pandas as pd
import pytest

from src.pension import ASSUMPTIONS, VALUATION_DATE, total_dbo
from src.pension_lab.config import load_config, values
from src.pension_lab.stress import (
    RISK_NAMES,
    assets_after,
    extra_inflation_factor,
    run_stress_tests,
    single_risk_stresses,
)

BASE_DBO = 2_821_995.55
HAND_PORTFOLIO = {"government": 0.5, "corporate": 0.25, "equity": 0.25, "bond_duration": 10,
                  "bond_convexity": 100}


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def results(config, valid_employees):
    return run_stress_tests(valid_employees, config)


def pick(results, stress_id, portfolio_id):
    chosen = results[(results["stress_id"] == stress_id)
                     & (results["portfolio_id"] == portfolio_id)]
    assert len(chosen) == 1
    return chosen.iloc[0]


def test_every_stress_applies_to_every_portfolio(results):
    assert len(results) == 18
    assert results["stress_id"].nunique() == 6
    assert results["portfolio_id"].nunique() == 3


def test_gap_and_funding_level_are_consistent(results):
    assert (results["gap"] == results["assets"] - results["obligation"]).all()
    assert (results["funding_level"] == results["assets"] / results["obligation"]).all()


def test_assets_by_hand():
    # Bonds: -10 x -1 point = +10%, plus 0.5 x 100 x 0.01^2 = +0.5%, so +10.5%. Equity unchanged.
    # 100 x (1 + 0.75 x 0.105) = 107.875
    assert assets_after(100.0, HAND_PORTFOLIO, {"rate_shift": -0.01}) == pytest.approx(107.875)


def test_equity_and_spread_by_hand():
    # Corporate: -10 x 1 point + 0.5% = -9.5%. Equity -30%.
    # 100 x (1 + 0.25 x -0.095 + 0.25 x -0.30) = 90.125
    stress = {"equity_shock": -0.30, "spread_shift": 0.01}
    assert assets_after(100.0, HAND_PORTFOLIO, stress) == pytest.approx(90.125)


def test_rates_down_and_spreads_up_cancel_on_corporate_bonds():
    # Corporate yields don't move, so only the government half gains: 100 x (1 + 0.5 x 0.105)
    stress = {"rate_shift": -0.01, "spread_shift": 0.01}
    assert assets_after(100.0, HAND_PORTFOLIO, stress) == pytest.approx(105.25)


def test_inflation_factor_by_hand():
    payments = pd.DataFrame({"years_from_now": [3.0, 4.0, 10.0, 11.0],
                             "payment_number": [0, 1, 0, 1]})
    factors = extra_inflation_factor(payments, 0.01, 5)
    # Retires in 3 years, so only 3 years count, even for the later payment. Retires in 10: all 5
    assert factors.tolist() == pytest.approx([1.01 ** 3, 1.01 ** 3, 1.01 ** 5, 1.01 ** 5])


def test_lower_inflation_lowers_payments():
    payments = pd.DataFrame({"years_from_now": [10.0], "payment_number": [0]})
    assert extra_inflation_factor(payments, -0.01, 5).iloc[0] < 1


@pytest.mark.parametrize("stress_id, rate", [("rates_down", 0.025), ("rates_up", 0.045)])
def test_rate_stress_obligation_matches_the_dbo_at_that_rate(results, valid_employees, stress_id,
                                                             rate):
    expected = total_dbo(valid_employees, VALUATION_DATE, dict(ASSUMPTIONS, discount_rate=rate))
    assert pick(results, stress_id, "a_short_bonds")["obligation"] == pytest.approx(expected,
                                                                                    abs=1e-6)


def test_longer_lives_obligation_matches_the_dbo(results, valid_employees):
    expected = total_dbo(valid_employees, VALUATION_DATE, dict(ASSUMPTIONS, mortality_factor=0.9))
    assert pick(results, "longer_lives", "b_long_bonds")["obligation"] == pytest.approx(expected,
                                                                                        abs=1e-6)


def test_directions_by_hand(results):
    for portfolio_id in ("a_short_bonds", "b_long_bonds", "c_growth_mix"):
        assert pick(results, "rates_down", portfolio_id)["obligation"] > BASE_DBO
        assert pick(results, "rates_down", portfolio_id)["assets"] > BASE_DBO
        assert pick(results, "rates_up", portfolio_id)["obligation"] < BASE_DBO
        assert pick(results, "rates_up", portfolio_id)["assets"] < BASE_DBO
        assert pick(results, "inflation_up", portfolio_id)["obligation"] > BASE_DBO
        assert pick(results, "longer_lives", portfolio_id)["obligation"] > BASE_DBO
        assert pick(results, "equity_fall", portfolio_id)["assets"] < BASE_DBO
        assert pick(results, "equity_fall", portfolio_id)["obligation"] == pytest.approx(BASE_DBO,
                                                                                         abs=0.01)


def test_obligation_only_stresses_leave_assets_alone(results):
    for stress_id in ("inflation_up", "longer_lives"):
        rows = results[results["stress_id"] == stress_id]
        assert rows["assets"].to_numpy() == pytest.approx(BASE_DBO, abs=0.01)


def test_portfolio_a_has_no_equity_loss(results):
    # A holds no equities, so its whole loss in this stress comes from credit spreads
    row = pick(results, "equity_fall", "a_short_bonds")
    assert row["main_driver"] == "Credit spreads"


def test_single_risk_stresses_have_the_only_possible_driver(results):
    expected = {"rates_down": "Interest rates", "rates_up": "Interest rates",
                "inflation_up": "Inflation", "longer_lives": "Longevity"}
    for stress_id, driver in expected.items():
        rows = results[results["stress_id"] == stress_id]
        assert set(rows["main_driver"]) == {driver}


def test_main_driver_is_the_largest_single_effect(results, config, valid_employees):
    # Recalculate one combined case by running each risk as its own stress
    combined = values(config, "stresses")["combined_bad_case"]
    test_config = copy.deepcopy(config)
    single = {}
    for name, part in single_risk_stresses(combined).items():
        single[name] = {"value": {"label": name, **part}, "status": "illustrative", "source": "t",
                        "date": "t", "reason": "t"}
    test_config["stresses"] = single
    parts = run_stress_tests(valid_employees, test_config)
    for portfolio_id, rows in parts.groupby("portfolio_id"):
        largest = rows.loc[rows["gap"].abs().idxmax(), "stress"]
        assert pick(results, "combined_bad_case", portfolio_id)["main_driver"] == largest


def test_combined_case_contains_four_risks(config):
    combined = values(config, "stresses")["combined_bad_case"]
    assert len(single_risk_stresses(combined)) == 4
    assert set(single_risk_stresses(combined)) <= set(RISK_NAMES.values())


def test_starting_funding_level_changes_assets_not_obligation(config, valid_employees, results):
    changed = copy.deepcopy(config)
    changed["plan"]["starting_funding_level"]["value"] = 0.8
    lower = run_stress_tests(valid_employees, changed)
    assert np.allclose(lower["assets"], results["assets"] * 0.8)
    assert np.allclose(lower["obligation"], results["obligation"])


def test_wording_is_hypothetical_not_a_recommendation(config):
    for entry in config["stresses"].values():
        assert "Hypothetical" in entry["reason"]
