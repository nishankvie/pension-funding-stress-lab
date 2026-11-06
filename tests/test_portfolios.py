"""Portfolio returns must match small hand calculations and behave like real bonds."""

import numpy as np
import pytest
from numpy.polynomial.hermite_e import hermegauss

from src.pension import duration, sensitivity_table
from src.pension_lab.cashflows import employee_payment_table, present_value
from src.pension_lab.config import load_config, values
from src.pension_lab.portfolios import (
    PORTFOLIO_LABELS,
    bond_return,
    build_portfolios,
    corporate_bond_return,
    equity_return,
    portfolio_return,
    year_end_assets,
    zero_coupon_convexity,
)


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def portfolios(config, valid_employees):
    return build_portfolios(config, duration(sensitivity_table(valid_employees)))


def test_three_portfolios_use_the_brief_allocations(portfolios):
    assert [p["label"] for p in portfolios] == list(PORTFOLIO_LABELS.values())
    a, b, c = portfolios
    assert (a["government"], a["corporate"], a["equity"]) == (0.80, 0.20, 0.00)
    assert (b["government"], b["corporate"], b["equity"]) == (0.80, 0.20, 0.00)
    assert c["government"] + c["corporate"] == pytest.approx(0.40)
    assert c["equity"] == 0.60
    assert a["bond_duration"] == 5
    assert c["bond_duration"] == 6


def test_portfolio_b_duration_is_set_at_run_time(config):
    assert build_portfolios(config, 12.5)[1]["bond_duration"] == 12.5
    assert build_portfolios(config, 30.0)[1]["bond_duration"] == 30.0


def test_portfolios_carry_no_recommendation_wording(portfolios):
    for portfolio in portfolios:
        label = portfolio["label"].lower()
        assert "best" not in label and "recommended" not in label


def test_convexity_by_hand():
    # 5^2 + 5 / 1.035 = 25 + 4.8309
    assert zero_coupon_convexity(5, 0.035) == pytest.approx(29.8309, abs=0.0001)


def test_bond_return_by_hand():
    # 3.5% income, -5 x 1 point = -5%, +0.5 x 30 x 0.01^2 = +0.15%
    assert bond_return(0.035, 5, 30, 0.01) == pytest.approx(-0.0135)


def test_no_yield_change_leaves_only_income():
    assert bond_return(0.035, 22, 500, 0.0) == 0.035


def test_corporate_return_by_hand():
    # 4.5% income, -5 x 2 points = -10%, +0.5 x 30 x 0.02^2 = +0.6%
    assert corporate_bond_return(0.035, 0.01, 5, 30, 0.01, 0.01) == pytest.approx(-0.049)


def test_wider_spread_hurts_corporate_bonds():
    calm = corporate_bond_return(0.035, 0.01, 5, 30, 0.0, 0.0)
    wider = corporate_bond_return(0.035, 0.01, 5, 30, 0.0, 0.01)
    assert calm == pytest.approx(0.045)
    assert wider < calm


@pytest.mark.parametrize("years", [5, 10, 23])
@pytest.mark.parametrize("yield_change", [-0.01, 0.01])
def test_formula_is_close_to_exact_price_of_a_single_payment(years, yield_change):
    # A payment due in T years has an exact price, which needs no approximation
    y = 0.035
    d = years / (1 + y)
    exact = ((1 + y) / (1 + y + yield_change)) ** years - 1
    approximate = bond_return(0.0, d, zero_coupon_convexity(d, y), yield_change)
    assert approximate == pytest.approx(exact, abs=0.005)


def test_formula_never_loses_more_than_everything(portfolios):
    b = portfolios[1]
    changes = np.linspace(-0.15, 0.15, 3001)
    returns = bond_return(0.0, b["bond_duration"], b["bond_convexity"], changes)
    assert returns.min() > -1


def test_equity_average_and_volatility_are_exact():
    # Gauss-Hermite gives exact averages over a normal shock, with no random numbers
    shocks, weights = hermegauss(80)
    weights = weights / weights.sum()
    returns = equity_return(shocks, 0.06, 0.16)
    average = weights @ returns
    assert average == pytest.approx(0.06, abs=1e-9)
    assert np.sqrt(weights @ (returns - average) ** 2) == pytest.approx(0.16, abs=1e-6)


def test_equity_with_no_volatility_returns_the_expected_return():
    assert equity_return(1.7, 0.06, 0.0) == pytest.approx(0.06)


def test_equity_loss_stays_above_minus_100_percent():
    assert equity_return(-10.0, 0.06, 0.16) > -1


def test_portfolio_return_by_hand():
    mix = {"government": 0.8, "corporate": 0.2, "equity": 0.0}
    assert portfolio_return(mix, 0.02, 0.04, 0.50, 0.002) == pytest.approx(0.022)


def test_rebalancing_keeps_weights_fixed():
    # 50/50 with returns of +100% and 0%, over two years. Rebalanced: 1.5 x 1.5 = 2.25.
    # Left alone it would be 0.5 x 4 + 0.5 x 1 = 2.5, so this proves the weights were reset
    mix = {"government": 0.5, "corporate": 0.0, "equity": 0.5}
    yearly = portfolio_return(mix, 0.0, 0.0, 1.0, 0.0)
    assert (1 + yearly) ** 2 == pytest.approx(2.25)


def test_year_end_assets_by_hand():
    # 100 x 1.1 = 110, less 10 paid mid-year, which would have grown to 10 x 1.1^0.5
    assert year_end_assets(100.0, 0.10, [10.0], [0.5]) == pytest.approx(99.5119, abs=0.0001)


def test_contribution_is_added():
    assert year_end_assets(100.0, 0.0, [], [], contribution=5.0) == pytest.approx(105.0)


def test_many_paths_at_once():
    result = year_end_assets(np.array([100.0, 100.0]), np.array([0.1, 0.0]), [10.0], [0.0])
    assert result.tolist() == pytest.approx([100.0, 90.0])


@pytest.mark.parametrize("years", [1, 5, 10, 30])
def test_assets_earning_the_discount_rate_stay_equal_to_the_obligation(valid_employees, years):
    # The key link between assets and obligation: paying every payment from assets that earn
    # exactly the discount rate must leave exactly the value of the payments still to come
    rate = 0.035
    payments = employee_payment_table(valid_employees)
    assets = payments["present_value"].sum()

    for year in range(1, years + 1):
        due = payments[payments["simulation_year"] == year]
        assets = year_end_assets(assets, rate, due["expected_payment"],
                                 year - due["years_from_now"])

    remaining = payments[payments["simulation_year"] > years]
    obligation = (remaining["expected_payment"]
                  * (1 + rate) ** -(remaining["years_from_now"] - years)).sum()
    assert assets == pytest.approx(obligation, abs=1e-6)


def test_portfolio_b_tracks_the_obligation_better_than_a_when_rates_fall(portfolios, config,
                                                                         valid_employees):
    rate = 0.035
    payments = employee_payment_table(valid_employees)
    obligation_change = present_value(payments, rate - 0.01) / present_value(payments, rate) - 1
    a, b, _ = portfolios
    a_gain = bond_return(0.0, a["bond_duration"], a["bond_convexity"], -0.01)
    b_gain = bond_return(0.0, b["bond_duration"], b["bond_convexity"], -0.01)
    assert abs(obligation_change - b_gain) < abs(obligation_change - a_gain)
