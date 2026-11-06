"""The brief's rule: no simulation results are shown until these checks pass."""

import copy

import numpy as np
import pytest

from src.pension_lab.cashflows import employee_payment_table
from src.pension_lab.config import load_config, values
from src.pension_lab.economy import correlation_matrix
from src.pension_lab.portfolios import year_end_assets
from src.pension_lab.simulation import (
    ALL_RISKS,
    build_payment_book,
    correlated_shocks,
    cumulative_extra_inflation,
    employee_inflation_factors,
    remaining_duration,
    simulate,
)
from src.pension_lab.stress import extra_inflation_factor

BASE_DBO = 2_821_995.55
BASE_LEVEL = 2  # Position of mortality factor 1.00 in the grid
YEARS = 10


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def book(valid_employees, config):
    return build_payment_book(valid_employees, config)


@pytest.fixture(scope="module")
def run(book, config):
    return simulate(book, config, paths=5000)


def value_at(payments, year, rate, amounts=None):
    """Value payments due after a year, written out independently of the simulation code."""
    amounts = payments["expected_payment"] if amounts is None else amounts
    remaining = payments["simulation_year"] > year
    return (amounts[remaining] * (1 + rate) ** -(payments["years_from_now"][remaining] - year)).sum()


def scripted_shocks(config, paths=1):
    return np.zeros((paths, values(config, "simulation")["horizon_years"], 4))


# --- Zero volatility must equal the deterministic projection --------------------------------

def test_zero_volatility_obligation_matches_a_separate_projection(book, config, valid_employees):
    run = simulate(book, config, paths=3, active_risks=())
    payments = employee_payment_table(valid_employees)
    for year in range(YEARS + 1):
        expected = value_at(payments, year, 0.035)
        assert run["obligation"][:, year] == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("portfolio_id, yearly_return", [
    ("a_short_bonds", 0.035),
    ("b_long_bonds", 0.035),
    ("c_growth_mix", 0.05),
])
def test_zero_volatility_assets_match_a_separate_projection(book, config, valid_employees,
                                                            portfolio_id, yearly_return):
    run = simulate(book, config, paths=3, active_risks=())
    payments = employee_payment_table(valid_employees)
    assets = payments["present_value"].sum()
    for year in range(1, YEARS + 1):
        due = payments[payments["simulation_year"] == year]
        assets = year_end_assets(assets, yearly_return, due["expected_payment"],
                                 year - due["years_from_now"])
        assert run["portfolios"][portfolio_id]["assets"][:, year] == pytest.approx(assets, abs=1e-6)


def test_setting_every_volatility_to_zero_in_the_config_gives_the_same_answer(book, config):
    calm = copy.deepcopy(config)
    for name in ("rate_volatility", "inflation_volatility", "spread_volatility",
                 "equity_volatility"):
        calm["economy"][name]["value"] = 0.0
    calm["economy"]["mortality_factor_probabilities"]["value"] = [0.0, 0.0, 1.0, 0.0, 0.0]
    from_config = simulate(book, calm, paths=3)
    switched_off = simulate(book, config, paths=3, active_risks=())
    assert from_config["obligation"] == pytest.approx(switched_off["obligation"])


# --- Same seed, same result ------------------------------------------------------------------

def test_same_seed_gives_identical_results(book, config):
    first = simulate(book, config, paths=500, seed=7)
    second = simulate(book, config, paths=500, seed=7)
    assert np.array_equal(first["obligation"], second["obligation"])
    for portfolio_id in first["portfolios"]:
        assert np.array_equal(first["portfolios"][portfolio_id]["assets"],
                              second["portfolios"][portfolio_id]["assets"])


def test_different_seed_gives_different_results(book, config):
    first = simulate(book, config, paths=500, seed=7)
    second = simulate(book, config, paths=500, seed=8)
    assert not np.array_equal(first["obligation"], second["obligation"])


# --- Correlations and distributions match their targets --------------------------------------

def test_simulated_correlations_match_targets(config):
    matrix = correlation_matrix(values(config, "correlations"))
    shocks = correlated_shocks(np.random.default_rng(1), 20000, 10, matrix)
    measured = np.corrcoef(shocks.reshape(-1, 4), rowvar=False)
    assert measured == pytest.approx(matrix, abs=0.01)


def test_shocks_are_independent_from_year_to_year(config):
    matrix = correlation_matrix(values(config, "correlations"))
    shocks = correlated_shocks(np.random.default_rng(1), 20000, 2, matrix)
    assert np.corrcoef(shocks[:, 0, 0], shocks[:, 1, 0])[0, 1] == pytest.approx(0, abs=0.03)


def test_rate_spread_at_year_ten_matches_the_formula(run):
    # With pull k and yearly volatility v: v x sqrt((1 - (1-k)^20) / (1 - (1-k)^2))
    expected = 0.0075 * np.sqrt((1 - 0.9 ** 20) / (1 - 0.9 ** 2))
    assert run["rate"][:, -1].std() == pytest.approx(expected, rel=0.03)
    assert run["rate"][:, -1].mean() == pytest.approx(0.035, abs=0.001)


def test_equity_average_and_volatility_match_targets(run):
    assert run["equity"].mean() == pytest.approx(0.06, abs=0.003)
    assert run["equity"].std() == pytest.approx(0.16, abs=0.003)


def test_longevity_is_drawn_once_per_path_with_the_right_chances(run):
    factors, counts = np.unique(run["mortality_factor"], return_counts=True)
    assert factors.tolist() == [0.90, 0.95, 1.00, 1.05, 1.10]
    assert (counts / counts.sum()) == pytest.approx([0.1, 0.2, 0.4, 0.2, 0.1], abs=0.02)
    assert run["mortality_factor"].shape == (5000,)


def test_spread_never_falls_below_its_floor(run):
    assert run["spread"].min() >= 0


# --- Starting point --------------------------------------------------------------------------

def test_every_path_starts_at_the_input_funding_level(run):
    assert run["obligation"][:, 0] == pytest.approx(BASE_DBO, abs=0.01)
    for result in run["portfolios"].values():
        assert result["funding_level"][:, 0] == pytest.approx(1.0)


def test_starting_funding_level_input_is_respected(book, config):
    lower = copy.deepcopy(config)
    lower["plan"]["starting_funding_level"]["value"] = 0.85
    run = simulate(book, lower, paths=10)
    assert run["portfolios"]["a_short_bonds"]["funding_level"][:, 0] == pytest.approx(0.85)


# --- Scripted paths: each link checked against a separate calculation ------------------------

def test_scripted_rate_fall_revalues_the_obligation(book, config, valid_employees):
    shocks = scripted_shocks(config)
    shocks[0, 0, 0] = -0.01 / 0.0075  # Rates fall exactly 1 point in year 1
    run = simulate(book, config, shocks=shocks, mortality_index=np.array([BASE_LEVEL]))
    assert run["rate"][0, 1] == pytest.approx(0.025)
    payments = employee_payment_table(valid_employees)
    assert run["obligation"][0, 1] == pytest.approx(value_at(payments, 1, 0.025), abs=1e-6)


def test_scripted_longevity_uses_that_level_from_year_one(book, config, valid_employees):
    run = simulate(book, config, shocks=scripted_shocks(config), mortality_index=np.array([0]))
    assert run["obligation"][0, 0] == pytest.approx(BASE_DBO, abs=0.01)
    pension = values(config, "pension")
    payments = employee_payment_table(valid_employees, a=dict(pension, mortality_factor=0.9))
    assert run["obligation"][0, 1] == pytest.approx(value_at(payments, 1, 0.035), abs=1e-6)


def test_scripted_inflation_matches_the_stress_test_rule(book, config, valid_employees):
    # Inflation 1 point above base for 5 years, with no pull back, then back to base
    held = copy.deepcopy(config)
    held["economy"]["inflation_mean_reversion"]["value"] = 0.0
    shocks = scripted_shocks(config)
    shocks[0, 0, 1] = 0.01 / 0.015
    shocks[0, 5, 1] = -0.01 / 0.015
    run = simulate(book, held, shocks=shocks, mortality_index=np.array([BASE_LEVEL]))

    payments = employee_payment_table(valid_employees)
    scaled = payments["expected_payment"] * extra_inflation_factor(payments, 0.01, 5)
    for year in (5, 6, 10):
        assert run["obligation"][0, year] == pytest.approx(
            value_at(payments, year, 0.035, amounts=scaled), abs=1e-6)


def test_scripted_rate_fall_gives_hand_calculated_asset_return(book, config):
    # No payments fall in year 1. Income uses the rate at the START of the year.
    # Bonds, duration 5, convexity 5^2 + 5/1.035 = 29.8309:
    # price effect = +5% + 0.5 x 29.8309 x 0.01^2 = +5.14915%
    # Government 3.5% + 5.14915%, corporate 4.5% + 5.14915%, cost 0.2%
    shocks = scripted_shocks(config)
    shocks[0, 0, 0] = -0.01 / 0.0075
    run = simulate(book, config, shocks=shocks, mortality_index=np.array([BASE_LEVEL]))
    expected = 0.8 * 0.0864915 + 0.2 * 0.0964915 - 0.002
    assets = run["portfolios"]["a_short_bonds"]["assets"][0]
    assert assets[1] / assets[0] - 1 == pytest.approx(expected, abs=1e-6)


def test_scripted_spread_rise_hits_only_corporate_bonds_by_hand(book, config):
    # Spread +1 point. Corporate: 4.5% income, -5% + 0.149155% price. Government: 3.5% income
    shocks = scripted_shocks(config)
    shocks[0, 0, 2] = 0.01 / 0.005
    run = simulate(book, config, shocks=shocks, mortality_index=np.array([BASE_LEVEL]))
    expected = 0.8 * 0.035 + 0.2 * (0.045 - 0.05 + 0.00149155) - 0.002
    assets = run["portfolios"]["a_short_bonds"]["assets"][0]
    assert assets[1] / assets[0] - 1 == pytest.approx(expected, abs=1e-6)


def test_income_in_the_year_after_a_rate_fall_uses_the_lower_rate(book, config):
    # Year 2 starts at 2.5% and the pull back adds 0.1 x 1 point = +0.1 point during the year.
    # Portfolio C has no payments until year 3, so its year 2 return can be checked directly
    shocks = scripted_shocks(config)
    shocks[0, 0, 0] = -0.01 / 0.0075
    run = simulate(book, config, shocks=shocks, mortality_index=np.array([BASE_LEVEL]))
    assert run["rate"][0, 2] == pytest.approx(0.026)
    # Duration 6, convexity 36 + 6/1.035 = 41.7971; price = -0.6% + 0.5 x 41.7971 x 0.001^2
    price = -0.006 + 0.5 * 41.7971 * 0.001 ** 2
    # A zero equity shock gives the median, not the 6% average: 1.06 / sqrt(1 + 0.16^2 / 1.06^2) - 1
    median_equity = 1.06 / np.sqrt(1 + 0.16 ** 2 / 1.06 ** 2) - 1
    assert median_equity == pytest.approx(0.04813, abs=0.00001)
    expected = 0.2 * (0.025 + price) + 0.2 * (0.035 + price) + 0.6 * median_equity - 0.002
    assets = run["portfolios"]["c_growth_mix"]["assets"][0]
    assert assets[2] / assets[1] - 1 == pytest.approx(expected, abs=1e-5)


def test_scripted_equity_shock_moves_only_the_equity_part(book, config):
    shocks = scripted_shocks(config)
    shocks[0, 0, 3] = -2.0
    run = simulate(book, config, shocks=shocks, mortality_index=np.array([BASE_LEVEL]))
    equity = run["equity"][0, 0]
    assert equity < -0.2
    expected = 0.2 * 0.035 + 0.2 * 0.045 + 0.6 * equity - 0.002
    assets = run["portfolios"]["c_growth_mix"]["assets"][0]
    assert assets[1] / assets[0] - 1 == pytest.approx(expected, abs=1e-9)


def test_inflation_factor_by_hand():
    log_index = cumulative_extra_inflation(np.array([[0.02, 0.03, 0.03]]), 0.02)
    # One person retires in 1.5 years, one in 10. Now at year 2
    factors = employee_inflation_factors(log_index, np.array([1.5, 10.0]), 2)
    assert factors[0].tolist() == pytest.approx([1.01 ** 1.5, 1.01 ** 2])


def test_payments_are_scaled_when_they_leave_assets(book, config):
    held = copy.deepcopy(config)
    held["economy"]["inflation_mean_reversion"]["value"] = 0.0
    shocks = scripted_shocks(config)
    shocks[0, 0, 1] = 0.01 / 0.015
    higher = simulate(book, held, shocks=shocks, mortality_index=np.array([BASE_LEVEL]))
    calm = simulate(book, held, shocks=scripted_shocks(config),
                    mortality_index=np.array([BASE_LEVEL]))
    a = "a_short_bonds"
    assert higher["portfolios"][a]["assets"][0, -1] < calm["portfolios"][a]["assets"][0, -1]


# --- Directions ------------------------------------------------------------------------------

def test_higher_rate_lowers_obligation_across_paths(book, config):
    run = simulate(book, config, paths=2000, active_risks=("rate",))
    assert np.corrcoef(run["rate"][:, 5], run["obligation"][:, 5])[0, 1] < -0.95


def test_lower_mortality_raises_obligation(book, config):
    run = simulate(book, config, paths=2000, active_risks=("longevity",))
    obligation = run["obligation"][:, 5]
    factor = run["mortality_factor"]
    assert obligation[factor == 0.90].min() > obligation[factor == 1.10].max()


def test_rate_only_run_keeps_other_risks_still(book, config):
    run = simulate(book, config, paths=100, active_risks=("rate",))
    assert run["inflation"].std() == pytest.approx(0)
    assert run["spread"].std() == pytest.approx(0)
    assert run["equity"].std() == pytest.approx(0, abs=1e-12)
    assert set(run["mortality_factor"]) == {1.0}


def test_all_risks_are_on_by_default():
    assert set(ALL_RISKS) == {"rate", "inflation", "spread", "equity", "longevity"}


# --- Portfolio B -----------------------------------------------------------------------------

def test_portfolio_b_starts_at_the_obligation_duration_and_is_reset(run, book):
    durations = run["portfolios"]["b_long_bonds"]["bond_durations"]
    assert durations[0] == pytest.approx(book["obligation_duration"], abs=1e-9)
    assert durations[5] == pytest.approx(remaining_duration(book, BASE_LEVEL, 0.035, 5))
    assert len(set(durations.round(6))) == YEARS


def test_other_portfolios_keep_their_duration(run):
    assert set(run["portfolios"]["a_short_bonds"]["bond_durations"]) == {5.0}
    assert set(run["portfolios"]["c_growth_mix"]["bond_durations"]) == {6.0}


# --- Limits of the approximation stay rare ---------------------------------------------------

def test_bond_formula_rarely_leaves_its_safe_range(run):
    assert run["checks"]["wrong_way_share"]["b_long_bonds"] < 0.001


def test_assets_never_run_out_within_the_horizon(run):
    assert run["checks"]["lowest_assets"] > 0
