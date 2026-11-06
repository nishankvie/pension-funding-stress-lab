"""Risk measures must match small hand calculations and behave sensibly on real runs."""

import copy

import numpy as np
import pytest

from src.pension_lab.config import load_config
from src.pension_lab.drivers import (
    largest_drivers,
    outcome_range,
    risk_driver_table,
)
from src.pension_lab.risk_metrics import (
    MEASURES,
    average_of_lowest_outcomes,
    chance_below_any_time,
    chance_below_at_horizon,
    funding_at_risk,
    funding_bands,
    risk_summary,
    stability_table,
)
from src.pension_lab.simulation import build_payment_book, simulate

# Four paths, three dates. Path 2 dips below 90% and recovers; path 3 ends below
FUNDING = np.array([
    [1.0, 1.1, 1.2],
    [1.0, 0.8, 1.0],
    [1.0, 0.95, 0.85],
    [1.0, 1.0, 1.0],
])


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def book(valid_employees, config):
    return build_payment_book(valid_employees, config)


@pytest.fixture(scope="module")
def run(book, config):
    return simulate(book, config)


@pytest.fixture(scope="module")
def drivers(book, config):
    return risk_driver_table(book, config)


def test_chance_at_horizon_by_hand():
    assert chance_below_at_horizon(FUNDING, 0.90) == 0.25


def test_chance_at_any_time_counts_a_dip_that_recovers():
    assert chance_below_any_time(FUNDING, 0.90) == 0.50


def test_exactly_at_the_floor_is_not_below_it():
    assert chance_below_at_horizon(np.array([[1.0, 0.90]]), 0.90) == 0


def test_starting_below_the_floor_alone_does_not_count_as_a_fall():
    assert chance_below_any_time(np.array([[0.85, 0.95, 0.95]]), 0.90) == 0


def test_funding_at_risk_by_hand():
    funding = np.column_stack([np.ones(101), 1 + np.arange(-50, 51) / 100])
    # Changes run from -0.50 to +0.50 in steps of 0.01, so the 5th percentile is -0.45
    assert funding_at_risk(funding, 0.05) == pytest.approx(-0.45)


def test_average_of_lowest_outcomes_by_hand():
    funding = np.column_stack([np.ones(100), np.arange(1, 101) / 100])
    # Lowest 5 of 0.01 to 1.00: average of 0.01, 0.02, 0.03, 0.04, 0.05
    assert average_of_lowest_outcomes(funding, 0.05) == pytest.approx(0.03)


def test_outcome_range_by_hand():
    funding = np.column_stack([np.ones(101), np.arange(0, 101) / 100])
    assert outcome_range(funding) == pytest.approx(0.8)


def test_bands_cover_every_year_and_stay_in_order(run):
    bands = funding_bands(run)
    assert len(bands) == 3 * 11
    assert (bands["p10"] <= bands["p50"]).all()
    assert (bands["p50"] <= bands["p90"]).all()


def test_bands_start_together_at_the_funding_input(run):
    start = funding_bands(run).query("year == 0")
    assert start[["p10", "p50", "p90"]].to_numpy() == pytest.approx(1.0)


def test_any_time_chance_is_at_least_the_horizon_chance(run, config):
    summary = risk_summary(run, config)
    assert (summary["chance_below_floor_any_time"]
            >= summary["chance_below_floor_at_horizon"]).all()


def test_lowest_outcomes_sit_below_p10(run, config):
    summary = risk_summary(run, config)
    assert (summary["average_of_lowest_outcomes"] < summary["p10"]).all()


def test_changing_the_floor_changes_the_result(run, config):
    lower = copy.deepcopy(config)
    lower["attention"]["funding_floor"]["value"] = 0.70
    first = risk_summary(run, config)["chance_below_floor_any_time"]
    second = risk_summary(run, lower)["chance_below_floor_any_time"]
    assert (second < first).all()


def test_no_risk_means_no_chance_of_falling_below_the_floor(book, config):
    calm = simulate(book, config, paths=10, active_risks=())
    summary = risk_summary(calm, config)
    assert (summary["chance_below_floor_any_time"] == 0).all()
    # With no risk, A and B earn the 3.5% discount rate, so funding stays still. C earns 5%,
    # so its funding rises by 1.05 / 1.035 - 1 in the first year
    expected = [0.0, 0.0, 1.05 / 1.035 - 1]
    assert summary["funding_at_risk_1y"].to_numpy() == pytest.approx(expected, abs=1e-9)


def test_every_portfolio_has_a_range_for_each_risk(drivers):
    assert len(drivers) == 3 * 6


def test_equity_risk_cannot_affect_a_portfolio_with_no_equities(drivers):
    row = drivers[(drivers["risk"] == "Equity prices")
                  & (drivers["portfolio_id"] == "a_short_bonds")]
    assert row["outcome_range"].iloc[0] == pytest.approx(0, abs=1e-9)


def test_spread_risk_cannot_change_the_obligation(book, config):
    run = simulate(book, config, paths=200, active_risks=("spread",))
    assert run["obligation"].std(axis=0) == pytest.approx(0, abs=1e-6)


def test_portfolios_a_and_b_share_longevity_exposure(drivers):
    # Their base returns are identical, so longevity alone must spread them almost equally
    rows = drivers[drivers["risk"] == "Longevity"].set_index("portfolio_id")["outcome_range"]
    assert rows["a_short_bonds"] == pytest.approx(rows["b_long_bonds"], rel=0.001)


def test_matching_duration_narrows_the_rate_range(drivers):
    rows = drivers[drivers["risk"] == "Interest rates"].set_index("portfolio_id")["outcome_range"]
    assert rows["b_long_bonds"] < rows["a_short_bonds"]


def test_largest_driver_is_the_widest_single_risk(drivers):
    chosen = largest_drivers(drivers)
    singles = drivers[drivers["risk"] != "All risks"]
    for portfolio_id, risk in chosen.items():
        rows = singles[singles["portfolio_id"] == portfolio_id]
        widest = rows.loc[rows["outcome_range"].idxmax(), "risk"]
        assert risk == widest


def test_all_risks_is_never_picked_as_a_single_driver(drivers):
    assert "All risks" not in largest_drivers(drivers).values()


def test_stability_table_covers_every_measure(book, config):
    table = stability_table(book, config)
    assert len(table) == 3 * len(MEASURES)
    assert table["difference"].ge(0).all()


def test_results_are_stable_between_path_counts(book, config):
    table = stability_table(book, config)
    assert table["within_limit"].all(), table[~table["within_limit"]].to_string()
