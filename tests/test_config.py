"""The config must be complete, honestly labelled and consistent with the pension code."""

import copy

import numpy as np
import pytest

from src.pension import ASSUMPTIONS, dbo_portfolio
from src.pension_lab.config import assumptions_table, check_config, load_config, values
from src.pension_lab.economy import check_correlation_matrix, correlation_matrix


@pytest.fixture(scope="module")
def config():
    return load_config()


def test_every_entry_has_label_source_date_and_reason(config):
    rows = assumptions_table(config)
    assert len(rows) > 50
    for row in rows:
        assert row["status"] in ("estimated", "illustrative")
        assert row["source"] and row["reason"] and row["date"]


def test_entry_without_a_reason_is_refused(config):
    broken = copy.deepcopy(config)
    del broken["economy"]["rate_volatility"]["reason"]
    with pytest.raises(ValueError, match="rate_volatility"):
        check_config(broken)


def test_unknown_status_is_refused(config):
    broken = copy.deepcopy(config)
    broken["economy"]["rate_volatility"]["status"] = "guess"
    with pytest.raises(ValueError, match="guess"):
        check_config(broken)


def test_pension_values_equal_the_code_that_produced_the_dbo(config):
    assert values(config, "pension") == ASSUMPTIONS


def test_dbo_from_config_still_matches(config, valid_employees):
    total = dbo_portfolio(valid_employees, a=values(config, "pension"))[1]
    assert total == pytest.approx(2_821_995.55, abs=0.01)


def test_base_case_options_are_off(config):
    plan = values(config, "plan")
    assert plan["turnover_rate"] == 0
    assert plan["indexation_rate"] == 0


def test_correlation_matrix_is_valid(config):
    matrix = correlation_matrix(values(config, "correlations"))
    check_correlation_matrix(matrix)
    np.linalg.cholesky(matrix)
    assert matrix.shape == (4, 4)


def test_contradictory_correlations_are_refused():
    # A moves with B, A moves with C, but B moves against C: impossible
    matrix = np.array([[1.0, 0.9, 0.9, 0.0], [0.9, 1.0, -0.9, 0.0],
                       [0.9, -0.9, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    with pytest.raises(ValueError, match="contradict"):
        check_correlation_matrix(matrix)


def test_mortality_grid_probabilities_sum_to_one_and_average_one(config):
    economy = values(config, "economy")
    grid = np.array(economy["mortality_factor_grid"])
    probabilities = np.array(economy["mortality_factor_probabilities"])
    assert probabilities.sum() == pytest.approx(1)
    assert grid @ probabilities == pytest.approx(1)


def test_portfolio_weights_sum_to_one(config):
    portfolios = values(config, "portfolios")
    for name in ("a_short_bonds", "b_long_bonds", "c_growth_mix"):
        mix = portfolios[name]
        assert mix["government"] + mix["corporate"] + mix["equity"] == pytest.approx(1)


def test_all_six_stresses_are_defined(config):
    assert len(config["stresses"]) == 6


def test_attention_thresholds_are_labelled_illustrative(config):
    for entry in config["attention"].values():
        assert entry["status"] == "illustrative"
