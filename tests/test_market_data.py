"""Stored market data must be traceable, and labels must agree with what the data supports."""

import hashlib

import numpy as np
import pandas as pd
import pytest

from src.pension_lab.config import load_config
from src.pension_lab.economy import (
    calibration_table,
    correlation_row,
    mean_reversion_estimate,
    split_in_half,
    volatility_row,
    yearly_changes,
)
from src.pension_lab.market_data import (
    MARKET_DIR,
    SERIES,
    load_market_data,
    load_sources,
    month_end_values,
    parse_ecb_csv,
)


@pytest.fixture(scope="module")
def market():
    return load_market_data()


@pytest.fixture(scope="module")
def calibration(market):
    return calibration_table(market).set_index("parameter")


def test_every_series_records_source_date_and_unchanged_file():
    sources = load_sources()
    assert set(sources) == set(SERIES)
    for name, record in sources.items():
        assert record["url"].startswith("https://data-api.ecb.europa.eu/")
        assert record["download_date"]
        content = (MARKET_DIR / f"{name}.csv").read_bytes()
        assert hashlib.sha256(content).hexdigest() == record["sha256"]


def test_parse_keeps_only_date_and_value():
    text = "KEY,TIME_PERIOD,OBS_VALUE,OBS_STATUS\nX,2025-01,1.5,A\nX,2025-02,,M\n"
    table = parse_ecb_csv(text)
    assert table.columns.tolist() == ["date", "value"]
    assert table["value"].tolist() == [1.5]


def test_unfinished_month_is_left_out():
    table = pd.DataFrame({"date": ["2026-10-30", "2026-10-31", "2026-11-05"],
                          "value": [3.0, 3.1, 3.2]})
    monthly = month_end_values(table, "2026-11-06")
    assert monthly.tolist() == [3.1]


def test_rates_are_decimals_not_percentages(market):
    assert market["rate"].abs().max() < 0.10
    assert market["inflation"].abs().max() < 0.20


def test_volatility_by_hand():
    changes = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    row = volatility_row(changes, "x", "x")
    # Sample standard deviation of 1, 2, 3, 4 is the square root of 5/3
    assert row["full_sample"] == pytest.approx(1.2909944)
    assert row["first_half"] == pytest.approx(0.7071068)
    assert row["stable"]


def test_opposite_halves_are_not_stable():
    changes = pd.DataFrame({"a": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
                            "b": [1.0, 2.0, 3.0, 3.0, 2.0, 1.0]})
    row = correlation_row(changes, "a", "b", "ab")
    assert row["first_half"] == pytest.approx(1)
    assert row["second_half"] == pytest.approx(-1)
    assert not row["stable"]


def test_mean_reversion_recovers_a_known_process():
    level, b = 0.03, 0.95
    series = [0.05]
    for _ in range(50):
        series.append(level + b * (series[-1] - level))
    estimate = mean_reversion_estimate(pd.Series(series))
    assert estimate["yearly_speed"] == pytest.approx(1 - b ** 12, abs=1e-6)
    assert estimate["long_run_level"] == pytest.approx(level, abs=1e-6)


def test_halves_cover_all_the_data_without_overlap(market):
    changes = yearly_changes(market)
    first, second = split_in_half(changes)
    assert len(first) + len(second) == len(changes)
    assert first.index.max() < second.index.min()


def test_only_stable_parameters_are_labelled_estimated(calibration):
    config = load_config()
    entries = {**config["economy"], **config["correlations"]}
    for parameter, row in calibration.iterrows():
        assert entries[parameter]["status"] == ("estimated" if row["stable"] else "illustrative")


def test_estimated_values_equal_the_calculation_rounded(calibration):
    config = load_config()
    entry = config["correlations"]["rate_equity"]
    assert entry["status"] == "estimated"
    assert entry["value"] == round(calibration.loc["rate_equity", "full_sample"], 2)
