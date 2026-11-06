"""Turn simulated funding levels into the risk measures reported for each portfolio."""

import numpy as np
import pandas as pd

from src.pension_lab.config import values
from src.pension_lab.simulation import simulate

MEASURES = ["p10", "p50", "p90", "chance_below_floor_at_horizon", "chance_below_floor_any_time",
            "funding_at_risk_1y", "average_of_lowest_outcomes"]


def funding_bands(run):
    """Return the 10th, 50th and 90th percentile of funding level for each portfolio and year."""
    rows = []
    for portfolio_id, result in run["portfolios"].items():
        funding = result["funding_level"]
        p10, p50, p90 = np.percentile(funding, [10, 50, 90], axis=0)
        for year in range(funding.shape[1]):
            rows.append({"portfolio_id": portfolio_id, "portfolio": result["label"], "year": year,
                         "p10": p10[year], "p50": p50[year], "p90": p90[year]})
    return pd.DataFrame(rows)


def chance_below_at_horizon(funding, floor):
    """Return the share of paths below the floor in the final year."""
    return float((funding[:, -1] < floor).mean())


def chance_below_any_time(funding, floor):
    """Return the share of paths below the floor at any year end after the start."""
    return float((funding[:, 1:] < floor).any(axis=1).mean())


def funding_at_risk(funding, tail_share):
    """Return the low percentile of the change in funding level over the first year."""
    first_year_change = funding[:, 1] - funding[:, 0]
    return float(np.percentile(first_year_change, tail_share * 100))


def average_of_lowest_outcomes(funding, tail_share):
    """Return the average final funding level across the lowest share of paths."""
    final = np.sort(funding[:, -1])
    count = int(np.ceil(len(final) * tail_share))
    return float(final[:count].mean())


def risk_summary(run, config):
    """Return one row of risk measures per portfolio."""
    floor = values(config, "attention")["funding_floor"]
    tail_share = values(config, "simulation")["tail_share"]
    rows = []
    for portfolio_id, result in run["portfolios"].items():
        funding = result["funding_level"]
        p10, p50, p90 = np.percentile(funding[:, -1], [10, 50, 90])
        rows.append({
            "portfolio_id": portfolio_id,
            "portfolio": result["label"],
            "p10": p10,
            "p50": p50,
            "p90": p90,
            "chance_below_floor_at_horizon": chance_below_at_horizon(funding, floor),
            "chance_below_floor_any_time": chance_below_any_time(funding, floor),
            "funding_at_risk_1y": funding_at_risk(funding, tail_share),
            "average_of_lowest_outcomes": average_of_lowest_outcomes(funding, tail_share),
        })
    return pd.DataFrame(rows)


def stability_table(book, config):
    """Compare the default run with a larger run on a different seed."""
    settings = values(config, "simulation")
    limit = settings["stability_limit"]
    default = risk_summary(simulate(book, config), config).set_index("portfolio_id")
    larger_run = simulate(book, config, paths=settings["max_paths"], seed=settings["seed"] + 1)
    larger = risk_summary(larger_run, config).set_index("portfolio_id")

    rows = []
    for portfolio_id in default.index:
        for measure in MEASURES:
            first = default.loc[portfolio_id, measure]
            second = larger.loc[portfolio_id, measure]
            rows.append({
                "portfolio": default.loc[portfolio_id, "portfolio"],
                "measure": measure,
                "default_paths": first,
                "max_paths": second,
                "difference": abs(first - second),
                "within_limit": bool(abs(first - second) <= limit),
            })
    return pd.DataFrame(rows)
