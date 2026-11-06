"""Linked yearly simulation of the economy, the obligation and each portfolio's assets."""

import numpy as np

from src.pension import SENSITIVITY_SHIFT, duration, sensitivity_table
from src.pension_lab.cashflows import employee_payment_table
from src.pension_lab.config import values
from src.pension_lab.economy import check_correlation_matrix, correlation_matrix
from src.pension_lab.portfolios import (
    bond_return,
    build_portfolios,
    corporate_bond_return,
    equity_return,
    portfolio_return,
    year_end_assets,
    zero_coupon_convexity,
)

ALL_RISKS = ("rate", "inflation", "spread", "equity", "longevity")
RATE, INFLATION, SPREAD, EQUITY = 0, 1, 2, 3  # Positions in RISK_FACTORS


def build_payment_book(valid_df, config):
    """Work out every payment once per mortality level, so paths only look values up."""
    pension = values(config, "pension")
    plan = values(config, "plan")
    grid = values(config, "economy")["mortality_factor_grid"]

    amounts = []
    for factor in grid:
        table = employee_payment_table(
            valid_df, a=dict(pension, mortality_factor=factor),
            turnover_rate=plan["turnover_rate"],
            leaver_benefit_share=plan["leaver_benefit_share"],
            indexation_rate=plan["indexation_rate"],
        )
        amounts.append(table["expected_payment"].to_numpy())

    employee_codes, employee_ids = table["employee_id"].factorize()
    years_to_retirement = (table["years_from_now"] - table["payment_number"]).to_numpy()
    retirement_times = np.zeros(len(employee_ids))
    retirement_times[employee_codes] = years_to_retirement

    return {
        "grid": np.array(grid),
        "amounts": np.array(amounts),          # One row per mortality level
        "times": table["years_from_now"].to_numpy(),
        "years": table["simulation_year"].to_numpy(),
        "employee": employee_codes,
        "retirement_times": retirement_times,  # One value per employee
        "obligation_duration": duration(sensitivity_table(valid_df, a=pension)),
    }


def correlated_shocks(rng, paths, years, matrix):
    """Return linked standard normal shocks, shaped (paths, years, risk factors)."""
    check_correlation_matrix(matrix)
    independent = rng.standard_normal((paths, years, matrix.shape[0]))
    # Cholesky gives L with L x L-transpose = matrix, so multiplying independent shocks by L
    # produces shocks with exactly those correlations on average
    return independent @ np.linalg.cholesky(matrix).T


def economic_paths(economy, start_rate, shocks, active_risks):
    """Return yearly rate, inflation, spread and equity return paths."""
    paths, years, _ = shocks.shape
    rate = np.empty((paths, years + 1))
    inflation = np.empty((paths, years + 1))
    spread = np.empty((paths, years + 1))
    rate[:, 0] = start_rate
    inflation[:, 0] = economy["starting_inflation"]
    spread[:, 0] = economy["starting_spread"]

    rate_volatility = economy["rate_volatility"] if "rate" in active_risks else 0.0
    inflation_volatility = economy["inflation_volatility"] if "inflation" in active_risks else 0.0
    spread_volatility = economy["spread_volatility"] if "spread" in active_risks else 0.0
    equity_volatility = economy["equity_volatility"] if "equity" in active_risks else 0.0

    for t in range(years):
        rate[:, t + 1] = (rate[:, t]
                          + economy["rate_mean_reversion"] * (economy["long_run_rate"] - rate[:, t])
                          + rate_volatility * shocks[:, t, RATE])
        inflation[:, t + 1] = (inflation[:, t]
                               + economy["inflation_mean_reversion"]
                               * (economy["long_run_inflation"] - inflation[:, t])
                               + inflation_volatility * shocks[:, t, INFLATION])
        next_spread = (spread[:, t]
                       + economy["spread_mean_reversion"]
                       * (economy["long_run_spread"] - spread[:, t])
                       + spread_volatility * shocks[:, t, SPREAD])
        spread[:, t + 1] = np.maximum(next_spread, economy["spread_floor"])

    equity = equity_return(shocks[:, :, EQUITY], economy["equity_expected_return"],
                           equity_volatility)
    return rate, inflation, spread, equity


def cumulative_extra_inflation(inflation, base_inflation):
    """Return the log of total inflation above base up to each year, starting from 0."""
    log_extra = np.log1p(inflation[:, 1:] - base_inflation)
    start = np.zeros((inflation.shape[0], 1))
    return np.concatenate([start, np.cumsum(log_extra, axis=1)], axis=1)


def employee_inflation_factors(log_index, retirement_times, year):
    """Return each employee's payment scaling: extra inflation until they retire, or until now."""
    # Later inflation is unknown at this point, so it is assumed to return to base
    times = np.minimum(retirement_times, year)
    lower = np.floor(times).astype(int)
    upper = np.minimum(lower + 1, log_index.shape[1] - 1)
    fraction = times - lower
    return np.exp(log_index[:, lower] * (1 - fraction) + log_index[:, upper] * fraction)


def obligation_at_year(book, year, rates, mortality_index, inflation_factors):
    """Return each path's value of all payments due after this year, at that path's rate."""
    remaining = book["years"] > year
    obligation = np.zeros(len(rates))
    if not remaining.any():
        return obligation

    times_left = book["times"][remaining] - year
    employee = book["employee"][remaining]
    employee_count = len(book["retirement_times"])
    # Each column adds up one employee's payments, so scaling can then be applied per employee
    by_employee = np.zeros((remaining.sum(), employee_count))
    by_employee[np.arange(remaining.sum()), employee] = 1.0

    for level in range(len(book["grid"])):
        on_level = mortality_index == level
        if not on_level.any():
            continue
        discount = np.exp(-np.log1p(rates[on_level])[:, None] * times_left)
        weights = book["amounts"][level][remaining][:, None] * by_employee
        obligation[on_level] = ((discount @ weights) * inflation_factors[on_level]).sum(axis=1)
    return obligation


def remaining_duration(book, base_level, rate, year):
    """Return the duration of the payments still to come, measured the same way as the DBO's."""
    remaining = book["years"] > year
    if not remaining.any():
        return 0.0
    amounts = book["amounts"][base_level][remaining]
    times_left = book["times"][remaining] - year

    def value(discount_rate):
        return (amounts * (1 + discount_rate) ** -times_left).sum()

    shift = SENSITIVITY_SHIFT
    return (value(rate - shift) - value(rate + shift)) / (2 * value(rate) * shift)


def yearly_bond_durations(portfolio, matches_obligation, book, base_level, rate, years):
    """Return the bond duration for each year; a matching portfolio is reset every year."""
    if not matches_obligation:
        return np.full(years, float(portfolio["bond_duration"]))
    return np.array([remaining_duration(book, base_level, rate, year) for year in range(years)])


def simulate(book, config, paths=None, seed=None, active_risks=ALL_RISKS, shocks=None,
             mortality_index=None):
    """Return yearly rates, obligations and assets for every path and portfolio."""
    pension = values(config, "pension")
    economy = values(config, "economy")
    plan = values(config, "plan")
    settings = values(config, "simulation")
    portfolio_settings = values(config, "portfolios")
    paths = settings["paths"] if paths is None else paths
    years = settings["horizon_years"]
    start_rate = pension["discount_rate"]
    base_level = int(np.argmin(np.abs(book["grid"] - pension["mortality_factor"])))

    rng = np.random.default_rng(settings["seed"] if seed is None else seed)
    if shocks is None:
        matrix = correlation_matrix(values(config, "correlations"))
        shocks = correlated_shocks(rng, paths, years, matrix)
    paths = shocks.shape[0]
    if mortality_index is None:
        if "longevity" in active_risks:
            mortality_index = rng.choice(len(book["grid"]), size=paths,
                                         p=economy["mortality_factor_probabilities"])
        else:
            mortality_index = np.full(paths, base_level)

    rate, inflation, spread, equity = economic_paths(economy, start_rate, shocks, active_risks)
    log_index = cumulative_extra_inflation(inflation, economy["long_run_inflation"])

    # Every path starts from the same base obligation; its own mortality applies from year 1
    obligation = np.zeros((paths, years + 1))
    start_level = np.full(paths, base_level)
    start_factors = np.ones((paths, len(book["retirement_times"])))
    obligation[:, 0] = obligation_at_year(book, 0, rate[:, 0], start_level, start_factors)
    for year in range(1, years + 1):
        factors = employee_inflation_factors(log_index, book["retirement_times"], year)
        obligation[:, year] = obligation_at_year(book, year, rate[:, year], mortality_index,
                                                 factors)

    portfolios = build_portfolios(config, book["obligation_duration"])
    results = {}
    yield_rises = rate[:, 1:] - rate[:, :-1] + spread[:, 1:] - spread[:, :-1]
    wrong_way_share = {}
    for portfolio in portfolios:
        matches = portfolio_settings[portfolio["id"]]["bond_duration"] == "obligation"
        durations = yearly_bond_durations(portfolio, matches, book, base_level, start_rate, years)
        assets = np.zeros((paths, years + 1))
        assets[:, 0] = plan["starting_funding_level"] * obligation[:, 0]
        wrong_way = np.zeros((paths, years), dtype=bool)

        for year in range(1, years + 1):
            d = durations[year - 1]
            c = zero_coupon_convexity(d, start_rate)
            rate_change = rate[:, year] - rate[:, year - 1]
            spread_change = spread[:, year] - spread[:, year - 1]
            government = bond_return(rate[:, year - 1], d, c, rate_change)
            corporate = corporate_bond_return(rate[:, year - 1], spread[:, year - 1], d, c,
                                              rate_change, spread_change)
            total_return = portfolio_return(portfolio, government, corporate,
                                            equity[:, year - 1], portfolio_settings["yearly_cost"])

            due = book["years"] == year
            factors = employee_inflation_factors(log_index, book["retirement_times"], year)
            amounts = (book["amounts"][mortality_index][:, due]
                       * factors[:, book["employee"][due]])
            assets[:, year] = year_end_assets(assets[:, year - 1], total_return, amounts.T,
                                              year - book["times"][due],
                                              plan["yearly_contribution_eur"])
            if d > 0:
                wrong_way[:, year - 1] = yield_rises[:, year - 1] > d / c

        results[portfolio["id"]] = {
            "label": portfolio["label"],
            "assets": assets,
            "funding_level": assets / obligation,
            "bond_durations": durations,
        }
        wrong_way_share[portfolio["id"]] = float(wrong_way.mean())

    return {
        "rate": rate,
        "inflation": inflation,
        "spread": spread,
        "equity": equity,
        "mortality_factor": book["grid"][mortality_index],
        "obligation": obligation,
        "portfolios": results,
        "shocks": shocks,
        "checks": {
            "wrong_way_share": wrong_way_share,
            "spread_at_floor_share": float((spread[:, 1:] <= economy["spread_floor"]).mean()),
            "negative_rate_path_share": float((rate.min(axis=1) < 0).mean()),
            "lowest_assets": float(min(r["assets"].min() for r in results.values())),
        },
    }


if __name__ == "__main__":
    import time

    import pandas as pd

    from src.pension import DATA_DIR, validate_employees
    from src.pension_lab.config import load_config

    config = load_config()
    valid = validate_employees(pd.read_csv(DATA_DIR / "employees_raw.csv"))[0]
    book = build_payment_book(valid, config)

    start = time.time()
    run = simulate(book, config)
    print(f"5,000 paths in {time.time() - start:.1f} seconds")
    larger = simulate(book, config, paths=10000)
    print(f"Checks: {run['checks']}")
    print(f"Rate at year 10: average {run['rate'][:, -1].mean():.2%}, "
          f"volatility {run['rate'][:, -1].std():.2%}")
    print(f"Obligation at year 10: median EUR {np.median(run['obligation'][:, -1]):,.0f}")
    print("Median funding level at year 10, 5,000 and 10,000 paths:")
    for portfolio_id, result in run["portfolios"].items():
        small = np.median(result["funding_level"][:, -1])
        big = np.median(larger["portfolios"][portfolio_id]["funding_level"][:, -1])
        print(f"  {result['label']:<16} {small:.1%}  {big:.1%}")
    durations = run["portfolios"]["b_long_bonds"]["bond_durations"]
    print(f"Portfolio B duration by year: {np.round(durations, 2).tolist()}")
