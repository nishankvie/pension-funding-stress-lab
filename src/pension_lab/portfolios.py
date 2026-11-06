"""Three illustrative sample portfolios and their yearly returns. These are not products."""

import numpy as np

from src.pension_lab.config import values

PORTFOLIO_LABELS = {
    "a_short_bonds": "A: Short bonds",
    "b_long_bonds": "B: Long bonds",
    "c_growth_mix": "C: Growth mix",
}


def zero_coupon_convexity(duration, bond_yield):
    """Return the curvature of a single payment with this duration."""
    return duration ** 2 + duration / (1 + bond_yield)


def build_portfolios(config, obligation_duration):
    """Return the three portfolios, setting Portfolio B's duration to the obligation's."""
    settings = values(config, "portfolios")
    starting_yield = values(config, "pension")["discount_rate"]

    portfolios = []
    for portfolio_id, label in PORTFOLIO_LABELS.items():
        mix = settings[portfolio_id]
        duration = mix["bond_duration"]
        if duration == "obligation":
            duration = obligation_duration
        portfolios.append({
            "id": portfolio_id,
            "label": label,
            "government": mix["government"],
            "corporate": mix["corporate"],
            "equity": mix["equity"],
            "bond_duration": duration,
            "bond_convexity": zero_coupon_convexity(duration, starting_yield),
        })
    return portfolios


def rate_sensitivity_covered(portfolio, obligation_duration):
    """Return the share of the obligation's rate sensitivity the portfolio's bonds cover."""
    bonds = portfolio["government"] + portfolio["corporate"]
    return bonds * portfolio["bond_duration"] / obligation_duration


def bond_return(starting_yield, duration, convexity, yield_change):
    """Return a bond's yearly return: its income, plus the price effect of the yield change."""
    return starting_yield - duration * yield_change + 0.5 * convexity * yield_change ** 2


def corporate_bond_return(starting_rate, starting_spread, duration, convexity, rate_change,
                          spread_change):
    """Return a corporate bond's return: the same formula, on the rate plus the spread."""
    return bond_return(starting_rate + starting_spread, duration, convexity,
                       rate_change + spread_change)


def equity_return(shock, expected_return, volatility):
    """Return a yearly equity return from a standard normal shock (lognormal distribution)."""
    # These two lines choose the log-scale values that give exactly the stated average and
    # volatility. A shock of 0 therefore gives the median, which is a little below the average
    log_variance = np.log(1 + volatility ** 2 / (1 + expected_return) ** 2)
    log_mean = np.log(1 + expected_return) - log_variance / 2
    return np.exp(log_mean + np.sqrt(log_variance) * shock) - 1


def portfolio_return(portfolio, government_return, corporate_return, equity_yearly_return,
                     yearly_cost):
    """Return the yearly return. Rebalancing means the same weights apply at each year's start."""
    return (portfolio["government"] * government_return
            + portfolio["corporate"] * corporate_return
            + portfolio["equity"] * equity_yearly_return
            - yearly_cost)


def year_end_assets(assets, yearly_return, payment_amounts, years_left, contribution=0.0):
    """Return assets after one year. Each payment stops earning from the day it is paid."""
    growth = 1 + np.asarray(yearly_return, dtype=float)
    paid = 0.0
    for amount, left in zip(payment_amounts, years_left):
        paid = paid + amount * growth ** left
    return assets * growth + contribution - paid


if __name__ == "__main__":
    import pandas as pd

    from src.pension import DATA_DIR, duration, sensitivity_table, validate_employees
    from src.pension_lab.cashflows import employee_payment_table, present_value
    from src.pension_lab.config import load_config

    config = load_config()
    economy = values(config, "economy")
    pension = values(config, "pension")
    cost = values(config, "portfolios")["yearly_cost"]
    rate = pension["discount_rate"]
    spread = economy["starting_spread"]

    valid = validate_employees(pd.read_csv(DATA_DIR / "employees_raw.csv"))[0]
    payments = employee_payment_table(valid, a=pension)
    obligation_duration = duration(sensitivity_table(valid, a=pension))

    base = present_value(payments, rate)
    step = 0.005
    curvature = (present_value(payments, rate - step) + present_value(payments, rate + step)
                 - 2 * base) / (base * step ** 2)
    print(f"Obligation duration {obligation_duration:.2f}, convexity {curvature:.1f}")
    print(f"Obligation value if rates move 1 point: "
          f"down {present_value(payments, rate - 0.01) / base - 1:+.2%}, "
          f"up {present_value(payments, rate + 0.01) / base - 1:+.2%}")
    print()

    for portfolio in build_portfolios(config, obligation_duration):
        d, c = portfolio["bond_duration"], portfolio["bond_convexity"]

        def one_year(rate_change):
            government = bond_return(rate, d, c, rate_change)
            corporate = corporate_bond_return(rate, spread, d, c, rate_change, 0.0)
            return portfolio_return(portfolio, government, corporate,
                                    economy["equity_expected_return"], cost)

        print(f"{portfolio['label']:<16} gov {portfolio['government']:.0%}, "
              f"corp {portfolio['corporate']:.0%}, equity {portfolio['equity']:.0%}, "
              f"bond duration {d:.2f}, convexity {c:.1f}")
        print(f"  share of obligation's rate sensitivity covered "
              f"{rate_sensitivity_covered(portfolio, obligation_duration):.0%}")
        print(f"  one-year return: rates unchanged {one_year(0):+.2%}, "
              f"down 1 point {one_year(-0.01):+.2%}, up 1 point {one_year(0.01):+.2%}")
        turn = d / c
        print(f"  bond formula turns the wrong way beyond a yield rise of {turn:.1%}")
