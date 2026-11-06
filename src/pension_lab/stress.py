"""Six hypothetical stresses, each applied at once on the valuation date to every portfolio."""

import numpy as np
import pandas as pd

from src.pension import duration, sensitivity_table
from src.pension_lab.cashflows import employee_payment_table, present_value
from src.pension_lab.config import values
from src.pension_lab.portfolios import bond_return, build_portfolios

# Plain names for the shocks a stress can contain, used when naming the main driver
RISK_NAMES = {
    "rate_shift": "Interest rates",
    "inflation_shift": "Inflation",
    "mortality_factor": "Longevity",
    "equity_shock": "Equity prices",
    "spread_shift": "Credit spreads",
}


def extra_inflation_factor(payments, extra_inflation, inflation_years):
    """Return how much each payment rises when inflation runs above the base assumption."""
    # Earned pensions follow salary, so extra inflation only counts while someone is still
    # working. Pensions already being paid stay fixed unless indexation is switched on
    years_to_retirement = payments["years_from_now"] - payments["payment_number"]
    years_affected = np.minimum(years_to_retirement, inflation_years)
    return (1 + extra_inflation) ** years_affected


def obligation_after(payment_tables, stress, base_mortality, discount_rate):
    """Return the obligation after a stress, using the matching table of payments."""
    payments = payment_tables[stress.get("mortality_factor", base_mortality)]
    scaled = payments.copy()
    if "inflation_shift" in stress:
        scaled["expected_payment"] = scaled["expected_payment"] * extra_inflation_factor(
            payments, stress["inflation_shift"], stress["inflation_years"])
    return present_value(scaled, discount_rate + stress.get("rate_shift", 0.0))


def assets_after(starting_assets, portfolio, stress):
    """Return assets after an immediate shock: price changes only, with no income or cost."""
    rate_shift = stress.get("rate_shift", 0.0)
    spread_shift = stress.get("spread_shift", 0.0)
    d = portfolio["bond_duration"]
    c = portfolio["bond_convexity"]

    government_change = bond_return(0.0, d, c, rate_shift)
    corporate_change = bond_return(0.0, d, c, rate_shift + spread_shift)
    equity_change = stress.get("equity_shock", 0.0)

    total_change = (portfolio["government"] * government_change
                    + portfolio["corporate"] * corporate_change
                    + portfolio["equity"] * equity_change)
    return starting_assets * (1 + total_change)


def single_risk_stresses(stress):
    """Split a stress into one stress per risk it contains."""
    parts = {}
    for key in RISK_NAMES:
        if key in stress:
            part = {key: stress[key]}
            if key == "inflation_shift":
                part["inflation_years"] = stress["inflation_years"]
            parts[RISK_NAMES[key]] = part
    return parts


def main_driver(stress, portfolio, payment_tables, starting_assets, base_mortality, rate,
                base_gap):
    """Return the risk that moves the gap most when it is applied on its own."""
    moves = {}
    for name, part in single_risk_stresses(stress).items():
        gap = (assets_after(starting_assets, portfolio, part)
               - obligation_after(payment_tables, part, base_mortality, rate))
        moves[name] = abs(gap - base_gap)
    return max(moves, key=moves.get)


def build_payment_tables(valid_df, config):
    """Return the payments for the base mortality and each mortality level used in a stress."""
    pension = values(config, "pension")
    plan = values(config, "plan")
    factors = {pension["mortality_factor"]}
    for stress in values(config, "stresses").values():
        if "mortality_factor" in stress:
            factors.add(stress["mortality_factor"])

    tables = {}
    for factor in factors:
        tables[factor] = employee_payment_table(
            valid_df, a=dict(pension, mortality_factor=factor),
            turnover_rate=plan["turnover_rate"],
            leaver_benefit_share=plan["leaver_benefit_share"],
            indexation_rate=plan["indexation_rate"],
        )
    return tables


def run_stress_tests(valid_df, config):
    """Return one row per stress and portfolio: assets, obligation, gap, funding and main driver."""
    pension = values(config, "pension")
    rate = pension["discount_rate"]
    base_mortality = pension["mortality_factor"]
    payment_tables = build_payment_tables(valid_df, config)

    base_obligation = obligation_after(payment_tables, {}, base_mortality, rate)
    starting_assets = values(config, "plan")["starting_funding_level"] * base_obligation
    base_gap = starting_assets - base_obligation
    portfolios = build_portfolios(config, duration(sensitivity_table(valid_df, a=pension)))

    rows = []
    for stress_id, stress in values(config, "stresses").items():
        obligation = obligation_after(payment_tables, stress, base_mortality, rate)
        for portfolio in portfolios:
            assets = assets_after(starting_assets, portfolio, stress)
            rows.append({
                "stress_id": stress_id,
                "stress": stress["label"],
                "portfolio_id": portfolio["id"],
                "portfolio": portfolio["label"],
                "assets": assets,
                "obligation": obligation,
                "gap": assets - obligation,
                "funding_level": assets / obligation,
                "main_driver": main_driver(stress, portfolio, payment_tables, starting_assets,
                                           base_mortality, rate, base_gap),
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from src.pension import DATA_DIR, validate_employees
    from src.pension_lab.config import load_config

    config = load_config()
    valid = validate_employees(pd.read_csv(DATA_DIR / "employees_raw.csv"))[0]
    results = run_stress_tests(valid, config)
    for _, row in results.iterrows():
        print(f"{row['stress']:<42}{row['portfolio']:<17}assets {row['assets']:>10,.0f}  "
              f"obligation {row['obligation']:>10,.0f}  gap {row['gap']:>11,.0f}  "
              f"funded {row['funding_level']:>6.1%}  {row['main_driver']}")

    tables = build_payment_tables(valid, config)
    base_table = tables[1.0]
    rate = values(config, "pension")["discount_rate"]
    base = present_value(base_table, rate)
    print(f"\nInflation rule: until retirement "
          f"{obligation_after(tables, values(config, 'stresses')['inflation_up'], 1.0, rate) / base - 1:+.2%}, "
          f"brief's original rule on all payments {1.01 ** 5 - 1:+.2%}")
    years_left = base_table.groupby("employee_id")["years_from_now"].min()
    print(f"Employees retiring within 5 years: {(years_left < 5).sum()} of {len(years_left)}")
