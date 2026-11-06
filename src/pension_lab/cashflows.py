"""Turn the pension promise into a table of expected future payments, one row per payment."""

import numpy as np
import pandas as pd

from src.pension import ASSUMPTIONS, VALUATION_DATE, dbo_portfolio, survival_prob

# Rounding noise on the total is about EUR 0.000000001, and the smallest single payment is worth
# about EUR 0.003 today. This sits between the two, so noise passes but a lost payment fails
RECONCILIATION_TOLERANCE_EUR = 0.000001

# Payment times are rounded to this many decimals before choosing a year, so that floating-point
# noise can't push a payment due exactly at the end of a year into the next one
TIME_DECIMALS = 9


def check_inputs(results, valuation_date, turnover_rate, leaver_benefit_share, indexation_rate):
    """Stop with a clear message if the inputs would give a meaningless payment table."""
    valuation = pd.Timestamp(valuation_date)
    if (valuation.month, valuation.day) != (12, 31):
        raise ValueError("Year labels assume a 31 December valuation date.")
    if (results["years_to_retirement"] < 0).any():
        raise ValueError("Employees past retirement age are not covered by this model.")
    if not 0 <= turnover_rate < 1:
        raise ValueError("turnover_rate must be at least 0 and below 1.")
    if not 0 <= leaver_benefit_share <= 1:
        raise ValueError("leaver_benefit_share must be between 0 and 1.")
    if indexation_rate <= -1:
        raise ValueError("indexation_rate must be above -100%.")


def benefit_kept_factor(years_to_retirement, turnover_rate, leaver_benefit_share):
    """Return the expected share of the pension kept after allowing for people leaving."""
    # Stayers keep all of it and leavers keep their share. Share 0 means leavers lose everything,
    # which is the largest possible effect of turnover rather than a realistic Austrian rule
    stay_probability = (1 - turnover_rate) ** years_to_retirement
    return stay_probability + (1 - stay_probability) * leaver_benefit_share


def employee_payment_table(valid_df, valuation_date=VALUATION_DATE, a=ASSUMPTIONS,
                           turnover_rate=0.0, leaver_benefit_share=0.0, indexation_rate=0.0):
    """Return one row per employee per expected payment, using exact payment times."""
    results, _ = dbo_portfolio(valid_df, valuation_date, a)
    check_inputs(results, valuation_date, turnover_rate, leaver_benefit_share, indexation_rate)

    retirement_age = a["retirement_age"]
    payment_numbers = np.arange(0, a["max_age"] - retirement_age + 1)
    # Survival is split at retirement, exactly as in the DBO. Measuring straight from a part-year
    # age would use different yearly steps and the payments would stop matching the DBO
    survival_after_retirement = np.array(
        [survival_prob(retirement_age, retirement_age + k, a) for k in payment_numbers]
    )
    indexation_factors = (1 + indexation_rate) ** payment_numbers
    v = 1 / (1 + a["discount_rate"])
    valuation_year = pd.Timestamp(valuation_date).year

    tables = []
    for _, person in results.iterrows():
        years_from_now = person["years_to_retirement"] + payment_numbers
        kept = benefit_kept_factor(person["years_to_retirement"], turnover_rate,
                                   leaver_benefit_share)
        survival = person["survival_to_retirement"] * survival_after_retirement
        expected_payment = person["earned_pension"] * indexation_factors * kept * survival
        simulation_year = np.ceil(np.round(years_from_now, TIME_DECIMALS)).astype(int)

        tables.append(pd.DataFrame({
            "employee_id": person["employee_id"],
            "payment_number": payment_numbers,
            "age_at_payment": retirement_age + payment_numbers,
            "years_from_now": years_from_now,
            "simulation_year": simulation_year,
            "calendar_year": valuation_year + simulation_year,
            "earned_pension": person["earned_pension"],
            "indexation_factor": indexation_factors,
            "benefit_kept_factor": kept,
            "survival_probability": survival,
            "expected_payment": expected_payment,
            "present_value": expected_payment * v ** years_from_now,
        }))

    return pd.concat(tables, ignore_index=True)


def yearly_payment_table(payments):
    """Return expected payments and their present value added up for each year."""
    columns = ["expected_payment", "present_value"]
    return payments.groupby(["simulation_year", "calendar_year"], as_index=False)[columns].sum()


def present_value(payments, discount_rate):
    """Return today's value of the payments at any flat rate, using exact payment times."""
    v = 1 / (1 + discount_rate)
    return (payments["expected_payment"] * v ** payments["years_from_now"]).sum()


def reconcile_to_dbo(payments, valid_df, valuation_date=VALUATION_DATE, a=ASSUMPTIONS):
    """Compare the payments' present value with the DBO, in total and for each employee."""
    results, total_dbo = dbo_portfolio(valid_df, valuation_date, a)
    total_present_value = present_value(payments, a["discount_rate"])

    by_employee = payments.groupby("employee_id")["present_value"].sum()
    dbo_by_employee = results.set_index("employee_id")["dbo"]
    largest_employee_difference = (by_employee - dbo_by_employee).abs().max()
    total_difference = total_present_value - total_dbo

    within_tolerance = (abs(total_difference) <= RECONCILIATION_TOLERANCE_EUR
                        and largest_employee_difference <= RECONCILIATION_TOLERANCE_EUR)
    return {
        "total_dbo": total_dbo,
        "total_present_value": total_present_value,
        "total_difference": total_difference,
        "largest_employee_difference": largest_employee_difference,
        "tolerance": RECONCILIATION_TOLERANCE_EUR,
        "passed": bool(within_tolerance),
    }


if __name__ == "__main__":
    from src.pension import DATA_DIR, validate_employees

    raw = pd.read_csv(DATA_DIR / "employees_raw.csv")
    valid, _, queries = validate_employees(raw)
    payments = employee_payment_table(valid)
    yearly = yearly_payment_table(payments)
    check = reconcile_to_dbo(payments, valid)

    print(f"Raw rows {len(raw)}, valid employees {len(valid)}, client queries {len(queries)}")
    print(f"Payment rows {len(payments):,}")
    print(f"Total DBO                 EUR {check['total_dbo']:,.2f}")
    print(f"Present value of payments EUR {check['total_present_value']:,.2f}")
    print(f"Difference                EUR {check['total_difference']:.12f}")
    print(f"Largest employee diff     EUR {check['largest_employee_difference']:.12f}")
    print(f"Tolerance                 EUR {check['tolerance']:.6f}")
    print(f"Reconciled: {check['passed']}")
    print()
    print(f"Undiscounted payments     EUR {yearly['expected_payment'].sum():,.0f}")
    first = yearly.iloc[0]
    peak = yearly.loc[yearly["expected_payment"].idxmax()]
    print(f"First payment year {first['calendar_year']:.0f}, EUR {first['expected_payment']:,.0f}")
    print(f"Peak payment year  {peak['calendar_year']:.0f}, EUR {peak['expected_payment']:,.0f}")
    print(f"Last payment year  {yearly['calendar_year'].max()}")
    first_ten = yearly[yearly["simulation_year"] <= 10]
    print(f"Paid in first 10 years    EUR {first_ten['expected_payment'].sum():,.0f}, "
          f"{first_ten['present_value'].sum() / check['total_dbo']:.1%} of the DBO")
    print()
    print("Illustrative settings, not part of the base case:")
    for label, options in [
        ("indexation 1%", {"indexation_rate": 0.01}),
        ("turnover 3%, leavers keep nothing", {"turnover_rate": 0.03}),
        ("turnover 3%, leavers keep half",
         {"turnover_rate": 0.03, "leaver_benefit_share": 0.5}),
    ]:
        changed = employee_payment_table(valid, **options)["present_value"].sum()
        print(f"  {label:<36} EUR {changed:>11,.0f}  {changed / check['total_dbo'] - 1:+.1%}")
