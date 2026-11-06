"""Measure how much each risk spreads outcomes when it is the only one active."""

import numpy as np
import pandas as pd

from src.pension_lab.simulation import ALL_RISKS, simulate

RISK_LABELS = {
    "rate": "Interest rates",
    "inflation": "Inflation",
    "spread": "Credit spreads",
    "equity": "Equity prices",
    "longevity": "Longevity",
}


def outcome_range(funding):
    """Return the width between the 10th and 90th percentile of final funding level."""
    p10, p90 = np.percentile(funding[:, -1], [10, 90])
    return float(p90 - p10)


def risk_driver_table(book, config, paths=None):
    """Return the outcome range per portfolio for each risk alone, and for all risks together."""
    # Every run uses the same seed, so differences come from the risks and not from luck
    runs = {"All risks": simulate(book, config, paths=paths)}
    for risk in ALL_RISKS:
        runs[RISK_LABELS[risk]] = simulate(book, config, paths=paths, active_risks=(risk,))

    rows = []
    for label, run in runs.items():
        for portfolio_id, result in run["portfolios"].items():
            rows.append({
                "portfolio_id": portfolio_id,
                "portfolio": result["label"],
                "risk": label,
                "outcome_range": outcome_range(result["funding_level"]),
            })
    return pd.DataFrame(rows)


def largest_drivers(table):
    """Return the single risk with the widest outcome range for each portfolio."""
    singles = table[table["risk"] != "All risks"]
    positions = singles.groupby("portfolio_id")["outcome_range"].idxmax()
    return singles.loc[positions].set_index("portfolio_id")["risk"].to_dict()


if __name__ == "__main__":
    from src.pension import DATA_DIR, validate_employees
    from src.pension_lab.config import load_config, values
    from src.pension_lab.risk_metrics import funding_bands, risk_summary, stability_table
    from src.pension_lab.simulation import build_payment_book

    pd.set_option("display.width", 160)
    config = load_config()
    valid = validate_employees(pd.read_csv(DATA_DIR / "employees_raw.csv"))[0]
    book = build_payment_book(valid, config)
    run = simulate(book, config)

    print(risk_summary(run, config).drop(columns="portfolio_id").round(4).T.to_string())
    bands = funding_bands(run)
    print(bands[bands["year"].isin([1, 5, 10])].drop(columns="portfolio_id").round(3).to_string())

    table = risk_driver_table(book, config)
    print(table.pivot(index="risk", columns="portfolio", values="outcome_range").round(3).to_string())
    print(largest_drivers(table))

    stability = stability_table(book, config)
    print(stability.round(4).to_string())
    print("All within limit:", stability["within_limit"].all(),
          "largest difference", round(stability["difference"].max(), 4))
