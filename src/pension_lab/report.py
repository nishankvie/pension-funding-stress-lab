"""Build the downloadable HTML report. Every number comes from the results of the run."""

from html import escape

from src.pension_lab.config import assumptions_table, values

STYLE = """
body { font-family: Georgia, serif; color: #1f2933; background: #ffffff; margin: 0 auto;
       max-width: 860px; padding: 24px 16px; line-height: 1.5; }
h1, h2 { font-family: Helvetica, Arial, sans-serif; line-height: 1.2; }
h2 { margin-top: 36px; border-bottom: 1px solid #cbd2d9; padding-bottom: 4px; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; font-family: Helvetica, Arial, sans-serif; font-size: 14px;
        width: 100%; margin: 12px 0; }
th, td { border-bottom: 1px solid #e4e7eb; padding: 6px 8px; text-align: left; }
th { background: #f5f7fa; }
.note { background: #f5f7fa; border-left: 3px solid #52606d; padding: 8px 12px; }
"""


def eur(value):
    """Return a euro amount, e.g. EUR 2,821,996."""
    return f"EUR {value:,.0f}"


def percent(value):
    """Return a share as a percentage with one decimal."""
    return f"{value:.1%}"


def points(value):
    """Return a share as percentage points with a sign."""
    return f"{value * 100:+.1f} pts"


def html_table(headers, rows):
    """Return an HTML table, escaping everything so data can't change the page."""
    head = "".join(f"<th>{escape(str(header))}</th>" for header in headers)
    body = ""
    for row in rows:
        cells = "".join(f"<td>{escape(str(cell))}</td>" for cell in row)
        body += f"<tr>{cells}</tr>"
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def main_finding(results, config):
    """Return the headline sentences, built only from this run's results."""
    floor = values(config, "attention")["funding_floor"]
    sentences = []
    for _, row in results["summary"].iterrows():
        driver = results["largest_drivers"][row["portfolio_id"]].lower()
        sentences.append(
            f"{row['portfolio']} fell below {floor:.0%} funded at some year end in "
            f"{percent(row['chance_below_floor_any_time'])} of simulations; its largest single "
            f"risk was {driver}."
        )
    return "Under these illustrative assumptions: " + " ".join(sentences)


def data_section(results):
    """Return the data checks section."""
    queries = [(row["employee_id"], row["issue"], row["question"])
               for _, row in results["queries"].iterrows()]
    return (
        "<h2>Data checks</h2>"
        f"<p>Raw rows: {len(results['raw'])}. Valid employees: {len(results['valid'])}. "
        f"Client queries: {len(results['queries'])}. Changes logged: {len(results['log'])}. "
        "Records with a query stay out of the valuation until answered. All employees are "
        "synthetic.</p>"
        + html_table(["Employee", "Issue", "Question"], queries)
    )


def payments_section(results):
    """Return the future payments section."""
    yearly = results["yearly"]
    check = results["reconciliation"]
    peak = yearly.loc[yearly["expected_payment"].idxmax()]
    return (
        "<h2>Future payments</h2>"
        f"<p>Pension obligation (DBO, today's value of pensions earned so far): "
        f"{eur(check['total_dbo'])}. Present value of the payment table: "
        f"{eur(check['total_present_value'])}. Duration: {results['obligation_duration']:.1f} "
        f"years. Payments run from {yearly['calendar_year'].min()} to "
        f"{yearly['calendar_year'].max()}, and are highest in {int(peak['calendar_year'])} at "
        f"{eur(peak['expected_payment'])}.</p>"
    )


def stress_section(results):
    """Return the stress test section."""
    rows = [(row["stress"], row["portfolio"], eur(row["assets"]), eur(row["obligation"]),
             eur(row["gap"]), percent(row["funding_level"]), row["main_driver"])
            for _, row in results["stress"].iterrows()]
    return (
        "<h2>Stress tests</h2><p>These are hypothetical shocks applied at once. They are not "
        "historical replicas or forecasts.</p>"
        + html_table(["Stress", "Portfolio", "Assets", "Obligation", "Gap", "Funding",
                      "Main driver"], rows)
    )


def simulation_section(results, config):
    """Return the simulation section, including the inflation comparison."""
    settings = values(config, "simulation")
    floor = values(config, "attention")["funding_floor"]
    economy = values(config, "economy")
    risk_rows = [(row["portfolio"], percent(row["p10"]), percent(row["p50"]), percent(row["p90"]),
                  percent(row["chance_below_floor_at_horizon"]),
                  percent(row["chance_below_floor_any_time"]),
                  points(row["funding_at_risk_1y"]), percent(row["average_of_lowest_outcomes"]))
                 for _, row in results["summary"].iterrows()]

    wide = results["drivers"].pivot(index="risk", columns="portfolio", values="outcome_range")
    driver_rows = [[risk] + [points(value).replace("+", "") for value in wide.loc[risk]]
                   for risk in wide.index]

    comparison_rows = []
    for current, changed in zip(results["summary"].itertuples(),
                                results["inflation_comparison"].itertuples()):
        comparison_rows.append((current.portfolio, percent(current.chance_below_floor_any_time),
                                percent(changed.chance_below_floor_any_time)))

    return (
        f"<h2>Simulations</h2><p>{settings['paths']:,} simulated paths over "
        f"{settings['horizon_years']} years, seed {settings['seed']}.</p>"
        + html_table(["Portfolio", f"P10 year {settings['horizon_years']}", "P50", "P90",
                      f"Chance below {floor:.0%} at the end",
                      f"Chance below {floor:.0%} at any year end", "One-year funding-at-risk",
                      "Average of lowest outcomes"], risk_rows)
        + "<h3>Risk drivers</h3><p>Width between P10 and P90 at the end of the period, with each "
          "risk on its own. The individual effects do not simply add up when risks are linked.</p>"
        + html_table(["Risk"] + list(wide.columns), driver_rows)
        + "<h3>How much one setting matters</h3>"
        + html_table(
            ["Portfolio",
             f"Chance at any year end, inflation volatility {economy['inflation_volatility']:.1%}",
             f"Same, inflation volatility {economy['inflation_volatility_comparison']:.1%}"],
            comparison_rows)
    )


def assumptions_section(config):
    """Return every assumption with its label, source and reason."""
    rows = [(row["section"], row["name"], row["value"], row["status"], row["source"], row["reason"])
            for row in assumptions_table(config)]
    return (
        "<h2>Assumptions and limits</h2>"
        "<p>Main limits: synthetic employees, illustrative mortality rather than official tables, "
        "a closed plan with current employees only, one flat discount rate, no default losses on "
        "corporate bonds, and no reaction by the plan after a fall in funding.</p>"
        + html_table(["Section", "Name", "Value", "Status", "Source", "Reason"], rows)
    )


def audit_section(results, record):
    """Return the checks and run details."""
    rows = [(check["name"], check["status"], check["detail"]) for check in results["checks"]]
    return (
        "<h2>Audit trail</h2>"
        f"<p>Run time (UTC): {escape(record['run_time_utc'])}. Commit: "
        f"{escape(str(record['git']['commit']))}. Uncommitted changes: "
        f"{escape(str(record['git']['uncommitted_changes']))}. Input data fingerprint: "
        f"{escape(record['input']['data_sha256'])}.</p>"
        + html_table(["Check", "Status", "Detail"], rows)
    )


def build_report(results, config, record):
    """Return the complete report as one self-contained HTML page."""
    sections = [
        "<h1>Pension Funding Stress Lab</h1>",
        f'<p class="note">{escape(main_finding(results, config))}</p>',
        '<p class="note">Synthetic employees and illustrative portfolios. This is a student '
        "project, not an official valuation and not investment advice.</p>",
        data_section(results),
        payments_section(results),
        stress_section(results),
        simulation_section(results, config),
        assumptions_section(config),
        audit_section(results, record),
    ]
    body = "\n".join(sections)
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>Pension Funding Stress Lab</title><style>{STYLE}</style></head>"
            f"<body>{body}</body></html>")
