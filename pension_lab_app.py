"""Pension Funding Stress Lab. All calculations live in src/pension_lab; this only displays them."""

import copy
import json

import pandas as pd
import streamlit as st

from src.pension import DATA_DIR, VALUATION_DATE, explain_employee
from src.pension_lab.attention import GROUPS
from src.pension_lab.audit import run_record
from src.pension_lab.charts import (
    funding_range_chart,
    overlapping_stresses,
    payments_chart,
    stress_chart,
)
from src.pension_lab.config import assumptions_table, load_config, values
from src.pension_lab.market_data import SOURCE_CREDIT, load_calibration, load_sources
from src.pension_lab.pipeline import run_pipeline
from src.pension_lab.portfolios import build_portfolios, rate_sensitivity_covered
from src.pension_lab.report import build_report, eur, main_finding, percent, points

SECTIONS = {
    "data-checks": "Data checks",
    "future-payments": "Future payments",
    "portfolio-comparison": "Portfolio comparison",
    "stress-tests": "Stress tests",
    "simulations": "Simulations",
    "what-needs-attention": "What needs attention",
    "assumptions-and-limits": "Assumptions and limits",
    "audit-trail": "Audit trail",
}
ANCHORS = {label: anchor for anchor, label in SECTIONS.items()}
FINAL_QUESTION = ("Would this view be useful to the investors you work with, or do they look at "
                  "this differently?")
DASHBOARD_NOTE = "I've also built an advisor dashboard that turns client data into daily priorities."
DASHBOARD_URL = "https://eufintech.streamlit.app"
TRACE_DEFAULT_EMPLOYEE = "E017"


def settings_with(config, funding, horizon, paths, contribution, cost):
    """Return a copy of the assumptions with the reader's settings applied."""
    changed = copy.deepcopy(config)
    changed["plan"]["starting_funding_level"]["value"] = funding
    changed["plan"]["yearly_contribution_eur"]["value"] = contribution
    changed["simulation"]["horizon_years"]["value"] = horizon
    changed["simulation"]["paths"]["value"] = paths
    changed["portfolios"]["yearly_cost"]["value"] = cost
    return changed


@st.cache_resource(max_entries=4, show_spinner="Running the model...")
def cached_results(funding, horizon, paths, contribution, cost):
    """Run the whole pipeline once for each set of inputs."""
    config = settings_with(load_config(), funding, horizon, paths, contribution, cost)
    raw = pd.read_csv(DATA_DIR / "employees_raw.csv")
    return config, run_pipeline(raw, config)


def settings_panel(defaults):
    """Show the inputs and return the chosen values."""
    plan = values(defaults, "plan")
    simulation = values(defaults, "simulation")
    with st.sidebar:
        st.subheader("Settings")
        st.caption("All values are illustrative. Changing one reruns the model. If the stability "
                   "check fails, results moved by more than the limit between two runs, so treat "
                   "them with more care or use more paths.")
        funding = st.number_input("Starting funding level", 0.5, 1.5,
                                  float(plan["starting_funding_level"]), 0.05, format="%.2f")
        horizon = st.number_input("Simulation years", 1, 30, simulation["horizon_years"], 1)
        paths = st.slider("Simulated paths", simulation["min_paths"], simulation["max_paths"],
                          simulation["paths"], 1000)
        contribution = st.number_input("Yearly contribution (EUR)", 0, 1_000_000,
                                       int(plan["yearly_contribution_eur"]), 10_000)
        cost = st.number_input("Yearly portfolio cost", 0.0, 0.02,
                               float(values(defaults, "portfolios")["yearly_cost"]), 0.001,
                               format="%.3f")
    return funding, horizon, paths, contribution, cost


def table(rows, **options):
    """Show a table without its index."""
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", **options)


def section_header(label):
    """Show a section title that the pipeline map can link to."""
    st.header(label, anchor=ANCHORS[label])


def section_link(label):
    """Return a link to a section of this page."""
    return f"[{label}](#{ANCHORS[label]})"


def result_first(results, config):
    """Show the question, main finding, chart and trust line."""
    st.title("Pension Funding Stress Lab")
    st.markdown("**Given messy employee pension data, what must be paid in the future, and how "
                "well could different sample investment mixes keep up if the world changes?**")

    failed = [check["name"] for check in results["checks"] if check["status"] == "failed"]
    if failed:
        st.error(f"Failed model checks: {', '.join(failed)}. Review these before using any result.")

    st.info(main_finding(results, config))
    floor = values(config, "attention")["funding_floor"]
    st.plotly_chart(funding_range_chart(results["bands"], floor), config={"displayModeBar": False})
    st.caption(f"Funding level means assets divided by the pension obligation. The line is the "
               f"middle simulated outcome (P50), the shaded area runs from P10 to P90, and the "
               f"dashed line marks {floor:.0%}.")
    item_count = len(results["attention"]["items"])
    st.markdown(f"Illustrative assumptions and synthetic employees: this shows a method, not a "
                f"forecast. {item_count} items are listed under "
                f"{section_link('What needs attention')}.")


def trust_box(results):
    """Show what the reader can and cannot rely on."""
    passed = sum(check["status"] == "passed" for check in results["checks"])
    sources = load_sources()
    lines = [f"{record['description']} (ECB, downloaded {record['download_date']}, data to "
             f"{record['last_date']})" for record in sources.values()]
    with st.container(border=True):
        st.subheader("How far to trust this")
        st.markdown(
            f"- **Employees:** synthetic. No real person or company is shown.\n"
            f"- **Portfolios:** illustrative risk exposures, not products.\n"
            f"- **Checks:** {passed} of {len(results['checks'])} model checks passed on this run.\n"
            f"- **Main limits:** simplified mortality, one flat discount rate, current employees "
            f"only, and no reaction by the plan after a fall in funding.\n"
            f"- **Status:** a student project. Not an official valuation and not investment advice."
        )
        with st.expander("Public data used"):
            st.markdown("\n".join(f"- {line}" for line in lines))
            st.caption(f"{SOURCE_CREDIT} The equity index belongs to its provider, so it is not "
                       "published here. Only statistics calculated from it are shown.")


def pipeline_map():
    """Show the steps from data to report, each linking to its section."""
    st.subheader("Pipeline")
    st.markdown(" → ".join(section_link(label) for label in SECTIONS.values()))


def data_checks(results):
    """Show raw rows, valid rows and the query list."""
    section_header("Data checks")
    st.markdown("Data is checked first, because every later number depends on it. A record with "
                "a question stays out of the valuation until someone answers it, instead of "
                "being guessed.")
    first, second, third = st.columns(3)
    first.metric("Raw rows", len(results["raw"]))
    second.metric("Valid employees", len(results["valid"]))
    third.metric("Client queries", len(results["queries"]))
    table(results["queries"])
    st.download_button("Download the query list", results["queries"].to_csv(index=False),
                       "client_queries.csv", "text/csv")
    with st.expander("What was changed automatically"):
        st.markdown("Only exact duplicates and text-formatted salaries are fixed without asking.")
        table(results["log"].astype(str))


def employee_trace(results, config):
    """Show one employee's calculation step by step."""
    st.subheader("One employee, step by step")
    employees = results["valid"]["employee_id"].tolist()
    start = employees.index(TRACE_DEFAULT_EMPLOYEE) if TRACE_DEFAULT_EMPLOYEE in employees else 0
    chosen = st.selectbox("Employee", employees, index=start)
    row = results["valid"].set_index("employee_id", drop=False).loc[chosen]
    steps = explain_employee(row, VALUATION_DATE, values(config, "pension"))
    table([{"Step": label, "Value": value} for label, value in steps])
    with st.expander("Formulas"):
        st.markdown(
            "- Yearly salary = monthly salary x 14\n"
            "- Earned pension = 1% x service so far x projected final salary\n"
            "- Expected payment = earned pension x probability still alive\n"
            "- DBO = today's value of all expected payments"
        )


def future_payments(results, config):
    """Show the payment chart, DBO, duration and one traceable employee."""
    section_header("Future payments")
    check = results["reconciliation"]
    first, second = st.columns(2)
    first.metric("Pension obligation (DBO)", eur(check["total_dbo"]))
    second.metric("Duration", f"{results['obligation_duration']:.1f} years")
    st.caption("The DBO is today's value of the pensions earned so far. Duration shows how much "
               "it reacts to interest rates: about this many percent for each 1 point.")
    st.plotly_chart(payments_chart(results["yearly"]), config={"displayModeBar": False})
    st.markdown(f"Adding up the present value of all {len(results['payments']):,} expected "
                f"payments gives {eur(check['total_present_value'])}, which matches the DBO.")
    with st.expander("Yearly payments table"):
        table(results["yearly"].round(2))
        st.download_button("Download yearly payments", results["yearly"].to_csv(index=False),
                           "yearly_payments.csv", "text/csv")
    employee_trace(results, config)


def portfolio_comparison(results, config):
    """Show the three illustrative mixes and their assumptions."""
    section_header("Portfolio comparison")
    st.markdown("Three illustrative mixes describe different risk exposures. They are not "
                "products, and none is presented as preferable.")
    obligation_duration = results["obligation_duration"]
    rows = []
    for portfolio in build_portfolios(config, obligation_duration):
        rows.append({
            "Portfolio": portfolio["label"],
            "Government": f"{portfolio['government']:.0%}",
            "Corporate": f"{portfolio['corporate']:.0%}",
            "Equity": f"{portfolio['equity']:.0%}",
            "Bond duration": f"{portfolio['bond_duration']:.1f}",
            "Rate sensitivity covered":
                f"{rate_sensitivity_covered(portfolio, obligation_duration):.0%}",
        })
    table(rows)
    st.caption("Portfolio B's duration is reset each year to match the obligation's expected "
               "duration. It is longer than most ordinary bond mixes.")
    with st.expander("Formulas"):
        st.markdown(
            "- Bond return = starting yield - duration x yield change + 0.5 x convexity x yield "
            "change squared\n"
            "- Corporate bonds use the same formula on the rate plus the spread\n"
            "- Portfolios return to their target mix every year, and a yearly cost is deducted"
        )


def stress_tests(results, config):
    """Show the six stresses as a chart and a table."""
    section_header("Stress tests")
    st.markdown("These are hypothetical shocks applied at once. They are not historical replicas "
                "or forecasts.")
    floor = values(config, "attention")["funding_floor"]
    st.plotly_chart(stress_chart(results["stress"], floor), config={"displayModeBar": False})
    overlaps = overlapping_stresses(results["stress"])
    overlap_note = ""
    if overlaps:
        overlap_note = (f" All three portfolios land on the same point for: {', '.join(overlaps)}. "
                        "These stresses change the obligation and leave asset prices alone.")
    st.caption(f"The dashed line marks {floor:.0%} funded.{overlap_note}")
    rows = [{"Stress": row["stress"], "Portfolio": row["portfolio"],
             "Funding": percent(row["funding_level"]), "Main driver": row["main_driver"],
             "Gap": eur(row["gap"]), "Assets": eur(row["assets"]),
             "Obligation": eur(row["obligation"])}
            for _, row in results["stress"].iterrows()]
    table(rows, height="content")


def simulations(results, config):
    """Show the risk measures, drivers and the effect of one setting."""
    section_header("Simulations")
    settings = values(config, "simulation")
    economy = values(config, "economy")
    floor = values(config, "attention")["funding_floor"]
    st.markdown(f"{settings['paths']:,} simulated paths over {settings['horizon_years']} years, "
                f"with rates, inflation, spreads, equities and longevity linked. The range over "
                f"time is the chart at the top of this page.")
    rows = [{"Portfolio": row["portfolio"], "P10": percent(row["p10"]), "P50": percent(row["p50"]),
             "P90": percent(row["p90"]),
             f"Below {floor:.0%}, end": percent(row["chance_below_floor_at_horizon"]),
             f"Below {floor:.0%}, any year end": percent(row["chance_below_floor_any_time"]),
             "1-year funding-at-risk": points(row["funding_at_risk_1y"]),
             "Lowest outcomes, average": percent(row["average_of_lowest_outcomes"])}
            for _, row in results["summary"].iterrows()]
    table(rows)
    st.caption(f"P10 means 10% of paths ended lower. Funding-at-risk is the fall in funding over "
               f"the first year that only {settings['tail_share']:.0%} of paths exceeded.")

    st.subheader("Risk drivers")
    wide = results["drivers"].pivot(index="risk", columns="portfolio", values="outcome_range")
    driver_rows = [{"Risk": risk, **{name: f"{wide.loc[risk, name] * 100:.1f} pts"
                                     for name in wide.columns}} for risk in wide.index]
    table(driver_rows)
    st.caption("Width between P10 and P90 at the end, with each risk on its own. The individual "
               "effects do not simply add up when risks are linked.")

    st.subheader("How much one setting matters")
    comparison = [
        {"Portfolio": main.portfolio,
         f"Inflation volatility {economy['inflation_volatility']:.1%}":
             percent(main.chance_below_floor_any_time),
         f"Inflation volatility {economy['inflation_volatility_comparison']:.1%}":
             percent(other.chance_below_floor_any_time)}
        for main, other in zip(results["summary"].itertuples(),
                               results["inflation_comparison"].itertuples())
    ]
    table(comparison)
    st.caption(f"Chance of falling below {floor:.0%} at any year end under each setting.")


def attention_item(item):
    """Show one attention item with all of its parts."""
    with st.container(border=True):
        st.markdown(f"**{item['issue']}**")
        st.markdown(f"{item['why_it_matters']}\n\n"
                    f"**Supporting value:** {item['supporting_value']}\n\n"
                    f"**Next step:** {item['next_step']}\n\n"
                    f"**From:** {section_link(item['source_section'])}")


def what_needs_attention(results):
    """Show the list in collapsible groups, with thresholds and anything not run."""
    section_header("What needs attention")
    attention = results["attention"]
    thresholds = attention["thresholds"]
    st.markdown(f"Illustrative checks: a funding floor of {thresholds['funding_floor']:.0%}, and a "
                f"{thresholds['chance_below_floor_threshold']:.0%} chance of falling below it at "
                "any year end. Groups follow a fixed order, with no urgency score.")
    if attention["message"]:
        st.markdown(attention["message"])
    for group in GROUPS:
        items = [item for item in attention["items"] if item["group"] == group]
        if items:
            with st.expander(f"{group} ({len(items)})", expanded=group == "Model checks"):
                for item in items:
                    attention_item(item)
    if attention["not_run"]:
        st.markdown(f"Not run: {', '.join(attention['not_run'])}.")


def assumptions_and_limits(results, config):
    """Show every assumption, what history says, and how much the main ones matter."""
    section_header("Assumptions and limits")
    rows = assumptions_table(config)
    estimated = sum(row["status"] == "estimated" for row in rows)
    st.markdown(f"{len(rows)} assumptions: {estimated} estimated from public data, the rest "
                "illustrative. A value is only called estimated if both halves of the history "
                "broadly agree.")
    sensitivity = results["sensitivity"]
    changes = sensitivity[sensitivity["scenario"] != "base"]
    table([{"Change": row["scenario"], "Obligation": eur(row["total_dbo"]),
            "Effect": f"{row['pct_change']:+.1f}%"} for _, row in changes.iterrows()])
    with st.expander("Every assumption"):
        table([{"Section": row["section"], "Name": row["name"], "Value": str(row["value"]),
                "Status": row["status"], "Source": row["source"], "Reason": row["reason"]}
               for row in rows])
    with st.expander("What history says"):
        table(load_calibration().round(4))
    st.markdown(
        "**Main limits**\n"
        "- Illustrative mortality rather than official tables, and no employee turnover in the "
        "base case.\n"
        "- Closed plan, current employees only: no pensioners and no new service.\n"
        "- One flat discount rate rather than a full yield curve.\n"
        "- Salary is not simulated separately; inflation stands in for it.\n"
        "- No default losses on corporate bonds, and no currency risk.\n"
        "- Credit spread settings are illustrative, as no suitable free series was found.\n"
        "- Nobody reacts in the model: no extra contributions or changes of investments."
    )


def audit_trail(results, config):
    """Show the checks, and offer the run record and report as downloads."""
    section_header("Audit trail")
    table([{"Check": check["name"], "Status": check["status"], "Detail": check["detail"]}
           for check in results["checks"]])
    record = run_record(results, config)
    first, second = st.columns(2)
    first.download_button("Download the report", build_report(results, config, record),
                          "pension_lab_report.html", "text/html")
    second.download_button("Download the run record", json.dumps(record, indent=2),
                           "run_record.json", "application/json")
    with st.expander("Run record summary"):
        st.markdown(f"- Commit: `{record['git']['commit']}`\n"
                    f"- Code fingerprint: `{record['code_fingerprint'][:16]}`\n"
                    f"- Input data fingerprint: `{record['input']['data_sha256'][:16]}`\n"
                    f"- Random seed: {record['random_seed']}")


def what_this_could_become():
    """Show the closing text, the optional link and the final question."""
    st.header("What this could become")
    st.markdown("With real data and reviewed assumptions, the same path from employee records to "
                "investment risk could support a small, repeatable check. Whether that would help "
                "anyone is still an open question.")
    st.caption(f"{DASHBOARD_NOTE} Optional, separate from this page: [view it]({DASHBOARD_URL}).")
    st.markdown(f"**{FINAL_QUESTION}**")


def main():
    """Build the page in the agreed order."""
    st.set_page_config(page_title="Pension Funding Stress Lab", layout="centered")
    defaults = load_config()
    config, results = cached_results(*settings_panel(defaults))

    result_first(results, config)
    trust_box(results)
    pipeline_map()
    data_checks(results)
    future_payments(results, config)
    portfolio_comparison(results, config)
    stress_tests(results, config)
    simulations(results, config)
    what_needs_attention(results)
    assumptions_and_limits(results, config)
    audit_trail(results, config)
    what_this_could_become()


main()
