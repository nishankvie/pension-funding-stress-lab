"""Turn the results of a run into a short list of what to review. Adds no new calculation."""

from src.pension_lab.config import assumptions_table, values

# Fixed sort order. Within a group, items keep a fixed order (Portfolios A, B, C, then stresses
# in their listed order), so the list never reads as a ranking and needs no urgency score
GROUPS = ["Model checks", "Data quality", "Funding and stress", "Assumptions"]

NOTHING_CROSSED = ("None of the items crossed the configured checks. This does not mean there is "
                   "no risk.")

# Each output this list reads, and how to name it if it wasn't produced
OPTIONAL_OUTPUTS = {
    "checks": "Model checks",
    "queries": "Data checks",
    "summary": "Simulation",
    "largest_drivers": "Risk-driver analysis",
    "stress": "Stress tests",
    "sensitivity": "Sensitivity results",
    "inflation_comparison": "Inflation volatility comparison",
}


def make_item(group, issue, why_it_matters, supporting_value, next_step, source_section):
    """Return one attention item with every field the page must show."""
    return {
        "group": group,
        "issue": issue,
        "why_it_matters": why_it_matters,
        "supporting_value": supporting_value,
        "next_step": next_step,
        "source_section": source_section,
    }


def model_check_items(checks):
    """Return failed checks first, then incomplete ones."""
    items = []
    for status, why, next_step in [
        ("failed", "Results on this page may be wrong until this is explained.",
         "Review this check before relying on any result."),
        ("incomplete", "This part of the run could not be confirmed.",
         "Complete this check, or record why it does not apply."),
    ]:
        for check in checks:
            if check["status"] == status:
                items.append(make_item("Model checks", f"Model check {status}: {check['name']}",
                                       why, check["detail"], next_step, "Audit trail"))
    return items


def data_quality_items(results):
    """Return one item if any employee records need a client query."""
    query_count = len(results["queries"])
    if query_count == 0:
        return []
    return [make_item(
        "Data quality",
        f"{query_count} employee records need a client query",
        "These records stay out of the valuation, so the obligation may be incomplete.",
        f"{query_count} queries; {len(results['valid'])} valid employees from "
        f"{len(results['raw'])} raw rows",
        "Send the query list to the data owner, then rerun with the answers.",
        "Data checks",
    )]


def driver_item(results, row):
    """Return the main driver of one flagged portfolio, with its range from the driver analysis."""
    risk = results["largest_drivers"][row["portfolio_id"]]
    drivers = results["drivers"]
    chosen = drivers[(drivers["portfolio_id"] == row["portfolio_id"]) & (drivers["risk"] == risk)]
    width = chosen["outcome_range"].iloc[0]
    return make_item(
        "Funding and stress",
        f"{row['portfolio']}: largest single risk is {risk.lower()}",
        "This shows which assumption to review first. Single-risk effects do not add up when "
        "risks are linked.",
        f"P10 to P90 range of {width * 100:.1f} points with this risk alone",
        f"Review the assumptions for {risk.lower()} and their sources.",
        "Simulations",
    )


def funding_items(results, floor, threshold):
    """Return an item for each portfolio over the threshold, followed by its main driver."""
    items = []
    for _, row in results["summary"].iterrows():
        chance = row["chance_below_floor_any_time"]
        if chance < threshold:
            continue
        items.append(make_item(
            "Funding and stress",
            f"{row['portfolio']}: chance of falling below {floor:.0%} funded is over the check",
            "A fall below this level would show a funding gap worth reviewing, under these "
            "illustrative assumptions.",
            f"{chance:.1%} at any year end; check set at {threshold:.1%}",
            "Review the assumptions behind this result, and confirm which funding level matters "
            "for this plan.",
            "Simulations",
        ))
        if results.get("largest_drivers") is not None:
            items.append(driver_item(results, row))
    return items


def stress_items(results, floor):
    """Return one item per named stress that takes any portfolio below the floor."""
    stress = results["stress"]
    items = []
    for stress_label in stress["stress"].unique():
        below = stress[(stress["stress"] == stress_label) & (stress["funding_level"] < floor)]
        if below.empty:
            continue
        portfolios = ", ".join(below["portfolio"])
        funding = "; ".join(f"{row['portfolio']} {row['funding_level']:.1%}"
                            for _, row in below.iterrows())
        items.append(make_item(
            "Funding and stress",
            f"{stress_label} takes funding below {floor:.0%}: {portfolios}",
            "This hypothetical shock opens a gap large enough to pass the funding check.",
            funding,
            "Review whether the size of this hypothetical shock suits the plan.",
            "Stress tests",
        ))
    return items


def largest_assumption_item(sensitivity):
    """Return the assumption change with the largest effect on the obligation."""
    changes = sensitivity[sensitivity["scenario"] != "base"]
    largest = changes.loc[changes["pct_change"].abs().idxmax()]
    return make_item(
        "Assumptions",
        f"Largest sensitivity result: {largest['scenario']}",
        "The obligation depends most on this assumption among those tested.",
        f"{largest['pct_change']:+.1f}% change in the obligation",
        "Check this assumption against the plan's actual valuation basis.",
        "Assumptions and limits",
    )


def illustrative_count_item(config):
    """Return how many assumptions are illustrative, if any are."""
    rows = assumptions_table(config)
    illustrative = sum(row["status"] == "illustrative" for row in rows)
    if illustrative == 0:
        return []
    return [make_item(
        "Assumptions",
        f"{illustrative} of {len(rows)} assumptions are illustrative",
        "Results show the method, not real-world probabilities.",
        f"{illustrative} illustrative, {len(rows) - illustrative} estimated",
        "Replace the most important illustrative values with reviewed ones.",
        "Assumptions and limits",
    )]


def inflation_setting_item(results, config, threshold):
    """Return an item if the inflation setting changes which portfolios cross the check."""
    economy = values(config, "economy")
    measure = "chance_below_floor_any_time"
    changed = []
    for main, other in zip(results["summary"].to_dict("records"),
                           results["inflation_comparison"].to_dict("records")):
        if (main[measure] >= threshold) != (other[measure] >= threshold):
            changed.append(f"{main['portfolio']}: {main[measure]:.1%} at "
                           f"{economy['inflation_volatility']:.1%}, {other[measure]:.1%} at "
                           f"{economy['inflation_volatility_comparison']:.1%}")
    if not changed:
        return []
    return [make_item(
        "Assumptions",
        "The inflation volatility setting changes which portfolios cross the check",
        "This illustrative setting alone decides part of the list above.",
        "; ".join(changed) + f"; check set at {threshold:.1%}",
        "Review inflation volatility against the data and the plan's actual inflation link.",
        "Simulations",
    )]


def build_attention(results, config):
    """Return the attention list, the thresholds used, and anything that was not run."""
    settings = values(config, "attention")
    floor = settings["funding_floor"]
    threshold = settings["chance_below_floor_threshold"]

    not_run = [name for key, name in OPTIONAL_OUTPUTS.items() if results.get(key) is None]

    groups = {group: [] for group in GROUPS}
    if results.get("checks") is not None:
        groups["Model checks"] += model_check_items(results["checks"])
    if results.get("queries") is not None:
        groups["Data quality"] += data_quality_items(results)
    if results.get("summary") is not None:
        groups["Funding and stress"] += funding_items(results, floor, threshold)
    if results.get("stress") is not None:
        groups["Funding and stress"] += stress_items(results, floor)
    if results.get("sensitivity") is not None:
        groups["Assumptions"].append(largest_assumption_item(results["sensitivity"]))
    if results.get("summary") is not None and results.get("inflation_comparison") is not None:
        groups["Assumptions"] += inflation_setting_item(results, config, threshold)

    items = [item for group in GROUPS for item in groups[group]]
    # Counted only once real items exist, so it can't fill an otherwise empty list
    if items:
        items += illustrative_count_item(config)

    return {
        "items": items,
        "not_run": not_run,
        "thresholds": {
            "funding_floor": floor,
            "chance_below_floor_threshold": threshold,
            "status": "illustrative",
        },
        "message": "" if items else NOTHING_CROSSED,
    }
