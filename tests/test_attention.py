"""The attention list must reflect the run, follow fixed rules and suggest only review."""

import copy
import json
import re

import pandas as pd
import pytest

from src.pension_lab.attention import GROUPS, NOTHING_CROSSED, build_attention
from src.pension_lab.audit import run_record
from src.pension_lab.report import build_report

BANNED = r"\b(buy|buying|sell|selling|switch|switching|best|worst|recommend\w*|urgent|urgency score)\b"


@pytest.fixture(scope="module")
def config(pipeline_config):
    return pipeline_config


@pytest.fixture(scope="module")
def results(pipeline_results):
    return pipeline_results


def issues(attention):
    return [item["issue"] for item in attention["items"]]


def with_threshold(config, value, name="chance_below_floor_threshold"):
    changed = copy.deepcopy(config)
    changed["attention"][name]["value"] = value
    return changed


def test_every_item_has_all_five_parts(results):
    for item in results["attention"]["items"]:
        for field in ("issue", "why_it_matters", "supporting_value", "next_step",
                      "source_section"):
            assert item[field]


def test_failed_model_check_appears_first(results, config):
    changed = dict(results)
    changed["checks"] = [
        {"name": "Fine", "status": "passed", "detail": "x"},
        {"name": "Unfinished", "status": "incomplete", "detail": "y"},
        {"name": "Broken", "status": "failed", "detail": "z"},
    ]
    names = issues(build_attention(changed, config))
    assert names[0] == "Model check failed: Broken"
    assert names[1] == "Model check incomplete: Unfinished"
    assert not any("Fine" in name for name in names)


def test_groups_follow_the_fixed_order(results, config):
    changed = dict(results)
    changed["checks"] = [{"name": "Broken", "status": "failed", "detail": "z"}]
    groups = [item["group"] for item in build_attention(changed, config)["items"]]
    positions = [GROUPS.index(group) for group in groups]
    assert positions == sorted(positions)
    assert set(groups) == set(GROUPS)


def test_passing_checks_create_no_check_items(results):
    assert not any(item["group"] == "Model checks" for item in results["attention"]["items"])


def test_query_item_uses_the_real_count(results, config):
    first = results["attention"]["items"][0]
    assert first["issue"] == "8 employee records need a client query"
    assert first["source_section"] == "Data checks"

    changed = dict(results)
    changed["queries"] = results["queries"].head(3)
    item = build_attention(changed, config)["items"][0]
    assert item["issue"] == "3 employee records need a client query"


def test_no_queries_means_no_data_item(results, config):
    changed = dict(results)
    changed["queries"] = results["queries"].head(0)
    assert not any(item["group"] == "Data quality"
                   for item in build_attention(changed, config)["items"])


def test_portfolio_appears_only_when_its_threshold_is_crossed(results, config):
    summary = results["summary"].copy()
    summary["chance_below_floor_any_time"] = [0.10, 0.25, 0.2499]
    changed = dict(results, summary=summary)
    names = issues(build_attention(changed, config))
    funding = [name for name in names if "over the check" in name]
    assert len(funding) == 1
    assert funding[0].startswith("B: Long bonds")


def test_every_portfolio_over_the_threshold_is_listed_in_fixed_order(results):
    names = [name for name in issues(results["attention"]) if "over the check" in name]
    assert [name[0] for name in names] == ["A", "B", "C"]


def test_changing_the_threshold_in_the_config_changes_the_list(results, config):
    names = issues(build_attention(results, with_threshold(config, 0.65)))
    funding = [name for name in names if "over the check" in name]
    assert len(funding) == 1
    assert funding[0].startswith("C: Growth mix")


def test_changing_the_floor_in_the_config_changes_the_stress_items(results, config):
    lower = build_attention(results, with_threshold(config, 0.70, "funding_floor"))
    stress = [item for item in lower["items"] if item["source_section"] == "Stress tests"]
    assert len(stress) == 1
    assert "Combined bad case" in stress[0]["issue"]
    assert "63.2%" in stress[0]["supporting_value"]


def test_main_driver_follows_each_flagged_portfolio(results):
    names = issues(results["attention"])
    for portfolio, driver in [("A: Short bonds", "interest rates"), ("B: Long bonds", "inflation"),
                              ("C: Growth mix", "equity prices")]:
        funding = next(i for i, name in enumerate(names)
                       if name.startswith(portfolio) and "over the check" in name)
        assert names[funding + 1] == f"{portfolio}: largest single risk is {driver}"


def test_unflagged_portfolio_gets_no_driver_item(results, config):
    names = issues(build_attention(results, with_threshold(config, 0.65)))
    assert sum("largest single risk" in name for name in names) == 1


def test_stress_items_list_exactly_the_portfolios_below_the_floor(results):
    stress = results["stress"]
    below = stress[stress["funding_level"] < 0.90]
    items = [item for item in results["attention"]["items"]
             if item["source_section"] == "Stress tests"]
    assert len(items) == below["stress"].nunique()
    listed = sum(item["supporting_value"].count("%") for item in items)
    assert listed == len(below)


def test_largest_sensitivity_comes_from_the_table(results):
    table = results["sensitivity"]
    largest = table.loc[table["pct_change"].abs().idxmax()]
    assert f"Largest sensitivity result: {largest['scenario']}" in issues(results["attention"])


def test_inflation_setting_item_appears_only_if_it_changes_the_list(results, config):
    assert any("inflation volatility setting" in name for name in issues(results["attention"]))
    same = dict(results, inflation_comparison=results["summary"])
    assert not any("inflation volatility setting" in name
                   for name in issues(build_attention(same, config)))


@pytest.mark.parametrize("key, name, gone", [
    ("summary", "Simulation", "over the check"),
    ("largest_drivers", "Risk-driver analysis", "largest single risk"),
    ("stress", "Stress tests", "takes funding below"),
    ("sensitivity", "Sensitivity results", "Largest sensitivity"),
    ("queries", "Data checks", "client query"),
])
def test_missing_output_creates_no_item_and_is_named(results, config, key, name, gone):
    changed = dict(results)
    changed[key] = None
    attention = build_attention(changed, config)
    assert name in attention["not_run"]
    assert not any(gone in issue for issue in issues(attention))


def test_nothing_is_reported_as_not_run_on_a_full_run(results):
    assert results["attention"]["not_run"] == []


def test_empty_list_says_checks_not_crossed_not_no_risk(results, config):
    changed = {"checks": [], "queries": results["queries"].head(0), "valid": results["valid"],
               "raw": results["raw"]}
    attention = build_attention(changed, config)
    assert attention["items"] == []
    assert attention["message"] == NOTHING_CROSSED
    assert "does not mean there is no risk" in attention["message"]


def test_no_item_uses_action_or_ranking_words(results, config):
    # Force every kind of item, including failed and incomplete checks
    changed = dict(results)
    changed["checks"] = [{"name": "A", "status": "failed", "detail": "x"},
                         {"name": "B", "status": "incomplete", "detail": "y"}]
    attention = build_attention(changed, config)
    text = json.dumps(attention["items"]).lower()
    assert re.findall(BANNED, text) == []


def test_stress_wording_cannot_be_read_as_good_news(results):
    items = [item for item in results["attention"]["items"]
             if item["source_section"] == "Stress tests"]
    assert items
    for item in items:
        assert "pass" not in item["why_it_matters"].lower()


def test_no_hard_coded_example_results(results, config):
    summary = results["summary"].copy()
    summary["chance_below_floor_any_time"] = 0.7777
    attention = build_attention(dict(results, summary=summary), config)
    text = json.dumps(attention["items"])
    assert "77.8%" in text
    assert "62.2%" not in text


def test_thresholds_are_reported_and_labelled_illustrative(results):
    thresholds = results["attention"]["thresholds"]
    assert thresholds == {"funding_floor": 0.90, "chance_below_floor_threshold": 0.25,
                          "status": "illustrative"}


def test_attention_adds_no_calculation_outside_the_results():
    source = open("src/pension_lab/attention.py", encoding="utf-8").read()
    assert "simulate" not in source and "numpy" not in source


def test_list_matches_the_audit_record(results, config):
    record = run_record(results, config)
    assert record["attention"] == json.loads(json.dumps(results["attention"]))


def test_list_matches_the_report(results, config):
    from html import escape

    record = run_record(results, config)
    report = build_report(results, config, record)
    for item in results["attention"]["items"]:
        for field in ("issue", "why_it_matters", "supporting_value", "next_step"):
            assert escape(item[field]) in report


def test_report_places_attention_after_simulations_and_before_assumptions(results, config):
    report = build_report(results, config, run_record(results, config))
    assert (report.index("<h2>Simulations") < report.index("<h2>What needs attention")
            < report.index("<h2>Assumptions and limits"))


def test_report_shows_not_run_and_empty_message(results, config):
    changed = dict(results)
    changed["attention"] = {"items": [], "not_run": ["Simulation"], "message": NOTHING_CROSSED,
                            "thresholds": results["attention"]["thresholds"]}
    report = build_report(changed, config, run_record(changed, config))
    assert "Not run: Simulation." in report
    assert "does not mean there is no risk" in report


def test_attention_ignores_retail_dashboard_ideas():
    source = open("src/pension_lab/attention.py", encoding="utf-8").read().lower()
    for word in ("churn", "login", "crm", "etf"):
        assert word not in source
