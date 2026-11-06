"""The page must run, follow the agreed order, and show only what the code calculated."""

import re
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src.pension_lab import pipeline
from src.pension_lab.charts import (
    PORTFOLIO_COLORS,
    PORTFOLIO_MARKERS,
    funding_range_chart,
    overlapping_stresses,
    payments_chart,
    stress_chart,
)
from src.pension_lab.report import main_finding

APP_FILE = str(Path(__file__).parent.parent / "pension_lab_app.py")
EXPECTED_HEADERS = [
    "Data checks", "Future payments", "Portfolio comparison", "Stress tests", "Simulations",
    "What needs attention", "Assumptions and limits", "Audit trail", "What this could become",
]
BANNED = r"\b(buy|sell|switch|best|worst|recommend\w*)\b"


def start_app():
    st.cache_resource.clear()
    return AppTest.from_file(APP_FILE, default_timeout=180).run()


@pytest.fixture(scope="module")
def app():
    return start_app()


def page_text(app):
    """Return all visible text on the page, including table contents."""
    parts = [element.value for kind in ("title", "header", "subheader", "markdown", "caption",
                                        "info", "error") for element in getattr(app.main, kind)]
    parts += [str(metric.value) for metric in app.main.metric]
    parts += [frame.value.to_string() for frame in app.main.dataframe]
    return "\n".join(str(part) for part in parts)


def test_app_runs_without_errors(app):
    assert not app.exception


def test_sections_are_in_the_agreed_order(app):
    assert [header.value for header in app.main.header] == EXPECTED_HEADERS


def test_result_comes_first(app, pipeline_results, pipeline_config):
    assert app.main.title[0].value == "Pension Funding Stress Lab"
    assert app.main.info[0].value == main_finding(pipeline_results, pipeline_config)


def test_page_ends_with_the_question(app):
    assert app.main.markdown[-1].value == ("**Would this view be useful to the investors you work "
                                           "with, or do they look at this differently?**")


def test_dashboard_link_is_optional_and_comes_before_the_question(app):
    note = app.main.caption[-1].value
    assert "I've also built an advisor dashboard that turns client data into daily priorities." in note
    assert "https://eufintech.streamlit.app" in note
    assert "Optional" in note
    text = page_text(app)
    assert text.count("eufintech") == 1


def test_pipeline_map_links_to_every_section(app):
    map_text = next(item.value for item in app.main.markdown if " → " in item.value)
    for anchor in ("data-checks", "future-payments", "portfolio-comparison", "stress-tests",
                   "simulations", "what-needs-attention", "assumptions-and-limits", "audit-trail"):
        assert f"(#{anchor})" in map_text


def test_shown_numbers_come_from_the_run(app, pipeline_results):
    text = page_text(app)
    assert f"EUR {pipeline_results['reconciliation']['total_dbo']:,.0f}" in text
    for _, row in pipeline_results["summary"].iterrows():
        assert f"{row['chance_below_floor_any_time']:.1%}" in text
        assert f"{row['p50']:.1%}" in text
    for _, row in pipeline_results["stress"].iterrows():
        assert f"{row['funding_level']:.1%}" in text


def test_row_counts_are_shown(app):
    metrics = {metric.label: metric.value for metric in app.main.metric}
    assert metrics["Raw rows"] == "43"
    assert metrics["Valid employees"] == "33"
    assert metrics["Client queries"] == "8"


def test_every_attention_item_is_on_the_page(app, pipeline_results):
    text = page_text(app)
    for item in pipeline_results["attention"]["items"]:
        assert item["issue"] in text
        assert item["next_step"] in text
    assert "Illustrative checks" in text


def test_attention_groups_are_collapsible_with_counts(app, pipeline_results):
    labels = [expander.label for expander in app.main.expander]
    items = pipeline_results["attention"]["items"]
    funding = sum(item["group"] == "Funding and stress" for item in items)
    assert f"Funding and stress ({funding})" in labels


def test_required_labels_are_visible(app):
    text = page_text(app)
    for phrase in ("synthetic", "illustrative", "not investment advice", "hypothetical",
                   "do not simply add up", "Not an official valuation"):
        assert phrase.lower() in text.lower()


def test_no_action_wording_or_branding(app):
    assert re.findall(BANNED, page_text(app).lower()) == []


def test_public_data_shows_source_and_date(app):
    assert "downloaded 2026-11-06" in page_text(app)


def test_employee_trace_defaults_to_a_known_result_and_can_change(app):
    assert app.main.selectbox[0].value == "E017"
    text = page_text(app)
    assert "€63,742" in text
    changed = AppTest.from_file(APP_FILE, default_timeout=180).run()
    changed.main.selectbox[0].set_value("E001").run()
    assert not changed.exception
    assert "€63,742" not in page_text(changed)


def test_changing_a_setting_reruns_the_model():
    changed = AppTest.from_file(APP_FILE, default_timeout=180).run()
    before = changed.main.info[0].value
    changed.sidebar.number_input[0].set_value(0.95).run()
    assert not changed.exception
    assert changed.main.info[0].value != before


def test_failed_check_is_shown_at_the_top_and_first_in_the_list(monkeypatch):
    real = pipeline.run_pipeline

    def with_failure(raw, config, **options):
        results = real(raw, config, tests={"status": "passed", "detail": "x", "recorded": None})
        results = dict(results)
        results["checks"] = results["checks"] + [
            {"name": "Forced check", "status": "failed", "detail": "For this test"}]
        results["attention"] = pipeline.build_attention(results, config)
        return results

    monkeypatch.setattr(pipeline, "run_pipeline", with_failure)
    app = start_app()
    assert "Forced check" in app.main.error[0].value
    assert "Model checks (1)" in [expander.label for expander in app.main.expander]
    st.cache_resource.clear()


# --- Charts ----------------------------------------------------------------------------------

def test_payments_chart_uses_the_yearly_table(pipeline_results):
    figure = payments_chart(pipeline_results["yearly"])
    assert isinstance(figure, go.Figure)
    assert list(figure.data[0].y) == pipeline_results["yearly"]["expected_payment"].tolist()


def test_funding_chart_has_a_panel_per_portfolio_on_one_axis(pipeline_results):
    figure = funding_range_chart(pipeline_results["bands"], 0.90)
    assert len(figure.data) == 9
    assert figure.layout.yaxis2.matches == "y"


def test_overlapping_stresses_are_named_on_the_page(app, pipeline_results):
    overlaps = overlapping_stresses(pipeline_results["stress"])
    assert overlaps == ["Inflation up 1 point for 5 years", "People live longer"]
    assert "All three portfolios land on the same point" in page_text(app)


def test_funding_chart_panels_stay_short_enough_for_a_phone(pipeline_results):
    figure = funding_range_chart(pipeline_results["bands"], 0.90)
    titles = [note.text for note in figure.layout.annotations]
    assert titles == ["Portfolio A", "Portfolio B", "Portfolio C"]


def test_stress_chart_gives_each_portfolio_a_colour_and_a_shape(pipeline_results):
    figure = stress_chart(pipeline_results["stress"], 0.90)
    assert len(figure.data) == 3
    colours = {trace.marker.color for trace in figure.data}
    shapes = {trace.marker.symbol for trace in figure.data}
    assert colours == set(PORTFOLIO_COLORS.values())
    assert shapes == set(PORTFOLIO_MARKERS.values())
    stress = pipeline_results["stress"]
    first = stress[stress["portfolio_id"] == "a_short_bonds"]["funding_level"].tolist()
    assert list(figure.data[0].x) == pytest.approx(first)
