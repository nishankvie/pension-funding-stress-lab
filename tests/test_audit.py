"""The run record must contain everything in Part 8 and be honest about what was verified."""

import json
import re
from datetime import datetime, timezone

import pandas as pd
import pytest

from src.pension_lab import audit
from src.pension_lab.audit import data_hash, git_state, key_results, plain, run_record, save_run
from src.pension_lab.pipeline import check_result, run_pipeline, automated_tests_check
from src.pension_lab.report import build_report, main_finding
from src.pension_lab.test_status import code_fingerprint, count_in_summary, recorded_test_status

FIXED_TIME = datetime(2026, 11, 6, 12, 0, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def record(pipeline_results, pipeline_config):
    return run_record(pipeline_results, pipeline_config, FIXED_TIME)


@pytest.fixture(scope="module")
def report(pipeline_results, pipeline_config, record):
    return build_report(pipeline_results, pipeline_config, record)


def test_record_contains_every_part_8_item(record):
    assert record["run_time_utc"] == "2026-11-06T12:00:00+00:00"
    assert record["git"]["commit"]
    assert record["random_seed"] == 42
    assert record["input"]["data_sha256"]
    assert record["input"]["raw_rows"] == 43
    assert record["input"]["valid_rows"] == 33
    assert record["input"]["client_queries"] == 8
    assert len(record["assumptions"]) > 50
    assert set(record["public_data_sources"]) == {"yield_20y", "inflation", "equity_index"}
    assert record["key_results"]["total_dbo"] == pytest.approx(2_821_995.55, abs=0.01)
    assert record["test_status"]["status"] == "passed"


def test_every_assumption_keeps_its_label(record):
    for row in record["assumptions"]:
        assert row["status"] in ("estimated", "illustrative")
        assert row["source"] and row["date"] and row["reason"]


def test_record_can_be_written_and_read_as_json(record):
    assert json.loads(json.dumps(record)) == record


def test_plain_converts_numpy_and_dates():
    import datetime as dt
    import numpy as np
    assert plain({"a": np.float64(1.5), "b": dt.date(2026, 1, 2), "c": (np.int64(3),)}) == {
        "a": 1.5, "b": "2026-01-02", "c": [3]}


def test_changed_data_gives_a_different_hash(raw_employees):
    changed = raw_employees.copy()
    changed.loc[0, "employee_id"] = "E999"
    assert data_hash(changed) != data_hash(raw_employees)


def test_same_inputs_repeat_key_results_exactly(raw_employees, pipeline_config, pipeline_results):
    tests = {"status": "passed", "detail": "Given.", "recorded": None}
    second = run_pipeline(raw_employees, pipeline_config, tests=tests)
    assert key_results(second) == key_results(pipeline_results)


def test_missing_git_is_reported_not_invented(tmp_path):
    assert git_state(tmp_path / "missing") == {"commit": "unavailable",
                                               "uncommitted_changes": None}


def test_uncommitted_changes_are_flagged(tmp_path):
    import subprocess
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    for setting in (["user.email", "t@example.com"], ["user.name", "T"]):
        subprocess.run(["git", "config", *setting], cwd=tmp_path, check=True)
    (tmp_path / "a.txt").write_text("1")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "x"], cwd=tmp_path, check=True)
    assert git_state(tmp_path)["uncommitted_changes"] is False
    (tmp_path / "a.txt").write_text("2")
    assert git_state(tmp_path)["uncommitted_changes"] is True


# --- Test status -----------------------------------------------------------------------------

def make_project(root):
    (root / "src").mkdir()
    (root / "src" / "a.py").write_text("x = 1\n")
    return root


def write_status(path, fingerprint, passed=10, failed=0):
    path.write_text(json.dumps({"fingerprint": fingerprint, "passed": passed, "failed": failed}))


def test_fingerprint_changes_when_code_changes(tmp_path):
    root = make_project(tmp_path)
    before = code_fingerprint(root)
    (root / "src" / "a.py").write_text("x = 2\n")
    assert code_fingerprint(root) != before


def test_recorded_pass_on_the_same_code_counts(tmp_path):
    root = make_project(tmp_path)
    status_file = root / "status.json"
    write_status(status_file, code_fingerprint(root))
    assert recorded_test_status(status_file, root)["status"] == "passed"


def test_changed_code_makes_the_recorded_pass_incomplete(tmp_path):
    root = make_project(tmp_path)
    status_file = root / "status.json"
    write_status(status_file, code_fingerprint(root))
    (root / "src" / "a.py").write_text("x = 2\n")
    assert recorded_test_status(status_file, root)["status"] == "incomplete"


def test_recorded_failure_is_reported_as_failed(tmp_path):
    root = make_project(tmp_path)
    status_file = root / "status.json"
    write_status(status_file, code_fingerprint(root), passed=9, failed=1)
    assert recorded_test_status(status_file, root)["status"] == "failed"


def test_no_recorded_run_is_incomplete(tmp_path):
    root = make_project(tmp_path)
    assert recorded_test_status(root / "none.json", root)["status"] == "incomplete"


def test_summary_counts_are_read():
    assert count_in_summary("3 failed, 150 passed in 11.2s", "passed") == 150
    assert count_in_summary("3 failed, 150 passed in 11.2s", "failed") == 3
    assert count_in_summary("150 passed in 11.2s", "failed") == 0


# --- Model checks ----------------------------------------------------------------------------

def test_all_run_time_checks_pass_on_the_sample(pipeline_results):
    checks = pipeline_results["checks"]
    assert len(checks) == 10
    assert [c["name"] for c in checks if c["status"] != "passed"] == []


def test_test_status_passes_through_to_the_checks():
    status = {"status": "incomplete", "detail": "Changed."}
    assert automated_tests_check(status)["status"] == "incomplete"


def test_turnover_makes_reconciliation_incomplete_not_failed(raw_employees, pipeline_config):
    import copy
    changed = copy.deepcopy(pipeline_config)
    changed["plan"]["turnover_rate"]["value"] = 0.03
    tests = {"status": "passed", "detail": "Given.", "recorded": None}
    result = run_pipeline(raw_employees, changed, tests=tests)
    assert result["checks"][0]["status"] == "incomplete"


# --- Report ----------------------------------------------------------------------------------

def test_report_shows_real_values(report, pipeline_results):
    assert "EUR 2,821,996" in report
    assert "Raw rows: 43. Valid employees: 33. Client queries: 8." in report
    for _, row in pipeline_results["stress"].iterrows():
        assert f"{row['funding_level']:.1%}" in report
    for _, row in pipeline_results["summary"].iterrows():
        assert f"{row['chance_below_floor_any_time']:.1%}" in report


def test_report_follows_the_results_not_fixed_text(pipeline_results, pipeline_config, record):
    changed = dict(pipeline_results)
    changed["summary"] = pipeline_results["summary"].copy()
    changed["summary"]["chance_below_floor_any_time"] = 0.1234
    assert "12.3%" in main_finding(changed, pipeline_config)
    assert "12.3%" in build_report(changed, pipeline_config, record)


def test_report_shows_both_inflation_settings(report, pipeline_results):
    assert "inflation volatility 1.5%" in report
    assert "inflation volatility 1.0%" in report
    for _, row in pipeline_results["inflation_comparison"].iterrows():
        assert f"{row['chance_below_floor_any_time']:.1%}" in report


def test_inflation_comparison_is_a_real_second_run(pipeline_results, pipeline_config):
    # Lower inflation volatility must narrow Portfolio B's shortfall chance, and it must match
    # a separate simulation with that setting
    import copy
    from src.pension_lab.risk_metrics import risk_summary
    from src.pension_lab.simulation import build_payment_book, simulate

    changed = copy.deepcopy(pipeline_config)
    changed["economy"]["inflation_volatility"]["value"] = 0.01
    book = build_payment_book(pipeline_results["valid"], changed)
    expected = risk_summary(simulate(book, changed), changed)
    measure = "chance_below_floor_any_time"
    comparison = pipeline_results["inflation_comparison"]
    assert comparison[measure].tolist() == expected[measure].tolist()
    assert comparison[measure].iloc[1] < pipeline_results["summary"][measure].iloc[1]


def test_report_states_its_limits(report):
    for phrase in ("Synthetic employees", "not investment advice", "not an official valuation",
                   "hypothetical", "do not simply add up"):
        assert phrase in report


def test_report_has_no_action_wording_or_branding(report):
    text = re.sub(r"<[^>]+>", " ", report).lower()
    banned = r"\b(buy|sell|switch|best|worst|recommend\w*)\b"
    assert re.findall(banned, text) == []


def test_report_escapes_data(pipeline_results, pipeline_config, record):
    changed = dict(pipeline_results)
    changed["queries"] = pd.DataFrame([{"employee_id": "<script>alert(1)</script>",
                                        "issue": "x", "question": "y"}])
    report = build_report(changed, pipeline_config, record)
    assert "<script>" not in report
    assert "&lt;script&gt;" in report


def test_report_shows_a_failed_check(pipeline_results, pipeline_config, record):
    changed = dict(pipeline_results)
    changed["checks"] = [check_result("Example check", "failed", "Forced for this test")]
    report = build_report(changed, pipeline_config, record)
    assert "Example check" in report and "failed" in report


def test_report_is_self_contained(report):
    assert "<script" not in report
    assert "http://" not in report


# --- Saved files -----------------------------------------------------------------------------

def test_saved_run_contains_every_file(pipeline_results, pipeline_config, tmp_path):
    folder = save_run(pipeline_results, pipeline_config, tmp_path, FIXED_TIME)
    names = {path.name for path in folder.iterdir()}
    assert names == {"run_record.json", "report.html", "client_queries.csv", "cleaning_log.csv",
                     "yearly_payments.csv"}


def test_saved_csvs_match_the_run(pipeline_results, pipeline_config, tmp_path):
    folder = save_run(pipeline_results, pipeline_config, tmp_path, FIXED_TIME)
    assert len(pd.read_csv(folder / "client_queries.csv")) == 8
    assert len(pd.read_csv(folder / "cleaning_log.csv")) == 6
    yearly = pd.read_csv(folder / "yearly_payments.csv")
    assert yearly["expected_payment"].sum() == pytest.approx(
        pipeline_results["yearly"]["expected_payment"].sum())


def test_outputs_are_kept_out_of_git():
    gitignore = (audit.PROJECT_ROOT / ".gitignore").read_text()
    assert "outputs/" in gitignore
