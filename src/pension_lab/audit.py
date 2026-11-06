"""Build the machine-readable run record and save everything a reviewer would need."""

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.pension_lab.config import assumptions_table, values
from src.pension_lab.market_data import load_sources
from src.pension_lab.report import build_report
from src.pension_lab.test_status import PROJECT_ROOT, code_fingerprint

OUTPUT_DIR = PROJECT_ROOT / "outputs"


def data_hash(table):
    """Return a fingerprint of the input table, so a changed input can be detected."""
    return hashlib.sha256(table.to_csv(index=False).encode("utf-8")).hexdigest()


def git_state(root=PROJECT_ROOT):
    """Return the current commit and whether uncommitted changes exist, if git is available."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                                text=True, check=True).stdout.strip()
        changes = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True,
                                 text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unavailable", "uncommitted_changes": None}
    return {"commit": commit, "uncommitted_changes": bool(changes)}


def plain(value):
    """Return a value that can be written as JSON."""
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def key_results(results):
    """Return the headline results, which must repeat exactly for the same inputs and seed."""
    return plain({
        "total_dbo": results["reconciliation"]["total_dbo"],
        "payment_present_value": results["reconciliation"]["total_present_value"],
        "obligation_duration": results["obligation_duration"],
        "stress_tests": results["stress"].to_dict("records"),
        "risk_summary": results["summary"].to_dict("records"),
        "inflation_comparison": results["inflation_comparison"].to_dict("records"),
        "largest_drivers": results["largest_drivers"],
        "largest_stability_difference": results["stability"]["difference"].max(),
    })


def run_record(results, config, run_time=None):
    """Return the full audit record for one run."""
    run_time = run_time or datetime.now(timezone.utc)
    return plain({
        "run_time_utc": run_time.isoformat(timespec="seconds"),
        "git": git_state(),
        "code_fingerprint": code_fingerprint(),
        "random_seed": values(config, "simulation")["seed"],
        "paths": values(config, "simulation")["paths"],
        "input": {
            "source": results["source_name"],
            "data_sha256": data_hash(results["raw"]),
            "raw_rows": len(results["raw"]),
            "valid_rows": len(results["valid"]),
            "client_queries": len(results["queries"]),
            "changes_logged": len(results["log"]),
            "synthetic": True,
        },
        "assumptions": assumptions_table(config),
        "public_data_sources": load_sources(),
        "key_results": key_results(results),
        "model_checks": results["checks"],
        "test_status": results["tests"],
    })


def save_run(results, config, output_dir=OUTPUT_DIR, run_time=None):
    """Save the run record, report and CSV files in a new folder, and return that folder."""
    run_time = run_time or datetime.now(timezone.utc)
    folder = Path(output_dir) / run_time.strftime("run_%Y%m%d_%H%M%S")
    folder.mkdir(parents=True, exist_ok=True)

    record = run_record(results, config, run_time)
    (folder / "run_record.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    (folder / "report.html").write_text(build_report(results, config, record), encoding="utf-8")
    results["queries"].to_csv(folder / "client_queries.csv", index=False)
    results["log"].to_csv(folder / "cleaning_log.csv", index=False)
    results["yearly"].to_csv(folder / "yearly_payments.csv", index=False)
    return folder


if __name__ == "__main__":
    from src.pension import DATA_DIR
    from src.pension_lab.config import load_config
    from src.pension_lab.pipeline import run_pipeline

    config = load_config()
    results = run_pipeline(pd.read_csv(DATA_DIR / "employees_raw.csv"), config)
    folder = save_run(results, config)
    print(f"Saved to {folder}")
    for check in results["checks"]:
        print(f"  {check['status']:<11}{check['name']}: {check['detail']}")
