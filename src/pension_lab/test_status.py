"""Record that the tests passed on exactly this code and data, and check that later."""

import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
STATUS_FILE = PROJECT_ROOT / "config" / "test_status.json"
FINGERPRINT_PATTERNS = ["src/**/*.py", "tests/**/*.py", "config/assumptions.yaml", "pytest.ini",
                        "data/employees_raw.csv", "data/market/*"]


def code_fingerprint(root=PROJECT_ROOT):
    """Return one hash covering the code, tests, assumptions and data."""
    paths = set()
    for pattern in FINGERPRINT_PATTERNS:
        paths.update(path for path in Path(root).glob(pattern) if path.is_file())

    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def recorded_test_status(status_file=STATUS_FILE, root=PROJECT_ROOT):
    """Return passed, failed or incomplete for the code as it is now."""
    if not Path(status_file).exists():
        return {"status": "incomplete", "detail": "No recorded test run.", "recorded": None}

    recorded = json.loads(Path(status_file).read_text(encoding="utf-8"))
    if recorded["fingerprint"] != code_fingerprint(root):
        return {"status": "incomplete", "recorded": recorded,
                "detail": "Code or data changed since the tests were last recorded."}
    if recorded["failed"] > 0 or recorded["passed"] == 0:
        return {"status": "failed", "recorded": recorded,
                "detail": f"{recorded['failed']} failed, {recorded['passed']} passed."}
    return {"status": "passed", "recorded": recorded,
            "detail": f"{recorded['passed']} passed on this exact code and data."}


def count_in_summary(summary, word):
    """Return a count such as the 12 in '12 passed'."""
    match = re.search(rf"(\d+) {word}", summary)
    return int(match.group(1)) if match else 0


def record_test_run():
    """Run the tests and save the outcome with the fingerprint of what was tested."""
    result = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=PROJECT_ROOT,
                            capture_output=True, text=True)
    summary = result.stdout.strip().splitlines()[-1]
    record = {
        "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fingerprint": code_fingerprint(),
        "passed": count_in_summary(summary, "passed"),
        "failed": count_in_summary(summary, "failed") + count_in_summary(summary, "error"),
        "summary": summary,
    }
    STATUS_FILE.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


if __name__ == "__main__":
    print(record_test_run()["summary"])
