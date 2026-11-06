"""Load config/assumptions.yaml and refuse it if any entry is incomplete."""

from pathlib import Path

import yaml

CONFIG_FILE = Path(__file__).parent.parent.parent / "config" / "assumptions.yaml"
REQUIRED_FIELDS = {"value", "status", "source", "date", "reason"}
ALLOWED_STATUSES = {"estimated", "illustrative"}


def check_config(config):
    """Raise an error naming the entry if anything lacks its label, source, date or reason."""
    for section, entries in config.items():
        for name, entry in entries.items():
            missing = REQUIRED_FIELDS - set(entry)
            if missing:
                raise ValueError(f"{section}.{name} is missing: {', '.join(sorted(missing))}")
            if entry["status"] not in ALLOWED_STATUSES:
                raise ValueError(f"{section}.{name} has status {entry['status']!r}")


def load_config(path=CONFIG_FILE):
    """Return the checked assumptions, with every entry's value, label, source, date and reason."""
    with open(path, encoding="utf-8") as file:
        config = yaml.safe_load(file)
    check_config(config)
    return config


def values(config, section):
    """Return just the values of one section, as a plain dictionary."""
    return {name: entry["value"] for name, entry in config[section].items()}


def assumptions_table(config):
    """Return every assumption as one flat list of rows, for the app, report and audit record."""
    rows = []
    for section, entries in config.items():
        for name, entry in entries.items():
            rows.append({"section": section, "name": name, **entry})
    return rows
