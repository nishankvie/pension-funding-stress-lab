"""Download public ECB series, record where and when each came from, and load them by month."""

import hashlib
import io
import json
import urllib.request
from datetime import date
from pathlib import Path

import pandas as pd

from src.pension_lab.economy import calibration_table

MARKET_DIR = Path(__file__).parent.parent.parent / "data" / "market"
SOURCES_FILE = MARKET_DIR / "sources.json"
CALIBRATION_FILE = MARKET_DIR / "calibration.csv"
SOURCE_CREDIT = "Source: ECB statistics."
# The equity index belongs to its provider, so only statistics calculated from it are published
LOCAL_ONLY_SERIES = {"equity_index"}
ECB_API = "https://data-api.ecb.europa.eu/service/data"

SERIES = {
    "yield_20y": {
        "key": "YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SR_20Y",
        "start": "2004-09-01",
        "description": "Euro area AAA government bond yield curve, 20-year spot rate, % a year",
    },
    "inflation": {
        "key": "ICP/M.U2.N.000000.4.ANR",
        "start": "1997-01",
        "description": "Euro area HICP inflation, annual rate of change, %",
    },
    "equity_index": {
        "key": "FM/M.U2.EUR.DS.EI.DJES50I.HSTA",
        "start": "2004-09",
        "description": "EURO STOXX 50 price index, monthly value, excludes dividends",
    },
}


def series_url(name):
    """Return the ECB Data Portal address of one series."""
    series = SERIES[name]
    return f"{ECB_API}/{series['key']}?format=csvdata&startPeriod={series['start']}"


def parse_ecb_csv(text):
    """Return the date and value columns from the text of an ECB CSV download."""
    table = pd.read_csv(io.StringIO(text), usecols=["TIME_PERIOD", "OBS_VALUE"])
    return table.rename(columns={"TIME_PERIOD": "date", "OBS_VALUE": "value"}).dropna()


def download_all():
    """Save every series as a small CSV and record its source, download date and file hash."""
    MARKET_DIR.mkdir(parents=True, exist_ok=True)
    sources = {}
    for name, series in SERIES.items():
        url = series_url(name)
        request = urllib.request.Request(url, headers={"Accept": "text/csv"})
        with urllib.request.urlopen(request, timeout=60) as response:
            table = parse_ecb_csv(response.read().decode("utf-8"))

        path = MARKET_DIR / f"{name}.csv"
        table.to_csv(path, index=False)
        sources[name] = {
            "description": series["description"],
            "publisher": "European Central Bank Data Portal",
            "url": url,
            "download_date": date.today().isoformat(),
            "first_date": str(table["date"].iloc[0]),
            "last_date": str(table["date"].iloc[-1]),
            "rows": len(table),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    SOURCES_FILE.write_text(json.dumps(sources, indent=2) + "\n", encoding="utf-8")
    return sources


def load_sources():
    """Return the saved record of where each series came from."""
    return json.loads(SOURCES_FILE.read_text(encoding="utf-8"))


def month_end_values(table, download_date):
    """Return one value per complete month: the last one seen in that month."""
    values = pd.Series(table["value"].to_numpy(dtype=float),
                       index=pd.to_datetime(table["date"])).sort_index()
    monthly = values.resample("ME").last().dropna()
    # The month of the download isn't finished, so its last value isn't a month-end value
    download_month_start = pd.Timestamp(download_date).replace(day=1)
    return monthly[monthly.index < download_month_start]


def load_market_data():
    """Return rates, inflation and the equity index by month, with rates as decimals."""
    sources = load_sources()
    monthly = {}
    for name in SERIES:
        table = pd.read_csv(MARKET_DIR / f"{name}.csv")
        monthly[name] = month_end_values(table, sources[name]["download_date"])
    return pd.DataFrame({
        "rate": monthly["yield_20y"] / 100,
        "inflation": monthly["inflation"] / 100,
        "equity_index": monthly["equity_index"],
    })


def save_calibration():
    """Recalculate the calibration from the local data and store the results."""
    table = calibration_table(load_market_data())
    table.to_csv(CALIBRATION_FILE, index=False)
    return table


def load_calibration():
    """Return the stored calibration, which needs no raw index data."""
    return pd.read_csv(CALIBRATION_FILE)


if __name__ == "__main__":
    for name, record in download_all().items():
        print(f"{name:<14} {record['first_date']} to {record['last_date']}, {record['rows']} rows")
    save_calibration()
