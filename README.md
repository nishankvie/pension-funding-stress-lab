# Pension Funding Stress Lab

A student project that follows one path from messy employee data to investment risk:

data checks → yearly future payments → pension obligation → sample portfolios → stress tests and
linked simulations → what needs attention → report → audit trail

**All employees are synthetic and all portfolios are illustrative. This is not an official
valuation, a product or investment advice.** Almost every assumption is illustrative; see
`config/assumptions.yaml`, where each one has a label, source, date and reason.

Built with AI coding tools. The logic and results were checked by the author.

## Run it

```bash
pip install -r requirements-dev.txt
python -m pytest
streamlit run pension_lab_app.py
```

To save a run record, report and CSV files in `outputs/`:

```bash
python -m src.pension_lab.audit
```

## Public data

Source: ECB statistics. Rates and inflation are stored in `data/market/`, with source and download
date in `sources.json`.

The equity index belongs to its provider, so the raw series is not published here. Only statistics
calculated from it are stored, in `calibration.csv`. To recreate it locally:

```bash
python -m src.pension_lab.market_data
```

Without that file, the tests that need it are skipped and say so.
