# Structured Products Pricer

## Objective

This project aims to build a scalable Python application for pricing mono-underlying structured products.

The application supports:
- market data loading,
- rate curve calibration,
- implied volatility calibration,
- pricing of financial products,
- portfolio inventory management,
- risk aggregation,
- Streamlit dashboard.

## Products

Initial scope:
- Zero-Coupon Bond
- European Call / Put
- Call Spread
- Capital Protected Note
- Capped Capital Protected Note
- Reverse Convertible

Optional extensions:
- Barrier Option
- Bonus Certificate
- Simplified Autocall

## Models

Initial scope:
- Yield curve interpolation
- Nelson-Siegel curve fitting
- Black-Scholes pricing
- Implied volatility calibration
- Static replication model

Optional extensions:
- Monte Carlo GBM
- SSVI volatility surface

## Project structure

```text
src/
|- calibration/
|- dashboard/
|- factory/
|- market/
|- models/
|- portfolio/
|- products/
|- risk/
|- config.py
`- conventions.py
```

## Naming conventions

- Python packages, modules and functions use `snake_case`.
- Classes use `PascalCase`.
- All normalized tabular columns use `snake_case`.
- Dates are stored as timezone-naive pandas timestamps normalized at day granularity.
- Tenors such as `1M`, `6M`, `5Y` are kept as labels and enriched with a year-fraction column when relevant.
- Rate datasets expose both `rate_percent` and `rate_decimal`.

## Data flow

The repository follows a four-stage flow:

```text
external sources -> data/raw -> data/interim -> data/processed
```

- `external sources`: original course/project files kept outside the repo.
- `data/raw`: untouched copies staged inside the repo.
- `data/interim`: normalized flat files used by notebooks and pricing modules.
- `data/processed`: compact reporting tables ready for portfolio aggregation and dashboards.

## Current loaders

- `src.market.loaders.load_rate_curves`: reads and normalizes `1.rate_curves.parquet`.
- `src.market.loaders.load_option_quotes`: reads and normalizes `2.options.csv` with the `;` separator.
- `src.portfolio.inventory_loader.load_inventory_workbook`: reads and normalizes all sheets from `Inventaire.xlsx`.

## Notebooks

- `notebooks/00_architecture.ipynb`: architecture overview and data flow walkthrough.
- `notebooks/01_data_quality.ipynb`: first-pass market and inventory data quality checks.
- `notebooks/02_rates.ipynb`: curve construction, interpolation, discount factors and zero-coupon pricing.
- `notebooks/03_vanilla_pricing.ipynb`: vanilla call/put pricing, Greeks, parity and sensitivities.
- `notebooks/04_vol_calibration.ipynb`: implied-vol calibration, smiles, interpolated surface and calibration errors.
