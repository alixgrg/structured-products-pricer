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
- dashboard configuration scaffolding.

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
- SVI benchmark surface
- SSVI volatility surface

Optional extensions:
- Monte Carlo GBM
- Static replication model

## Project structure

```text
src/
|- calibration/
|- conventions/
|- dashboard/
|- factory/
|- market/
|- models/
|- portfolio/
|- rates/
|- products/
|- risk/
|- config.py
`- convention.py
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

- `src.market.loaders.load_rate_curves`: reads and normalizes `data/raw/rate_curves.parquet`.
- `src.market.loaders.load_option_quotes`: reads and normalizes `data/raw/options.csv` with the `;` separator.
- `src.portfolio.inventory_loader.load_inventory_workbook`: reads and normalizes all sheets from `data/raw/inventory.xlsx`.

By default, loaders first use the self-contained copies in `data/raw/` when
present, then fall back to the original course/project paths. This keeps the
notebooks reproducible without downloading external market data.

## Market Foundations Validation

The stabilized market-foundations layer is checked by
`notebooks/00_QA_clean_structured_products_pricer.ipynb` and exports QA tables
to `reports/qa/`.

Stable QA exports kept under version control:
- `qa_status_summary.csv`;
- `qa_repricing_summary.csv`;
- `qa_vol_filter_report.csv`;
- `qa_vol_surface_diagnostics.csv`;
- `qa_bootstrap_points.csv`;
- `qa_bootstrap_instrument_checks.csv`;
- `qa_portfolio_by_product.csv`;
- `qa_portfolio_by_portfolio.csv`.

Validated with the available course data:
- rate-curve loader, tenor normalization, zero-rate interpolation and
  positive discount factors on the historical curve file;
- deposit/FRA/swap bootstrapping, day-count/business-day handling and
  zero-coupon repricing on a controlled synthetic quote set;
- option quote loader, market-price construction, implied-volatility inversion
  from option prices and one-underlying volatility-surface calibration;
- SSVI arbitrage diagnostics, vanilla repricing error and portfolio ingestion
  / aggregation for currently supported product rows.

Known validation limits:
- the course rate file is a historical curve panel, not a raw
  deposit/FRA/swap instrument set. Bootstrapping is therefore validated on
  synthetic quotes, while the course file validates the historical curve loader
  and ZC consistency;
- the option panel is short-dated and noisy, with missing implied volatility.
  IVs are inferred from bid/mid/ask prices after liquidity and moneyness
  filters. The notebook selects the most stable single underlying for SSVI
  rather than forcing all underlyings into one incoherent surface;
- raw SVI is kept as a benchmark and may raise calendar-arbitrage warnings on
  noisy market data. SSVI is the preferred surface when butterfly and calendar
  checks pass;
- vanilla repricing is a practical train/test split inside the available panel,
  not a full multi-date out-of-sample market backtest;
- swaps and autocall schedules are loaded and flagged in the portfolio QA, but
  their full event-driven pricing is outside the market-foundations milestone.

## Notebooks

- `notebooks/00_QA_clean_structured_products_pricer.ipynb`: validation notebook to run before continuing development.
- `notebooks/99_benchmark_pricer_v2_appendix.ipynb`: cleaned benchmark appendix kept for traceability, not for reference QA exports.
