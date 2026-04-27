# Market Foundations Validation Notes

This note records what is validated in the market-foundations milestone and
what remains limited by the available course datasets. No additional ECB or
external market data is downloaded.

## Validated Scope

- **Loaders**: `data/raw/rate_curves.parquet`, `data/raw/options.csv` and
  `data/raw/inventory.xlsx` are preferred when present. The original course
  paths remain a fallback.
- **Rates**: the historical rate-curve file validates parsing, tenor
  normalization, date/country selection, zero-rate interpolation and positive
  discount factors. Deposit/FRA/swap bootstrapping is validated on a controlled
  synthetic quote set because raw money-market/swap quotes are not available in
  the course files.
- **Volatility**: market option prices are cleaned, converted to implied vols
  and calibrated one underlying at a time. The QA notebook selects the
  underlying with the most stable SVI/SSVI repricing diagnostics.
- **Arbitrage checks**: SSVI is the decision surface when calendar and
  butterfly checks pass. Raw SVI remains a benchmark and can stay in `WARN` if
  the noisy short-dated panel violates calendar monotonicity.
- **Market repricing**: the notebook validates vanilla repricing on a
  deterministic train/test split inside the available option panel.
- **Portfolio ingestion**: inventory rows are loaded and aggregations are
  checked. Unsupported products are explicitly flagged instead of silently
  priced.

## Data Limits To Mention In The Report

- The rate data is already a curve panel by country/date/tenor. It is suitable
  for historical ZC curve construction, but it does not prove that the
  bootstrapping engine can consume real deposit/FRA/swap quotes from the course
  data.
- The options file has short maturities and missing implied-volatility fields.
  Implied vols are therefore inferred from prices, and filters on bid/ask
  spread and moneyness are part of the validation methodology.
- A single volatility surface must not mix AAPL, NVDA, META, TSLA, etc. The QA
  selection currently chooses NVDA because it gives the cleanest SSVI diagnostics
  on the filtered panel.
- Repricing errors are useful sanity checks, but they are not a full market
  model validation across dates, maturities and regimes.
- Autocalls and swaps are present in the inventory. Their full event-driven
  pricing is a next milestone, not part of this market-foundations closure.

## Current QA Interpretation

Expected stabilized status:
- `FAIL = 0`;
- `WARN` accepted only for documented limits, currently raw SVI calendar
  diagnostics and unsupported portfolio product scope;
- SSVI, curve construction, loader checks and vanilla repricing should pass.
