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
├── calibration/
├── dashboard/
├── factory/
├── market/
├── models/
├── portfolio/
├── products/
└── risk/
```