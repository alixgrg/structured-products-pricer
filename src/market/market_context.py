"""Helpers to build portfolio market contexts from market datasets."""

from __future__ import annotations

import pandas as pd


def build_spot_by_underlying(
    option_quotes: pd.DataFrame,
    *,
    underlying_column: str = "underlying",
    spot_column: str = "underlying_price",
    date_column: str = "valuation_date",
) -> dict[str, float]:
    """Return latest available spot by underlying from normalized option quotes."""
    if option_quotes is None or option_quotes.empty:
        return {}

    required = {underlying_column, spot_column}
    missing = required.difference(option_quotes.columns)
    if missing:
        raise ValueError(f"Missing columns for spot extraction: {sorted(missing)}")

    data = option_quotes.copy()
    data[underlying_column] = data[underlying_column].astype("string").str.strip().str.upper()
    data[spot_column] = pd.to_numeric(data[spot_column], errors="coerce")
    data = data.dropna(subset=[underlying_column, spot_column])
    data = data[data[spot_column] > 0.0]

    if date_column in data.columns:
        data[date_column] = pd.to_datetime(data[date_column], errors="coerce")
        data = data.sort_values([underlying_column, date_column])
    else:
        data = data.sort_values([underlying_column])

    return data.groupby(underlying_column)[spot_column].last().astype(float).to_dict()


def build_volatility_by_underlying(
    calibrated_quotes: pd.DataFrame,
    *,
    underlying_column: str = "underlying",
    iv_column: str = "implied_vol",
) -> dict[str, float]:
    """Return median implied volatility by underlying from calibrated quotes."""
    if calibrated_quotes is None or calibrated_quotes.empty:
        return {}

    required = {underlying_column, iv_column}
    missing = required.difference(calibrated_quotes.columns)
    if missing:
        raise ValueError(f"Missing columns for volatility extraction: {sorted(missing)}")

    data = calibrated_quotes.copy()
    data[underlying_column] = data[underlying_column].astype("string").str.strip().str.upper()
    data[iv_column] = pd.to_numeric(data[iv_column], errors="coerce")
    data = data.dropna(subset=[underlying_column, iv_column])
    data = data[data[iv_column] > 0.0]

    return data.groupby(underlying_column)[iv_column].median().astype(float).to_dict()
