"""Market repricing validation for vanilla option panels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from src.calibration.base import CalibrationResult
from src.models.black_scholes import black_scholes_price_and_greeks


class VolatilitySurfaceLike(Protocol):
    """Protocol implemented by SVI/SSVI/interpolated vol surfaces."""

    def volatility(self, maturity: float | np.ndarray, log_moneyness: float | np.ndarray) -> float | np.ndarray:
        ...


@dataclass(frozen=True, slots=True)
class RepricingValidationResult:
    """Detailed repricing errors and aggregate summaries."""

    line_errors: pd.DataFrame
    summary: pd.DataFrame
    by_maturity: pd.DataFrame
    by_moneyness_bucket: pd.DataFrame
    calibration_result: CalibrationResult


def reprice_vanilla_market_quotes(
    quotes: pd.DataFrame,
    vol_surface: VolatilitySurfaceLike,
    *,
    price_column: str = "market_price",
    option_type_column: str = "option_type",
    spot_column: str = "underlying_price",
    strike_column: str = "strike",
    maturity_column: str = "time_to_maturity_years",
    log_moneyness_column: str = "log_moneyness",
    rate: float = 0.0,
    dividend_yield: float = 0.0,
    rate_column: str | None = None,
    dividend_yield_column: str | None = None,
) -> RepricingValidationResult:
    """Reprice market vanilla quotes and compute model errors.

    This function validates a calibrated surface by repricing the same vanilla
    option panel used for calibration or an out-of-sample panel.
    """
    required = {
        price_column,
        option_type_column,
        spot_column,
        strike_column,
        maturity_column,
        log_moneyness_column,
    }
    missing = required.difference(quotes.columns)
    if missing:
        raise ValueError(f"Missing columns for repricing validation: {sorted(missing)}")

    data = quotes.copy()
    data = data.dropna(subset=list(required)).copy()
    data = data[
        (data[price_column] > 0.0)
        & (data[spot_column] > 0.0)
        & (data[strike_column] > 0.0)
        & (data[maturity_column] > 0.0)
    ].copy()
    if data.empty:
        raise ValueError("No usable quotes for repricing validation.")

    def row_rate(row: pd.Series) -> float:
        if rate_column is not None and rate_column in data.columns and pd.notna(row[rate_column]):
            return float(row[rate_column])
        return float(rate)

    def row_dividend(row: pd.Series) -> float:
        if dividend_yield_column is not None and dividend_yield_column in data.columns and pd.notna(row[dividend_yield_column]):
            return float(row[dividend_yield_column])
        return float(dividend_yield)

    model_vols: list[float] = []
    model_prices: list[float] = []

    for _, row in data.iterrows():
        maturity = float(row[maturity_column])
        log_moneyness = float(row[log_moneyness_column])
        vol = float(vol_surface.volatility(maturity, log_moneyness))
        model_vols.append(vol)
        price = black_scholes_price_and_greeks(
            option_type=str(row[option_type_column]).lower(),
            spot=float(row[spot_column]),
            strike=float(row[strike_column]),
            maturity=maturity,
            rate=row_rate(row),
            volatility=vol,
            dividend_yield=row_dividend(row),
        ).price
        model_prices.append(price)

    data["model_vol"] = model_vols
    data["model_price"] = model_prices
    data["price_error"] = data["model_price"] - data[price_column]
    data["abs_price_error"] = data["price_error"].abs()
    data["relative_price_error"] = data["price_error"] / data[price_column]
    data["abs_relative_price_error"] = data["relative_price_error"].abs()

    if "implied_vol" in data.columns:
        data["vol_error"] = data["model_vol"] - data["implied_vol"]
        data["abs_vol_error"] = data["vol_error"].abs()
    else:
        data["vol_error"] = np.nan
        data["abs_vol_error"] = np.nan

    data["maturity_bucket"] = data[maturity_column].round(2)
    data["moneyness_bucket"] = pd.cut(
        np.exp(data[log_moneyness_column]),
        bins=[0.0, 0.8, 0.95, 1.05, 1.20, np.inf],
        labels=["deep_otm_put/itm_call", "otm_put/itm_call", "atm", "otm_call/itm_put", "deep_otm_call/itm_put"],
        include_lowest=True,
    )

    summary = _error_summary(data)
    by_maturity = _grouped_error_summary(data, "maturity_bucket")
    by_moneyness = _grouped_error_summary(data, "moneyness_bucket")

    rmse = float(np.sqrt(np.mean(np.square(data["price_error"]))))
    result = CalibrationResult(
        model_name="market_vanilla_repricing_validation",
        parameters={
            "quote_count": float(len(data)),
            "mae_price": float(data["abs_price_error"].mean()),
            "rmse_price": rmse,
            "max_abs_price_error": float(data["abs_price_error"].max()),
            "mae_vol": float(data["abs_vol_error"].mean()) if data["abs_vol_error"].notna().any() else float("nan"),
        },
        objective_value=rmse,
    )

    return RepricingValidationResult(
        line_errors=data.reset_index(drop=True),
        summary=summary,
        by_maturity=by_maturity,
        by_moneyness_bucket=by_moneyness,
        calibration_result=result,
    )


def _error_summary(data: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "quote_count": [float(len(data))],
            "mean_error": [float(data["price_error"].mean())],
            "mae": [float(data["abs_price_error"].mean())],
            "rmse": [float(np.sqrt(np.mean(np.square(data["price_error"]))))],
            "max_abs_error": [float(data["abs_price_error"].max())],
            "mean_abs_relative_error": [float(data["abs_relative_price_error"].mean())],
            "mean_abs_vol_error": [float(data["abs_vol_error"].mean()) if data["abs_vol_error"].notna().any() else float("nan")],
        }
    )


def _grouped_error_summary(data: pd.DataFrame, group_column: str) -> pd.DataFrame:
    grouped = data.groupby(group_column, dropna=False, observed=False)
    out = grouped.agg(
        quote_count=("price_error", "size"),
        mean_error=("price_error", "mean"),
        mae=("abs_price_error", "mean"),
        max_abs_error=("abs_price_error", "max"),
        mean_abs_relative_error=("abs_relative_price_error", "mean"),
        mean_abs_vol_error=("abs_vol_error", "mean"),
    ).reset_index()
    rmse = grouped["price_error"].apply(lambda x: float(np.sqrt(np.mean(np.square(x))))).rename("rmse").reset_index()
    return out.merge(rmse, on=group_column, how="left")


__all__ = [
    "RepricingValidationResult",
    "VolatilitySurfaceLike",
    "reprice_vanilla_market_quotes",
]
