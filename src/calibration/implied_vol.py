"""Implied-volatility calibration and surface interpolation.

This module provides:
- option panel cleaning for calibration use,
- Black-Scholes implied-volatility inversion,
- cross-sectional panel calibration,
- smile/surface interpolation helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite

import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.optimize import brentq

from src.calibration.base import CalibrationResult
from src.models.black_scholes import black_scholes_price_and_greeks


_PANEL_REQUIRED_COLUMNS = {
    "option_type",
    "strike",
    "underlying_price",
    "time_to_maturity_years",
}


@dataclass(frozen=True, slots=True)
class ImpliedVolSurface:
    """Interpolated implied-volatility surface in (maturity, log-moneyness)."""

    _linear: LinearNDInterpolator
    _nearest: NearestNDInterpolator

    @classmethod
    def from_quotes(
        cls,
        quotes: pd.DataFrame,
        *,
        maturity_column: str = "time_to_maturity_years",
        log_moneyness_column: str = "log_moneyness",
        iv_column: str = "implied_vol",
    ) -> "ImpliedVolSurface":
        required = {maturity_column, log_moneyness_column, iv_column}
        missing = required.difference(quotes.columns)
        if missing:
            raise ValueError(f"Missing required columns for surface build: {sorted(missing)}")

        clean = quotes[list(required)].dropna().copy()
        clean = clean[clean[iv_column] > 0.0]

        if len(clean) < 3:
            raise ValueError("At least three calibrated points are required to build a surface.")

        points = np.column_stack(
            [
                clean[maturity_column].to_numpy(dtype=float),
                clean[log_moneyness_column].to_numpy(dtype=float),
            ]
        )
        values = clean[iv_column].to_numpy(dtype=float)

        return cls(
            _linear=LinearNDInterpolator(points, values, fill_value=np.nan),
            _nearest=NearestNDInterpolator(points, values),
        )

    def evaluate(
        self,
        maturity: float | np.ndarray,
        log_moneyness: float | np.ndarray,
    ) -> float | np.ndarray:
        """Evaluate interpolated IV, with nearest-neighbor fallback outside the convex hull."""
        maturity_arr = np.asarray(maturity, dtype=float)
        moneyness_arr = np.asarray(log_moneyness, dtype=float)

        linear_values = self._linear(maturity_arr, moneyness_arr)
        nearest_values = self._nearest(maturity_arr, moneyness_arr)
        values = np.where(np.isnan(linear_values), nearest_values, linear_values)

        if np.isscalar(maturity) and np.isscalar(log_moneyness):
            return float(values)

        return np.asarray(values, dtype=float)


def clean_option_panel(
    option_quotes: pd.DataFrame,
    *,
    price_column: str = "market_price",
    min_maturity_years: float = 7.0 / 365.25,
    max_bid_ask_spread_ratio: float = 0.60,
    min_price: float = 1e-4,
) -> pd.DataFrame:
    """Prepare normalized option quotes for implied-vol calibration.

    The function keeps only usable rows and creates:
    - market_price,
    - moneyness,
    - log_moneyness.
    """
    missing = _PANEL_REQUIRED_COLUMNS.difference(option_quotes.columns)
    if missing:
        raise ValueError(f"Missing required columns for panel cleaning: {sorted(missing)}")

    panel = option_quotes.copy()

    for column in ("strike", "underlying_price", "time_to_maturity_years", "bid", "ask", "mid", "last"):
        if column in panel.columns:
            panel[column] = pd.to_numeric(panel[column], errors="coerce")

    panel["option_type"] = panel["option_type"].astype("string").str.strip().str.lower()
    panel["option_type"] = panel["option_type"].replace({"c": "call", "p": "put"})

    if "mid" in panel.columns:
        panel[price_column] = panel["mid"]
    else:
        panel[price_column] = np.nan

    if "last" in panel.columns:
        panel[price_column] = panel[price_column].fillna(panel["last"])

    if {"bid", "ask"}.issubset(panel.columns):
        panel[price_column] = panel[price_column].fillna((panel["bid"] + panel["ask"]) / 2.0)

    panel = panel[
        panel["option_type"].isin(["call", "put"])
        & (panel["strike"] > 0.0)
        & (panel["underlying_price"] > 0.0)
        & (panel["time_to_maturity_years"] >= min_maturity_years)
        & (panel[price_column] >= min_price)
    ].copy()

    panel["moneyness"] = panel["strike"] / panel["underlying_price"]
    panel["log_moneyness"] = np.log(panel["moneyness"])

    if {"bid", "ask"}.issubset(panel.columns):
        panel["bid_ask_spread_ratio"] = (panel["ask"] - panel["bid"]) / panel[price_column]
        panel = panel[
            panel["bid_ask_spread_ratio"].isna()
            | (panel["bid_ask_spread_ratio"] <= max_bid_ask_spread_ratio)
        ].copy()

    panel = panel[np.isfinite(panel["log_moneyness"])].copy()

    # Remove obvious static-arbitrage violations.
    intrinsic = np.where(
        panel["option_type"].to_numpy() == "call",
        np.maximum(panel["underlying_price"] - panel["strike"], 0.0),
        np.maximum(panel["strike"] - panel["underlying_price"], 0.0),
    )
    panel = panel[panel[price_column] >= intrinsic - 1e-10].copy()

    sort_columns = [
        column
        for column in (
            "valuation_date",
            "underlying",
            "time_to_maturity_years",
            "option_type",
            "strike",
        )
        if column in panel.columns
    ]
    panel = panel.sort_values(by=sort_columns, ignore_index=True)

    return panel


def implied_volatility_from_price(
    *,
    option_type: str,
    market_price: float,
    spot: float,
    strike: float,
    maturity: float,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
    sigma_lower: float = 1e-6,
    sigma_upper: float = 3.0,
    tolerance: float = 1e-10,
    max_iterations: int = 200,
) -> float:
    """Invert Black-Scholes price to implied volatility.

    Returns NaN if the quote is not invertible under no-arbitrage bounds.
    """
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'.")

    if spot <= 0.0 or strike <= 0.0 or maturity <= 0.0 or market_price <= 0.0:
        return float("nan")

    discount_rate = exp(-rate * maturity)
    discount_dividend = exp(-dividend_yield * maturity)

    if option_type == "call":
        lower_bound = max(spot * discount_dividend - strike * discount_rate, 0.0)
        upper_bound = spot * discount_dividend
    else:
        lower_bound = max(strike * discount_rate - spot * discount_dividend, 0.0)
        upper_bound = strike * discount_rate

    if market_price < lower_bound - 1e-10 or market_price > upper_bound + 1e-10:
        return float("nan")

    def objective(volatility: float) -> float:
        return (
            black_scholes_price_and_greeks(
                option_type=option_type,
                spot=spot,
                strike=strike,
                maturity=maturity,
                rate=rate,
                volatility=volatility,
                dividend_yield=dividend_yield,
            ).price
            - market_price
        )

    try:
        f_low = objective(sigma_lower)
        if abs(f_low) <= tolerance:
            return float(sigma_lower)

        current_upper = sigma_upper
        f_high = objective(current_upper)

        for _ in range(8):
            if f_low * f_high <= 0.0:
                break
            current_upper *= 2.0
            f_high = objective(current_upper)

        if f_low * f_high > 0.0:
            return float("nan")

        iv = brentq(
            objective,
            sigma_lower,
            current_upper,
            xtol=tolerance,
            maxiter=max_iterations,
        )
        return float(iv)
    except (ValueError, RuntimeError, OverflowError):
        return float("nan")


def calibrate_implied_vol_panel(
    option_quotes: pd.DataFrame,
    *,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
    rate_column: str | None = None,
    dividend_yield_column: str | None = None,
    price_column: str = "market_price",
    drop_outliers: bool = True,
    outlier_iqr_scale: float = 3.0,
) -> tuple[pd.DataFrame, CalibrationResult]:
    """Calibrate implied vols quote-by-quote on a cleaned option panel."""
    panel = clean_option_panel(option_quotes, price_column=price_column)

    def row_rate(row: pd.Series) -> float:
        if rate_column is not None and rate_column in panel.columns and pd.notna(row[rate_column]):
            return float(row[rate_column])
        return float(rate)

    def row_dividend(row: pd.Series) -> float:
        if (
            dividend_yield_column is not None
            and dividend_yield_column in panel.columns
            and pd.notna(row[dividend_yield_column])
        ):
            return float(row[dividend_yield_column])
        return float(dividend_yield)

    panel["implied_vol"] = panel.apply(
        lambda row: implied_volatility_from_price(
            option_type=str(row["option_type"]),
            market_price=float(row[price_column]),
            spot=float(row["underlying_price"]),
            strike=float(row["strike"]),
            maturity=float(row["time_to_maturity_years"]),
            rate=row_rate(row),
            dividend_yield=row_dividend(row),
        ),
        axis=1,
    )

    panel = panel[np.isfinite(panel["implied_vol"]) & (panel["implied_vol"] > 0.0)].copy()

    if drop_outliers and not panel.empty:
        panel = _drop_iv_outliers(panel, outlier_iqr_scale=outlier_iqr_scale)

    if panel.empty:
        result = CalibrationResult(
            model_name="black_scholes_implied_vol",
            parameters={"quote_count": 0.0, "median_implied_vol": float("nan")},
            objective_value=float("nan"),
        )
        return panel, result

    panel["repriced"] = panel.apply(
        lambda row: black_scholes_price_and_greeks(
            option_type=str(row["option_type"]),
            spot=float(row["underlying_price"]),
            strike=float(row["strike"]),
            maturity=float(row["time_to_maturity_years"]),
            rate=row_rate(row),
            volatility=float(row["implied_vol"]),
            dividend_yield=row_dividend(row),
        ).price,
        axis=1,
    )

    panel["calibration_error"] = panel["repriced"] - panel[price_column]
    rmse = float(np.sqrt(np.mean(np.square(panel["calibration_error"]))))

    result = CalibrationResult(
        model_name="black_scholes_implied_vol",
        parameters={
            "quote_count": float(len(panel)),
            "median_implied_vol": float(panel["implied_vol"].median()),
            "max_abs_error": float(panel["calibration_error"].abs().max()),
        },
        objective_value=rmse,
    )

    return panel.reset_index(drop=True), result


def _drop_iv_outliers(panel: pd.DataFrame, *, outlier_iqr_scale: float) -> pd.DataFrame:
    """Drop extreme implied vols within each option type / maturity bucket."""
    work = panel.copy()
    work["maturity_bucket"] = work["time_to_maturity_years"].round(2)

    mask = pd.Series(True, index=work.index)

    for _, group in work.groupby(["option_type", "maturity_bucket"]):
        if len(group) < 6:
            continue

        q1 = group["implied_vol"].quantile(0.25)
        q3 = group["implied_vol"].quantile(0.75)
        iqr = q3 - q1

        if not isfinite(iqr) or iqr <= 0.0:
            continue

        lower = q1 - outlier_iqr_scale * iqr
        upper = q3 + outlier_iqr_scale * iqr
        mask.loc[group.index] = group["implied_vol"].between(lower, upper)

    return work[mask].drop(columns=["maturity_bucket"]).reset_index(drop=True)


def build_surface_grid(
    surface: ImpliedVolSurface,
    *,
    maturity_grid: np.ndarray,
    log_moneyness_grid: np.ndarray,
) -> pd.DataFrame:
    """Evaluate a surface on a rectangular grid and return a tidy dataframe."""
    t_mesh, k_mesh = np.meshgrid(maturity_grid, log_moneyness_grid, indexing="ij")
    iv_mesh = surface.evaluate(t_mesh, k_mesh)

    return pd.DataFrame(
        {
            "maturity": t_mesh.ravel(),
            "log_moneyness": k_mesh.ravel(),
            "implied_vol": np.asarray(iv_mesh).ravel(),
        }
    )


def build_smile_slice(
    calibrated_quotes: pd.DataFrame,
    *,
    target_maturity: float,
    tolerance: float = 20.0 / 365.25,
) -> pd.DataFrame:
    """Return one smile slice around a target maturity."""
    if "implied_vol" not in calibrated_quotes.columns:
        raise ValueError("calibrated_quotes must contain an 'implied_vol' column.")

    selected = calibrated_quotes[
        (calibrated_quotes["time_to_maturity_years"] - target_maturity).abs() <= tolerance
    ].copy()

    return selected.sort_values("log_moneyness", ignore_index=True)


__all__ = [
    "ImpliedVolSurface",
    "build_smile_slice",
    "build_surface_grid",
    "calibrate_implied_vol_panel",
    "clean_option_panel",
    "implied_volatility_from_price",
]
