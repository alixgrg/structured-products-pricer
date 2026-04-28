"""Shared pricing input resolution helpers."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from src.market.market_data import MarketData
from src.rates.yield_curve import YieldCurve


class VolatilitySurfaceLike(Protocol):
    def volatility(
        self,
        maturity: float | np.ndarray,
        log_moneyness: float | np.ndarray,
    ) -> float | np.ndarray:
        ...


def require_market_spot(market_data: MarketData | None) -> float:
    """Return the market spot or raise a pricing-friendly error."""
    if market_data is None or market_data.spot is None:
        raise ValueError("market_data.spot is required.")
    spot = float(market_data.spot)
    if spot <= 0.0:
        raise ValueError("market_data.spot must be strictly positive.")
    return spot


def resolve_pricing_rate(
    *,
    maturity: float | None = None,
    yield_curve: YieldCurve | None = None,
    model_rate: float | None = None,
    market_data: MarketData | None = None,
) -> float:
    """Resolve the pricing rate.

    Priority:
    1. market_data.rate: line-level / dashboard input
    2. yield_curve: calibrated curve fallback
    3. model_rate: static default fallback
    """
    if market_data is not None and market_data.rate is not None:
        return float(market_data.rate)

    if yield_curve is not None:
        if maturity is None:
            raise ValueError("maturity is required when resolving a rate from a yield curve.")
        return float(yield_curve.zero_rate(maturity))

    if model_rate is not None:
        return float(model_rate)

    raise ValueError("No rate available. Provide market_data.rate, yield_curve, or model.rate.")


def resolve_pricing_volatility(
    *,
    maturity: float | None = None,
    log_moneyness: float | None = None,
    volatility_surface: VolatilitySurfaceLike | None = None,
    model_volatility: float | None = None,
    market_data: MarketData | None = None,
) -> float:
    """Resolve the pricing volatility.

    Priority:
    1. market_data.volatility: line-level / underlying-level input
    2. volatility_surface: calibrated surface fallback
    3. model_volatility: static default fallback
    """
    if market_data is not None and market_data.volatility is not None:
        return _positive_float(market_data.volatility, "market_data.volatility")

    if volatility_surface is not None:
        if maturity is None or log_moneyness is None:
            raise ValueError(
                "maturity and log_moneyness are required when resolving volatility from a surface."
            )
        surface_vol = float(volatility_surface.volatility(maturity, log_moneyness))
        return _positive_float(surface_vol, "surface volatility")

    if model_volatility is not None:
        return _positive_float(model_volatility, "model.volatility")

    raise ValueError(
        "No volatility available. Provide market_data.volatility, volatility_surface, or model.volatility."
    )


def resolve_dividend_yield(
    *,
    product_dividend_yield: float | None = None,
    model_dividend_yield: float = 0.0,
    market_data: MarketData | None = None,
) -> float:
    """Resolve dividend yield.

    Priority:
    1. product.dividend_yield when explicitly set
    2. market_data.dividend_yield
    3. model.dividend_yield
    """
    if product_dividend_yield is not None:
        return float(product_dividend_yield)

    if market_data is not None:
        return float(market_data.dividend_yield)

    return float(model_dividend_yield)


def _positive_float(value: float, field_name: str) -> float:
    result = float(value)
    if result <= 0.0:
        raise ValueError(f"{field_name} must be strictly positive.")
    return result


__all__ = [
    "VolatilitySurfaceLike",
    "require_market_spot",
    "resolve_dividend_yield",
    "resolve_pricing_rate",
    "resolve_pricing_volatility",
]