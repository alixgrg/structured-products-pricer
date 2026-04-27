"""Shared pricing input resolution helpers."""

from __future__ import annotations

from src.market.market_data import MarketData
from src.rates.yield_curve import YieldCurve


def require_market_spot(market_data: MarketData | None) -> float:
    """Return the market spot or raise a pricing-friendly error."""
    if market_data is None or market_data.spot is None:
        raise ValueError("market_data.spot is required.")
    return float(market_data.spot)


def resolve_pricing_rate(
    *,
    maturity: float | None = None,
    yield_curve: YieldCurve | None = None,
    model_rate: float | None = None,
    market_data: MarketData | None = None,
) -> float:
    """Resolve the pricing rate from curve, model input, or market data."""
    if yield_curve is not None:
        if maturity is None:
            raise ValueError("maturity is required when resolving a rate from a yield curve.")
        return float(yield_curve.zero_rate(maturity))

    if model_rate is not None:
        return float(model_rate)

    if market_data is not None and market_data.rate is not None:
        return float(market_data.rate)

    raise ValueError("No rate available. Provide model.rate, yield_curve, or market_data.rate.")


def resolve_dividend_yield(
    *,
    product_dividend_yield: float | None = None,
    model_dividend_yield: float = 0.0,
    market_data: MarketData | None = None,
) -> float:
    """Resolve dividend yield with product-specific values first."""
    if product_dividend_yield is not None:
        return float(product_dividend_yield)

    if market_data is not None:
        return float(market_data.dividend_yield)

    return float(model_dividend_yield)

