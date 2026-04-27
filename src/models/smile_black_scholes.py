"""Black-Scholes model backed by a calibrated volatility surface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from src.market.market_data import MarketData
from src.models.base_model import PricingModel
from src.models.black_scholes import BlackScholesResult, black_scholes_price_and_greeks
from src.products.vanilla_option import VanillaOption
from src.rates.yield_curve import YieldCurve


class VolatilitySurfaceLike(Protocol):
    def volatility(self, maturity: float | np.ndarray, log_moneyness: float | np.ndarray) -> float | np.ndarray:
        ...


@dataclass(frozen=True, slots=True)
class SmileBlackScholesModel(PricingModel):
    """European vanilla pricer using SVI/SSVI/interpolated implied vol."""

    volatility_surface: VolatilitySurfaceLike
    rate: float | None = None
    dividend_yield: float = 0.0
    yield_curve: YieldCurve | None = None

    def price(self, product, market_data: MarketData | None = None) -> float:
        return self.price_and_greeks(product, market_data).price

    def risk(self, product, market_data: MarketData | None = None) -> dict[str, float]:
        result = self.price_and_greeks(product, market_data)
        return {
            "price": result.price,
            "delta": result.delta,
            "gamma": result.gamma,
            "vega": result.vega,
            "theta": result.theta,
            "rho": result.rho,
        }

    def price_and_greeks(self, product, market_data: MarketData | None = None) -> BlackScholesResult:
        if not isinstance(product, VanillaOption):
            raise TypeError("SmileBlackScholesModel supports VanillaOption only.")

        spot = self._resolve_spot(market_data)
        rate = self._resolve_rate(product, market_data)
        dividend_yield = self._resolve_dividend_yield(product, market_data)
        log_moneyness = float(np.log(product.strike / spot))
        volatility = float(self.volatility_surface.volatility(product.maturity, log_moneyness))
        if volatility <= 0.0:
            raise ValueError("surface volatility must be strictly positive.")

        return black_scholes_price_and_greeks(
            option_type=product.option_type,
            spot=spot,
            strike=product.strike,
            maturity=product.maturity,
            rate=rate,
            volatility=volatility,
            dividend_yield=dividend_yield,
            notional=product.notional,
        )

    def _resolve_spot(self, market_data: MarketData | None) -> float:
        if market_data is None or market_data.spot is None:
            raise ValueError("market_data.spot is required.")
        return float(market_data.spot)

    def _resolve_rate(self, product: VanillaOption, market_data: MarketData | None) -> float:
        if self.yield_curve is not None:
            return float(self.yield_curve.zero_rate(product.maturity))
        if self.rate is not None:
            return float(self.rate)
        if market_data is not None and market_data.rate is not None:
            return float(market_data.rate)
        raise ValueError("No rate available. Provide model.rate, yield_curve, or market_data.rate.")

    def _resolve_dividend_yield(self, product: VanillaOption, market_data: MarketData | None) -> float:
        if product.dividend_yield is not None:
            return float(product.dividend_yield)
        if market_data is not None:
            return float(market_data.dividend_yield)
        return float(self.dividend_yield)


__all__ = ["SmileBlackScholesModel"]
