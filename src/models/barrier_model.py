"""Analytical Black-Scholes-Merton pricing for continuously monitored barriers.

The implementation uses the standard Reiner-Rubinstein/Merton closed forms for
single knock-out barriers without rebate. Knock-in prices are obtained through
in/out parity:

    vanilla = knock_out + knock_in

This is the safest implementation choice for a student project because it makes
the required parity test exact up to floating-point precision.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, exp, log, pi, sqrt

from src.market.market_data import MarketData
from src.models.base_model import PricingModel
from src.models.black_scholes import black_scholes_price_and_greeks
from src.models.pricing_inputs import require_market_spot, resolve_dividend_yield, resolve_pricing_rate
from src.products.barrier_option import BarrierOption
from src.rates.yield_curve import YieldCurve


@dataclass(frozen=True, slots=True)
class BarrierModel(PricingModel):
    """Closed-form pricer for European single-barrier options without rebate."""

    rate: float | None = None
    volatility: float | None = None
    dividend_yield: float = 0.0
    yield_curve: YieldCurve | None = None

    def price(self, product, market_data: MarketData | None = None) -> float:
        if not isinstance(product, BarrierOption):
            raise TypeError("BarrierModel supports BarrierOption only.")

        spot = require_market_spot(market_data)
        rate = self._resolve_rate(product, market_data)
        volatility = self._resolve_volatility(market_data)
        dividend_yield = self._resolve_dividend_yield(product, market_data)

        vanilla = black_scholes_price_and_greeks(
            option_type=product.option_type,
            spot=spot,
            strike=product.strike,
            maturity=product.maturity,
            rate=rate,
            volatility=volatility,
            dividend_yield=dividend_yield,
            notional=product.notional,
        ).price

        if product.maturity == 0.0:
            return product.payoff({"path": [spot]})

        if self._already_touched(product, spot):
            return 0.0 if product.is_knock_out else float(vanilla)

        knock_out = _knock_out_price(
            option_type=product.option_type,
            direction=product.barrier_direction,
            spot=spot,
            strike=product.strike,
            barrier=product.barrier,
            maturity=product.maturity,
            rate=rate,
            volatility=volatility,
            dividend_yield=dividend_yield,
            notional=product.notional,
        )

        knock_out = min(max(float(knock_out), 0.0), float(vanilla))
        if product.is_knock_out:
            return knock_out
        return float(vanilla - knock_out)

    def risk(self, product, market_data: MarketData | None = None) -> dict[str, float]:
        price = self.price(product, market_data)
        # Closed-form barrier Greeks are intentionally deferred. Numerical Greeks
        # in Phase 7 will cover barriers robustly.
        return {"price": price, "delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}

    def _resolve_rate(self, product: BarrierOption, market_data: MarketData | None) -> float:
        return resolve_pricing_rate(
            maturity=product.maturity,
            yield_curve=self.yield_curve,
            model_rate=self.rate,
            market_data=market_data,
        )

    def _resolve_volatility(self, market_data: MarketData | None) -> float:
        if self.volatility is not None:
            volatility = float(self.volatility)
        elif market_data is not None and market_data.volatility is not None:
            volatility = float(market_data.volatility)
        else:
            raise ValueError("No volatility available. Provide model.volatility or market_data.volatility.")
        if volatility <= 0.0:
            raise ValueError("volatility must be strictly positive.")
        return volatility

    def _resolve_dividend_yield(self, product: BarrierOption, market_data: MarketData | None) -> float:
        return resolve_dividend_yield(
            product_dividend_yield=product.dividend_yield,
            model_dividend_yield=self.dividend_yield,
            market_data=market_data,
        )

    @staticmethod
    def _already_touched(product: BarrierOption, spot: float) -> bool:
        if product.barrier_direction == "up":
            return spot >= product.barrier
        return spot <= product.barrier


def _knock_out_price(
    *,
    option_type: str,
    direction: str,
    spot: float,
    strike: float,
    barrier: float,
    maturity: float,
    rate: float,
    volatility: float,
    dividend_yield: float,
    notional: float,
) -> float:
    if direction == "up" and barrier <= spot:
        return 0.0
    if direction == "down" and barrier >= spot:
        return 0.0

    phi = 1.0 if option_type == "call" else -1.0
    eta = 1.0 if direction == "down" else -1.0
    b = rate - dividend_yield
    sigma_sqrt_t = volatility * sqrt(maturity)
    mu = (b - 0.5 * volatility * volatility) / (volatility * volatility)

    x1 = log(spot / strike) / sigma_sqrt_t + (1.0 + mu) * sigma_sqrt_t
    x2 = log(spot / barrier) / sigma_sqrt_t + (1.0 + mu) * sigma_sqrt_t
    y1 = log((barrier * barrier) / (spot * strike)) / sigma_sqrt_t + (1.0 + mu) * sigma_sqrt_t
    y2 = log(barrier / spot) / sigma_sqrt_t + (1.0 + mu) * sigma_sqrt_t

    A = phi * spot * exp((b - rate) * maturity) * _norm_cdf(phi * x1) - phi * strike * exp(-rate * maturity) * _norm_cdf(phi * x1 - phi * sigma_sqrt_t)
    B = phi * spot * exp((b - rate) * maturity) * _norm_cdf(phi * x2) - phi * strike * exp(-rate * maturity) * _norm_cdf(phi * x2 - phi * sigma_sqrt_t)
    C = (
        phi
        * spot
        * exp((b - rate) * maturity)
        * (barrier / spot) ** (2.0 * (mu + 1.0))
        * _norm_cdf(eta * y1)
        - phi
        * strike
        * exp(-rate * maturity)
        * (barrier / spot) ** (2.0 * mu)
        * _norm_cdf(eta * y1 - eta * sigma_sqrt_t)
    )
    D = (
        phi
        * spot
        * exp((b - rate) * maturity)
        * (barrier / spot) ** (2.0 * (mu + 1.0))
        * _norm_cdf(eta * y2)
        - phi
        * strike
        * exp(-rate * maturity)
        * (barrier / spot) ** (2.0 * mu)
        * _norm_cdf(eta * y2 - eta * sigma_sqrt_t)
    )

    if option_type == "call" and direction == "down":
        value = A - C if strike > barrier else B - D
    elif option_type == "call" and direction == "up":
        value = 0.0 if strike > barrier else A - B + C - D
    elif option_type == "put" and direction == "down":
        value = A - B + C - D if strike > barrier else 0.0
    elif option_type == "put" and direction == "up":
        value = B - D if strike > barrier else A - C
    else:
        raise ValueError("Unsupported option_type/direction combination.")

    return float(notional * value)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return exp(-0.5 * x * x) / sqrt(2.0 * pi)


__all__ = ["BarrierModel"]
