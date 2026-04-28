"""Black-Scholes model for European vanilla options."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from math import erf, exp, log, pi, sqrt

from src.market.market_data import MarketData
from src.models.base_model import PricingModel
from src.models.pricing_inputs import (
    require_market_spot,
    resolve_dividend_yield,
    resolve_pricing_rate,
    resolve_pricing_volatility,
)
from src.products.vanilla_option import VanillaOption
from src.rates.yield_curve import YieldCurve


@dataclass(frozen=True, slots=True)
class BlackScholesResult:
    """Container for Black-Scholes price and Greeks."""

    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BlackScholesModel(PricingModel):
    """Black-Scholes model for European vanilla options.

    The implementation supports:
    - call and put,
    - continuous dividend yield,
    - constant rate or rate from a YieldCurve,
    - analytical Greeks.

    Greeks conventions:
    - delta: dV / dS
    - gamma: d²V / dS²
    - vega: dV / dσ, for a 1.00 volatility move
    - theta: calendar decay per year
    - rho: dV / dr, for a 1.00 rate move
    """

    rate: float | None = None
    volatility: float | None = None
    dividend_yield: float = 0.0
    yield_curve: YieldCurve | None = None

    def price(self, product, market_data: MarketData | None = None) -> float:
        """Return the option price."""
        return self.price_and_greeks(product, market_data).price

    def risk(self, product, market_data: MarketData | None = None) -> dict[str, float]:
        """Return Greeks as risk metrics."""
        result = self.price_and_greeks(product, market_data)

        return {
            "price": result.price,
            "delta": result.delta,
            "gamma": result.gamma,
            "vega": result.vega,
            "theta": result.theta,
            "rho": result.rho,
        }

    def greeks(self, product, market_data: MarketData | None = None) -> dict[str, float]:
        """Alias for risk metrics."""
        return self.risk(product, market_data)

    def price_and_greeks(
        self,
        product,
        market_data: MarketData | None = None,
    ) -> BlackScholesResult:
        """Return price and analytical Greeks."""
        if not isinstance(product, VanillaOption):
            raise TypeError("BlackScholesModel supports VanillaOption only.")

        spot = self._resolve_spot(market_data)
        rate = self._resolve_rate(product, market_data)
        volatility = self._resolve_volatility(market_data)
        dividend_yield = self._resolve_dividend_yield(product, market_data)

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
        return require_market_spot(market_data)

    def _resolve_rate(
        self,
        product: VanillaOption,
        market_data: MarketData | None,
    ) -> float:
        return resolve_pricing_rate(
            maturity=product.maturity,
            yield_curve=self.yield_curve,
            model_rate=self.rate,
            market_data=market_data,
        )

    def _resolve_volatility(self, market_data: MarketData | None) -> float:
        return resolve_pricing_volatility(
            model_volatility=self.volatility,
            market_data=market_data,
        )

    def _resolve_dividend_yield(
        self,
        product: VanillaOption,
        market_data: MarketData | None,
    ) -> float:
        return resolve_dividend_yield(
            product_dividend_yield=product.dividend_yield,
            model_dividend_yield=self.dividend_yield,
            market_data=market_data,
        )


def black_scholes_price_and_greeks(
    *,
    option_type: str,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
    notional: float = 1.0,
) -> BlackScholesResult:
    """Analytical Black-Scholes price and Greeks."""
    _validate_inputs(
        option_type=option_type,
        spot=spot,
        strike=strike,
        maturity=maturity,
        volatility=volatility,
        notional=notional,
    )

    if maturity == 0.0:
        return _expiry_result(
            option_type=option_type,
            spot=spot,
            strike=strike,
            notional=notional,
        )

    d_1 = d1(
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        volatility=volatility,
        dividend_yield=dividend_yield,
    )
    d_2 = d_1 - volatility * sqrt(maturity)

    discount_rate = exp(-rate * maturity)
    discount_dividend = exp(-dividend_yield * maturity)

    if option_type == "call":
        price = (
            spot * discount_dividend * normal_cdf(d_1)
            - strike * discount_rate * normal_cdf(d_2)
        )
        delta = discount_dividend * normal_cdf(d_1)
        theta = (
            -spot
            * discount_dividend
            * normal_pdf(d_1)
            * volatility
            / (2.0 * sqrt(maturity))
            - rate * strike * discount_rate * normal_cdf(d_2)
            + dividend_yield * spot * discount_dividend * normal_cdf(d_1)
        )
        rho = strike * maturity * discount_rate * normal_cdf(d_2)
    else:
        price = (
            strike * discount_rate * normal_cdf(-d_2)
            - spot * discount_dividend * normal_cdf(-d_1)
        )
        delta = discount_dividend * (normal_cdf(d_1) - 1.0)
        theta = (
            -spot
            * discount_dividend
            * normal_pdf(d_1)
            * volatility
            / (2.0 * sqrt(maturity))
            + rate * strike * discount_rate * normal_cdf(-d_2)
            - dividend_yield * spot * discount_dividend * normal_cdf(-d_1)
        )
        rho = -strike * maturity * discount_rate * normal_cdf(-d_2)

    gamma = (
        discount_dividend
        * normal_pdf(d_1)
        / (spot * volatility * sqrt(maturity))
    )
    vega = spot * discount_dividend * normal_pdf(d_1) * sqrt(maturity)

    multiplier = float(notional)

    return BlackScholesResult(
        price=float(multiplier * price),
        delta=float(multiplier * delta),
        gamma=float(multiplier * gamma),
        vega=float(multiplier * vega),
        theta=float(multiplier * theta),
        rho=float(multiplier * rho),
    )


def d1(
    *,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> float:
    """Return Black-Scholes d1."""
    return float(
        (
            log(spot / strike)
            + (rate - dividend_yield + 0.5 * volatility**2) * maturity
        )
        / (volatility * sqrt(maturity))
    )


def d2(
    *,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> float:
    """Return Black-Scholes d2."""
    return d1(
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        volatility=volatility,
        dividend_yield=dividend_yield,
    ) - volatility * sqrt(maturity)


def normal_pdf(x: float) -> float:
    """Standard normal probability density function."""
    return float(exp(-0.5 * x * x) / sqrt(2.0 * pi))


def normal_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    return float(0.5 * (1.0 + erf(x / sqrt(2.0))))


def _validate_inputs(
    *,
    option_type: str,
    spot: float,
    strike: float,
    maturity: float,
    volatility: float,
    notional: float,
) -> None:
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'.")
    if spot <= 0.0:
        raise ValueError("spot must be strictly positive.")
    if strike <= 0.0:
        raise ValueError("strike must be strictly positive.")
    if maturity < 0.0:
        raise ValueError("maturity must be non-negative.")
    if volatility <= 0.0:
        raise ValueError("volatility must be strictly positive.")
    if notional <= 0.0:
        raise ValueError("notional must be strictly positive.")


def _expiry_result(
    *,
    option_type: str,
    spot: float,
    strike: float,
    notional: float,
) -> BlackScholesResult:
    """Return price and simplified Greeks at expiry."""
    if option_type == "call":
        price = max(spot - strike, 0.0)
        delta = 1.0 if spot > strike else 0.0
    else:
        price = max(strike - spot, 0.0)
        delta = -1.0 if spot < strike else 0.0

    return BlackScholesResult(
        price=float(notional * price),
        delta=float(notional * delta),
        gamma=0.0,
        vega=0.0,
        theta=0.0,
        rho=0.0,
    )


__all__ = [
    "BlackScholesModel",
    "BlackScholesResult",
    "black_scholes_price_and_greeks",
    "d1",
    "d2",
    "normal_cdf",
    "normal_pdf",
]
