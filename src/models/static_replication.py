"""Static-replication pricing model.

This model is intentionally central for the project:
- products exposing ``get_legs()`` are priced as signed vanilla-option portfolios;
- products exposing ``decomposition()`` are priced as signed portfolios of vanilla
  options and zero-coupon bonds;
- CouponBond and InterestRateSwap are delegated to the extended DiscountingModel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from src.market.market_data import MarketData
from src.models.base_model import PricingModel
from src.models.black_scholes import black_scholes_price_and_greeks
from src.models.discounting_model import DiscountingModel
from src.products.coupon_bond import CouponBond
from src.products.swap import InterestRateSwap
from src.products.vanilla_option import VanillaOption
from src.products.zero_coupon_bond import ZeroCouponBond
from src.rates.yield_curve import YieldCurve
from src.models.pricing_inputs import (
    resolve_dividend_yield,
    resolve_pricing_rate,
    resolve_pricing_volatility,
)


class VolatilitySurfaceLike(Protocol):
    def volatility(self, maturity: float | np.ndarray, log_moneyness: float | np.ndarray) -> float | np.ndarray:
        ...


@dataclass(frozen=True, slots=True)
class StaticReplicationModel(PricingModel):
    """Price static-replication products and deterministic rate products."""

    yield_curve: YieldCurve | None = None
    vol_surface: VolatilitySurfaceLike | None = None
    rate: float | None = None
    volatility: float | None = None
    dividend_yield: float = 0.0
    discount_model: DiscountingModel | None = None

    def price(self, product, market_data: MarketData | None = None) -> float:
        if isinstance(product, (ZeroCouponBond, CouponBond, InterestRateSwap)):
            return self._discount_model().price(product, market_data)

        if hasattr(product, "get_legs"):
            return self._price_option_legs(product.get_legs(), market_data, multiplier=float(getattr(product, "notional", 1.0)))

        if hasattr(product, "decomposition"):
            total = 0.0
            for leg in product.decomposition():
                total += float(leg.quantity) * self._price_leg_product(leg.product, market_data)
            return float(total)

        raise TypeError("StaticReplicationModel supports get_legs(), decomposition(), CouponBond and InterestRateSwap products.")

    def risk(self, product, market_data: MarketData | None = None) -> dict[str, float]:
        if isinstance(product, (ZeroCouponBond, CouponBond, InterestRateSwap)):
            risk = self._discount_model().risk(product, market_data)
            return {
                "price": float(risk.get("price", self.price(product, market_data))),
                "delta": 0.0,
                "gamma": 0.0,
                "vega": 0.0,
                "theta": 0.0,
                "rho": float(risk.get("rho", 0.0)),
                "dv01": float(risk.get("dv01", 0.0)),
            }

        totals = {"price": 0.0, "delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
        if hasattr(product, "get_legs"):
            multiplier = float(getattr(product, "notional", 1.0))
            for option, quantity in product.get_legs():
                greeks = self._price_option_result(option, market_data).to_dict()
                for key in totals:
                    totals[key] += multiplier * float(quantity) * float(greeks[key])
            return totals

        if hasattr(product, "decomposition"):
            for leg in product.decomposition():
                leg_product = leg.product
                quantity = float(leg.quantity)
                if isinstance(leg_product, VanillaOption):
                    greeks = self._price_option_result(leg_product, market_data).to_dict()
                    for key in totals:
                        totals[key] += quantity * float(greeks[key])
                elif isinstance(leg_product, ZeroCouponBond):
                    risk = self._discount_model().risk(leg_product, market_data)
                    totals["price"] += quantity * float(risk.get("price", 0.0))
                    totals["rho"] += quantity * float(risk.get("rho", 0.0))
                else:
                    raise TypeError(f"Unsupported replication leg: {type(leg_product)!r}")
            return totals

        raise TypeError(f"Unsupported product type for risk: {type(product)!r}")

    def _price_option_legs(self, legs: list[tuple[VanillaOption, float]], market_data: MarketData | None, *, multiplier: float) -> float:
        total = 0.0
        for option, quantity in legs:
            total += float(quantity) * self._price_leg_product(option, market_data)
        return float(multiplier * total)

    def _price_leg_product(self, leg_product, market_data: MarketData | None) -> float:
        if isinstance(leg_product, VanillaOption):
            return float(self._price_option_result(leg_product, market_data).price)
        if isinstance(leg_product, ZeroCouponBond):
            return float(self._discount_model().price(leg_product, market_data))
        if isinstance(leg_product, CouponBond):
            return float(self._discount_model().price(leg_product, market_data))
        raise TypeError(f"Unsupported replication leg: {type(leg_product)!r}")

    def _price_option_result(self, option: VanillaOption, market_data: MarketData | None):
        if market_data is None or market_data.spot is None:
            raise ValueError("market_data.spot is required to price option legs.")
        spot = float(market_data.spot)
        rate = self._rate(option.maturity, market_data)
        dividend_yield = self._dividend_yield(option, market_data)
        volatility = self._volatility(option, spot, market_data)
        return black_scholes_price_and_greeks(
            option_type=option.option_type,
            spot=spot,
            strike=option.strike,
            maturity=option.maturity,
            rate=rate,
            volatility=volatility,
            dividend_yield=dividend_yield,
            notional=option.notional,
        )

    def _discount_model(self) -> DiscountingModel:
        if self.discount_model is not None:
            return self.discount_model
        return DiscountingModel(rate=self.rate, yield_curve=self.yield_curve)

    def _rate(self, maturity: float, market_data: MarketData | None) -> float:
        return resolve_pricing_rate(
            maturity=maturity,
            yield_curve=self.yield_curve,
            model_rate=self.rate,
            market_data=market_data,
        )

    def _volatility(self, option: VanillaOption, spot: float, market_data: MarketData | None) -> float:
        log_moneyness = float(np.log(option.strike / spot))
        return resolve_pricing_volatility(
            maturity=option.maturity,
            log_moneyness=log_moneyness,
            volatility_surface=self.vol_surface,
            model_volatility=self.volatility,
            market_data=market_data,
        )

    def _dividend_yield(self, option: VanillaOption, market_data: MarketData | None) -> float:
        return resolve_dividend_yield(
            product_dividend_yield=option.dividend_yield,
            model_dividend_yield=self.dividend_yield,
            market_data=market_data,
        )


__all__ = ["StaticReplicationModel"]
