"""Discounting model for deterministic cash-flow products."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from src.market.market_data import MarketData
from src.models.base_model import PricingModel
from src.products.zero_coupon_bond import ZeroCouponBond
from src.rates.yield_curve import YieldCurve


CompoundingMethod = Literal["continuous", "annual"]


@dataclass(frozen=True, slots=True)
class DiscountingModel(PricingModel):
    """Simple discounting model.

    The model can use either:
    - a YieldCurve,
    - a constant rate passed to the model,
    - a constant rate from MarketData.
    """

    rate: float | None = None
    yield_curve: YieldCurve | None = None
    compounding: CompoundingMethod = "continuous"

    def price(self, product, market_data: MarketData | None = None) -> float:
        """Price a zero-coupon bond."""
        if not isinstance(product, ZeroCouponBond):
            raise TypeError("DiscountingModel currently supports ZeroCouponBond only.")

        return float(product.notional * self.discount_factor(product.maturity, market_data))

    def risk(self, product, market_data: MarketData | None = None) -> dict[str, float]:
        """Return basic rate-risk metrics for a zero-coupon bond."""
        price = self.price(product, market_data)
        maturity = float(product.maturity)

        # For a ZC under continuous compounding:
        # dP / dr = -T * P
        dv01 = maturity * price * 1e-4

        return {
            "price": price,
            "duration": maturity,
            "dv01": dv01,
        }

    def discount_factor(
        self,
        maturity: float,
        market_data: MarketData | None = None,
    ) -> float:
        """Return the discount factor for a given maturity."""
        if maturity < 0.0:
            raise ValueError("maturity must be non-negative.")

        if self.yield_curve is not None:
            return float(
                self.yield_curve.discount_factor(
                    maturity,
                    compounding=self.compounding,
                )
            )

        rate = self._resolve_rate(market_data)

        if self.compounding == "continuous":
            return float(np.exp(-rate * maturity))

        if self.compounding == "annual":
            return float(1.0 / (1.0 + rate) ** maturity)

        raise ValueError("compounding must be either 'continuous' or 'annual'.")

    def _resolve_rate(self, market_data: MarketData | None = None) -> float:
        """Resolve the pricing rate."""
        if self.rate is not None:
            return float(self.rate)

        if market_data is not None and market_data.rate is not None:
            return float(market_data.rate)

        raise ValueError("No rate available. Provide model.rate, yield_curve, or market_data.rate.")