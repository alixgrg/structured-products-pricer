"""Discounting model for deterministic cash-flow products.

Phase 2 extension:
- ZeroCouponBond
- CouponBond
- InterestRateSwap

The pricing convention for swaps is aligned with the product definition:
positive value = PV(fixed leg) - PV(floating leg).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from src.market.market_data import MarketData
from src.models.base_model import PricingModel
from src.models.pricing_inputs import resolve_pricing_rate
from src.products.coupon_bond import CouponBond
from src.products.swap import InterestRateSwap
from src.products.zero_coupon_bond import ZeroCouponBond
from src.rates.yield_curve import YieldCurve

CompoundingMethod = Literal["continuous", "annual"]


@dataclass(frozen=True, slots=True)
class DiscountingModel(PricingModel):
    """Discount deterministic cash flows with a curve or a constant rate."""

    rate: float | None = None
    yield_curve: YieldCurve | None = None
    compounding: CompoundingMethod = "continuous"

    def price(self, product, market_data: MarketData | None = None) -> float:
        if isinstance(product, ZeroCouponBond):
            return float(product.notional * self.discount_factor(product.maturity, market_data))

        if isinstance(product, CouponBond):
            return self.price_coupon_bond(product, market_data)

        if isinstance(product, InterestRateSwap):
            return self.price_swap(product, market_data)

        raise TypeError(
            "DiscountingModel supports ZeroCouponBond, CouponBond and InterestRateSwap only."
        )

    def risk(self, product, market_data: MarketData | None = None) -> dict[str, float]:
        price = self.price(product, market_data)
        dv01 = self.dv01(product, market_data)
        #maturity = float(getattr(product, "maturity", 0.0))
        duration = abs(dv01) / (abs(price) * 1e-4) if abs(price) > 1e-14 else 0.0
        return {
            "price": price,
            "duration": duration,
            "dv01": dv01,
            "rho": -dv01 * 1e4,
        }

    def price_coupon_bond(self, product: CouponBond, market_data: MarketData | None = None) -> float:
        total = 0.0
        for payment_time, amount in product.get_cash_flows():
            maturity = float(payment_time)
            total += float(amount) * self.discount_factor(maturity, market_data)
        return float(total)

    def price_swap(self, product: InterestRateSwap, market_data: MarketData | None = None) -> float:
        fixed_pv = self.fixed_leg_pv(product, market_data)
        floating_pv = self.float_leg_pv(product, market_data)
        return float(fixed_pv - floating_pv)

    def fixed_leg_pv(self, product: InterestRateSwap, market_data: MarketData | None = None) -> float:
        return float(
            sum(amount * self.discount_factor(float(payment_time), market_data) for payment_time, amount in product.fixed_leg_cash_flows())
        )

    def float_leg_pv(self, product: InterestRateSwap, market_data: MarketData | None = None) -> float:
        payment_times = product.payment_times()
        previous = 0.0
        total = 0.0
        for payment_time in payment_times:
            t = float(payment_time)
            accrual = max(t - previous, 0.0)
            forward = self.forward_rate(previous, t, market_data)
            amount = product.notional * forward * accrual
            total += amount * self.discount_factor(t, market_data)
            previous = t
        return float(total)

    def discount_factor(self, maturity: float, market_data: MarketData | None = None) -> float:
        if maturity < 0.0:
            raise ValueError("maturity must be non-negative.")

        if self.yield_curve is not None:
            return float(self.yield_curve.discount_factor(maturity, compounding=self.compounding))

        rate = self._resolve_rate(market_data)
        if self.compounding == "continuous":
            return float(np.exp(-rate * maturity))
        if self.compounding == "annual":
            return float(1.0 / (1.0 + rate) ** maturity)
        raise ValueError("compounding must be either 'continuous' or 'annual'.")

    def forward_rate(self, start: float, end: float, market_data: MarketData | None = None) -> float:
        if end <= start:
            raise ValueError("end must be greater than start.")
        if self.yield_curve is not None and hasattr(self.yield_curve, "forward_rate"):
            return float(self.yield_curve.forward_rate(start, end))

        df_start = self.discount_factor(start, market_data)
        df_end = self.discount_factor(end, market_data)
        return float((df_start / df_end - 1.0) / (end - start))

    def dv01(self, product, market_data: MarketData | None = None, bump: float = 1e-4) -> float:
        """Return price impact for a +1bp parallel rate move: P(r) - P(r+bump)."""
        if self.yield_curve is not None:
            # Keep this model simple and robust for the project: for curve-based
            # pricing, approximate first-order sensitivity from cash-flow times.
            return self._curve_dv01_proxy(product, market_data, bump)

        base_rate = self._resolve_rate(market_data)
        up_model = replace(self, rate=base_rate + bump)
        return float(self.price(product, market_data) - up_model.price(product, market_data))

    def _curve_dv01_proxy(self, product, market_data: MarketData | None, bump: float) -> float:
        if isinstance(product, ZeroCouponBond):
            price = self.price(product, market_data)
            return float(product.maturity * price * bump)

        if isinstance(product, CouponBond):
            total = 0.0
            for payment_time, amount in product.get_cash_flows():
                t = float(payment_time)
                pv = float(amount) * self.discount_factor(t, market_data)
                total += t * pv * bump
            return float(total)

        if isinstance(product, InterestRateSwap):
            # Approximate by fixed leg duration only. A full curve bump engine can
            # replace this in Phase 4/bonus numerical Greeks.
            total = 0.0
            for payment_time, amount in product.fixed_leg_cash_flows():
                t = float(payment_time)
                pv = float(amount) * self.discount_factor(t, market_data)
                total += t * pv * bump
            return float(total)

        raise TypeError(f"Unsupported product type for DV01: {type(product)!r}")

    def _resolve_rate(self, market_data: MarketData | None = None) -> float:
        return resolve_pricing_rate(model_rate=self.rate, market_data=market_data)


__all__ = ["DiscountingModel"]
