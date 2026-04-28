"""Discounting model for deterministic cash-flow products."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import exp
from typing import Literal

import numpy as np

from src.market.market_data import MarketData
from src.models.base_model import PricingModel
from src.models.pricing_inputs import resolve_pricing_rate
from src.products.basis_swap import BasisSwap
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
        if isinstance(product, BasisSwap):
            return self.price_basis_swap(product, market_data)

        raise TypeError(
            "DiscountingModel supports ZeroCouponBond, CouponBond, InterestRateSwap and BasisSwap only."
        )

    def risk(self, product, market_data: MarketData | None = None) -> dict[str, float]:
        price = self.price(product, market_data)
        dv01 = self.dv01(product, market_data)

        # Project/reporting convention:
        # - dv01 = price(+1bp) - price(base), usually negative for fixed-rate bonds.
        # - rho is reported as a positive rate exposure for fixed-income products:
        #   rho = -dV/dr ~= -dv01 / 1bp.
        #
        # This keeps backward compatibility with existing tests and reports.
        rho_reported = -dv01 * 1e4

        duration = abs(dv01) / (abs(price) * 1e-4) if abs(price) > 1e-14 else 0.0

        return {
            "price": price,
            "duration": duration,
            "dv01": dv01,
            "rho": rho_reported,
            "delta": 0.0,
            "gamma": 0.0,
            "vega": 0.0,
            "theta": 0.0,
        }

    def price_coupon_bond(self, product: CouponBond, market_data: MarketData | None = None) -> float:
        total = 0.0
        for payment_time, amount in product.get_cash_flows():
            total += float(amount) * self.discount_factor(float(payment_time), market_data)
        return float(total)

    def price_swap(self, product: InterestRateSwap, market_data: MarketData | None = None) -> float:
        return float(self.fixed_leg_pv(product, market_data) - self.float_leg_pv(product, market_data))

    def fixed_leg_pv(self, product: InterestRateSwap, market_data: MarketData | None = None) -> float:
        return float(
            sum(
                amount * self.discount_factor(float(payment_time), market_data)
                for payment_time, amount in product.fixed_leg_cash_flows()
            )
        )

    def float_leg_pv(self, product: InterestRateSwap, market_data: MarketData | None = None) -> float:
        return self._floating_leg_pv(product.notional, product.payment_times(), 0.0, market_data)

    def price_basis_swap(self, product: BasisSwap, market_data: MarketData | None = None) -> float:
        receive_pv = self._floating_leg_pv(
            product.notional,
            product.receive_payment_times(),
            product.spread,
            market_data,
        )
        pay_pv = self._floating_leg_pv(
            product.notional,
            product.pay_payment_times(),
            0.0,
            market_data,
        )
        return float(receive_pv - pay_pv)

    def _floating_leg_pv(
        self,
        notional: float,
        payment_times: list[float],
        spread: float,
        market_data: MarketData | None = None,
    ) -> float:
        previous = 0.0
        total = 0.0
        for payment_time in payment_times:
            t = float(payment_time)
            accrual = max(t - previous, 0.0)
            forward = self.forward_rate(previous, t, market_data)
            amount = float(notional) * (forward + float(spread)) * accrual
            total += amount * self.discount_factor(t, market_data)
            previous = t
        return float(total)

    def discount_factor(self, maturity: float, market_data: MarketData | None = None) -> float:
        if maturity < 0.0:
            raise ValueError("maturity must be non-negative.")
        if maturity == 0.0:
            return 1.0

        # Production priority:
        # 1. explicit line-level market_data.rate,
        # 2. calibrated yield curve,
        # 3. model fallback rate.
        if market_data is not None and market_data.rate is not None:
            rate = resolve_pricing_rate(
                maturity=maturity,
                yield_curve=None,
                model_rate=None,
                market_data=market_data,
            )
            return self._flat_discount_factor(rate, maturity)

        if self.yield_curve is not None:
            return float(self.yield_curve.discount_factor(maturity))

        rate = resolve_pricing_rate(
            maturity=maturity,
            yield_curve=None,
            model_rate=self.rate,
            market_data=None,
        )
        return self._flat_discount_factor(rate, maturity)

    def forward_rate(
        self,
        start: float,
        end: float,
        market_data: MarketData | None = None,
    ) -> float:
        if end <= start:
            raise ValueError("end must be greater than start.")

        df_start = self.discount_factor(start, market_data)
        df_end = self.discount_factor(end, market_data)
        accrual = end - start
        return float((df_start / df_end - 1.0) / accrual)
    
    def rho(self, product, market_data: MarketData | None = None, bump: float = 1e-4) -> float:
        """Central finite-difference rho: dV / dr."""
        up_model, up_market = self._shifted_pricing_inputs(market_data, bump)
        down_model, down_market = self._shifted_pricing_inputs(market_data, -bump)

        up_price = up_model.price(product, up_market)
        down_price = down_model.price(product, down_market)

        return float((up_price - down_price) / (2.0 * bump))

    def dv01(self, product, market_data: MarketData | None = None, bump: float = 1e-4) -> float:
        """Price change for a +1bp parallel rate move."""
        base_price = self.price(product, market_data)
        bumped_model, bumped_market = self._shifted_pricing_inputs(market_data, bump)
        bumped_price = bumped_model.price(product, bumped_market)
        return float(bumped_price - base_price)

    def parallel_shift(self, bump: float) -> "DiscountingModel":
        if self.yield_curve is not None:
            return replace(self, yield_curve=self.yield_curve.parallel_shift(bump))

        if self.rate is None:
            return replace(self, rate=float(bump))

        return replace(self, rate=float(self.rate) + float(bump))

    def _shifted_pricing_inputs(
        self,
        market_data: MarketData | None,
        bump: float,
    ) -> tuple["DiscountingModel", MarketData | None]:
        """Shift the effective rate source consistently."""
        if market_data is not None and market_data.rate is not None:
            return self, replace(market_data, rate=float(market_data.rate) + float(bump))

        return self.parallel_shift(bump), market_data

    def _flat_discount_factor(self, rate: float, maturity: float) -> float:
        if self.compounding == "continuous":
            return float(exp(-float(rate) * float(maturity)))
        if self.compounding == "annual":
            return float(1.0 / ((1.0 + float(rate)) ** float(maturity)))
        raise ValueError("Unsupported compounding method.")

__all__ = ["DiscountingModel"]
