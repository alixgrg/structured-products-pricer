"""Route products to their appropriate pricing model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.market.market_data import MarketData
from src.models.barrier_model import BarrierModel
from src.models.black_scholes import BlackScholesModel
from src.models.discounting_model import DiscountingModel
from src.models.monte_carlo import MonteCarloGBMModel
from src.models.static_replication import StaticReplicationModel
from src.products.autocall import AutocallProduct
from src.products.barrier_option import BarrierOption
from src.products.basis_swap import BasisSwap
from src.products.coupon_bond import CouponBond
from src.products.swap import InterestRateSwap
from src.products.vanilla_option import VanillaOption
from src.products.zero_coupon_bond import ZeroCouponBond
from src.rates.yield_curve import YieldCurve


@dataclass(slots=True)
class PricingRouter:
    """Map product instances to pricing models."""

    vanilla_model: BlackScholesModel
    barrier_model: BarrierModel
    monte_carlo_model: MonteCarloGBMModel
    static_replication_model: StaticReplicationModel
    discounting_model: DiscountingModel

    @classmethod
    def with_defaults(
        cls,
        *,
        rate: float | None = 0.03,
        volatility: float | None = 0.20,
        dividend_yield: float = 0.0,
        yield_curve: YieldCurve | None = None,
        vol_surface: Any | None = None,
        n_paths: int = 50_000,
        n_steps: int = 252,
        seed: int | None = 42,
    ) -> "PricingRouter":
        """Build a router with fallback assumptions.

        Important:
        - market_data.rate overrides yield_curve and model rate;
        - market_data.volatility overrides vol_surface and model volatility;
        - these defaults are fallback values only.
        """
        discounting_model = DiscountingModel(rate=rate, yield_curve=yield_curve)

        return cls(
            vanilla_model=BlackScholesModel(
                rate=rate,
                volatility=volatility,
                dividend_yield=dividend_yield,
                yield_curve=yield_curve,
            ),
            barrier_model=BarrierModel(
                rate=rate,
                volatility=volatility,
                dividend_yield=dividend_yield,
                yield_curve=yield_curve,
            ),
            monte_carlo_model=MonteCarloGBMModel(
                n_paths=n_paths,
                n_steps=n_steps,
                seed=seed,
                rate=rate,
                volatility=volatility,
                dividend_yield=dividend_yield,
                yield_curve=yield_curve,
            ),
            static_replication_model=StaticReplicationModel(
                yield_curve=yield_curve,
                vol_surface=vol_surface,
                rate=rate,
                volatility=volatility,
                dividend_yield=dividend_yield,
                discount_model=discounting_model,
            ),
            discounting_model=discounting_model,
        )

    def model_for(self, product: object):
        if isinstance(product, AutocallProduct) or bool(getattr(product, "requires_monte_carlo", False)):
            return self.monte_carlo_model
        if isinstance(product, BarrierOption):
            return self.barrier_model
        if isinstance(product, VanillaOption):
            return self.vanilla_model
        if isinstance(product, (ZeroCouponBond, CouponBond, InterestRateSwap, BasisSwap)):
            return self.discounting_model
        if hasattr(product, "get_legs") or hasattr(product, "decomposition"):
            return self.static_replication_model
        raise TypeError(f"No pricing model registered for product type {type(product)!r}.")

    def price(self, product: object, market_data: MarketData | None = None) -> float:
        return float(self.model_for(product).price(product, market_data))

    def risk(self, product: object, market_data: MarketData | None = None) -> dict[str, float]:
        model = self.model_for(product)
        if not hasattr(model, "risk"):
            return {"price": self.price(product, market_data)}
        return dict(model.risk(product, market_data))

    def price_and_risk(self, product: object, market_data: MarketData | None = None) -> dict[str, float]:
        metrics = self.risk(product, market_data)
        metrics.setdefault("price", self.price(product, market_data))
        return metrics


__all__ = ["PricingRouter"]
