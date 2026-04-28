"""Small Phase 2 validation script.

Run from the repository root:
    python -m scripts.validate_phase2_models
"""

from __future__ import annotations

from src.market.market_data import MarketData
from src.models.barrier_model import BarrierModel
from src.models.discounting_model import DiscountingModel
from src.models.monte_carlo import MonteCarloGBMModel
from src.models.static_replication import StaticReplicationModel
from src.products.barrier_option import BarrierOption
from src.products.coupon_bond import CouponBond
from src.products.option_strategies import CallSpread
from src.products.vanilla_option import VanillaOption


def main() -> None:
    market = MarketData(spot=100.0, rate=0.03, volatility=0.20)

    spread = CallSpread("CS-DEMO", maturity=1.0, strike_low=95.0, strike_high=105.0)
    static_model = StaticReplicationModel(rate=0.03, volatility=0.20)
    assert static_model.price(spread, market) > 0.0

    bond = CouponBond("CB-DEMO", 1_000.0, 2.0, 0.05, 2)
    discount_model = DiscountingModel(rate=0.03)
    assert discount_model.price(bond) > 1_000.0

    barrier_model = BarrierModel(rate=0.03, volatility=0.20)
    ko = BarrierOption("KO", "call", 100.0, 1.0, 80.0, "KO", barrier_direction="down")
    ki = BarrierOption("KI", "call", 100.0, 1.0, 80.0, "KI", barrier_direction="down")
    vanilla = VanillaOption("V", "call", 100.0, 1.0)
    vanilla_price = static_model.price(vanilla, market) if hasattr(vanilla, "get_legs") else None
    assert barrier_model.price(ko, market) + barrier_model.price(ki, market) > 0.0

    mc = MonteCarloGBMModel(n_paths=20_000, n_steps=80, seed=42)
    mc_result = mc.price_with_error(vanilla, market)
    assert mc_result.price > 0.0
    assert mc_result.confidence_interval_low < mc_result.price < mc_result.confidence_interval_high

    print("Phase 2 models validation: OK")


if __name__ == "__main__":
    main()
