from __future__ import annotations

import pandas as pd
import pytest

from src.factory.builders import (
    build_product_from_row,
    build_autocalls_from_frame,
    create_default_product_registry,
)
from src.factory.pricing_router import PricingRouter
from src.market.market_data import MarketData
from src.models.barrier_model import BarrierModel
from src.models.black_scholes import BlackScholesModel
from src.models.discounting_model import DiscountingModel
from src.models.monte_carlo import MonteCarloGBMModel
from src.models.static_replication import StaticReplicationModel
from src.products.autocall import AutocallProduct
from src.products.barrier_option import BarrierOption
from src.products.coupon_bond import CouponBond
from src.products.option_strategies import CallSpread
from src.products.structured_notes import CapitalProtectedNote
from src.products.swap import InterestRateSwap
from src.products.vanilla_option import VanillaOption
from src.products.zero_coupon_bond import ZeroCouponBond


def test_default_registry_builds_call_spread_from_product_type_value() -> None:
    registry = create_default_product_registry()
    row = pd.Series(
        {
            "product_id": "CS-1",
            "product_type": "Call Spread",
            "underlying": "AAPL",
            "time_to_maturity_years": 1.0,
            "strike_1": 95.0,
            "strike_2": 110.0,
            "quantity": 2.0,
        }
    )

    product = registry.build("Call Spread", row=row)

    assert isinstance(product, CallSpread)
    assert product.product_id == "CS-1"
    assert product.notional == pytest.approx(2.0)
    assert len(product.get_legs()) == 2


def test_build_product_from_row_infers_vanilla_and_barrier() -> None:
    vanilla = build_product_from_row(
        pd.Series(
            {
                "source_sheet": "options",
                "product_type": "Call",
                "underlying": "MSFT",
                "time_to_maturity_years": 0.5,
                "strike_1": 100.0,
                "quantity": 1.0,
            }
        )
    )
    barrier = build_product_from_row(
        pd.Series(
            {
                "source_sheet": "options",
                "product_type": "Put Down-and-Out",
                "underlying": "MSFT",
                "time_to_maturity_years": 1.0,
                "strike_1": 100.0,
                "barrier_level": 75.0,
                "quantity": 1.0,
            }
        )
    )

    assert isinstance(vanilla, VanillaOption)
    assert vanilla.option_type == "call"
    assert isinstance(barrier, BarrierOption)
    assert barrier.option_type == "put"
    assert barrier.barrier_type == "KO"
    assert barrier.barrier_direction == "down"


def test_build_product_from_row_builds_rates_and_structured_note() -> None:
    swap = build_product_from_row(
        pd.Series(
            {
                "source_sheet": "swaps",
                "product_id": "IRS-1",
                "notional": 1_000_000.0,
                "time_to_maturity_years": 3.0,
                "fixed_rate": 0.03,
                "fixed_leg_frequency": "6M",
                "floating_rate_index_1": "EURIBOR6M",
            }
        )
    )
    note = build_product_from_row(
        pd.Series(
            {
                "source_sheet": "structured_notes",
                "product_id": "SN-1",
                "product_type": "Capital Protected Note",
                "quantity": 100.0,
                "time_to_maturity_years": 2.0,
                "underlying": "AAPL",
                "participation_rate": 1.0,
                "spot_reference": 100.0,
            }
        )
    )

    assert isinstance(swap, InterestRateSwap)
    assert isinstance(note, CapitalProtectedNote)
    assert len(note.decomposition()) == 2


def test_build_autocalls_from_frame_groups_observation_rows() -> None:
    frame = pd.DataFrame(
        {
            "source_sheet": ["autocalls", "autocalls"],
            "product_id": ["AUTO-1", "AUTO-1"],
            "underlying": ["SX5E", "SX5E"],
            "observation_date": [0.5, 1.0],
            "autocall_trigger_level": [100.0, 95.0],
            "coupon_rate": [0.08, 0.08],
            "barrier_protection": [70.0, 70.0],
            "quantity": [100.0, 100.0],
            "initial_spot": [100.0, 100.0],
        }
    )

    products = build_autocalls_from_frame(frame)

    assert len(products) == 1
    assert isinstance(products[0], AutocallProduct)
    assert len(products[0].observation_dates) == 2
    assert products[0].trigger_levels == pytest.approx([1.0, 0.95])


def test_pricing_router_selects_expected_models() -> None:
    router = PricingRouter.with_defaults(rate=0.03, volatility=0.20, n_paths=2_000, n_steps=50)

    assert isinstance(router.model_for(VanillaOption("C", "call", 100.0, 1.0)), BlackScholesModel)
    assert isinstance(router.model_for(BarrierOption("B", "call", 100.0, 1.0, 80.0, "KO", barrier_direction="down")), BarrierModel)
    assert isinstance(router.model_for(ZeroCouponBond("Z", 100.0, 1.0)), DiscountingModel)
    assert isinstance(router.model_for(CouponBond("CB", 100.0, 2.0, 0.04)), DiscountingModel)
    assert isinstance(router.model_for(InterestRateSwap("S", 1_000_000.0, 2.0, 0.03, "EURIBOR6M")), DiscountingModel)
    assert isinstance(router.model_for(CallSpread("CS", 1.0, 95.0, 105.0)), StaticReplicationModel)
    assert isinstance(router.model_for(AutocallProduct("A", "SX5E", [0.5, 1.0], [1.0, 1.0], 0.08, 0.7)), MonteCarloGBMModel)


def test_pricing_router_prices_factory_products() -> None:
    router = PricingRouter.with_defaults(rate=0.03, volatility=0.20, n_paths=5_000, n_steps=50, seed=123)
    market = MarketData(spot=100.0, rate=0.03, volatility=0.20, dividend_yield=0.0)

    vanilla = build_product_from_row(
        pd.Series(
            {
                "product_type": "Call",
                "time_to_maturity_years": 1.0,
                "strike_1": 100.0,
                "quantity": 1.0,
            }
        )
    )
    spread = build_product_from_row(
        pd.Series(
            {
                "product_type": "Call Spread",
                "time_to_maturity_years": 1.0,
                "strike_1": 95.0,
                "strike_2": 110.0,
                "quantity": 1.0,
            }
        )
    )

    assert router.price(vanilla, market) > 0.0
    assert router.price(spread, market) > 0.0
    assert "delta" in router.risk(vanilla, market)
    assert "vega" in router.risk(spread, market)
