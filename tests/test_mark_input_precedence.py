import math

import pytest

from src.factory.pricing_router import PricingRouter
from src.market.market_data import MarketData
from src.models.barrier_model import BarrierModel
from src.models.black_scholes import BlackScholesModel
from src.models.discounting_model import DiscountingModel
from src.models.monte_carlo import MonteCarloGBMModel
from src.products.barrier_option import BarrierOption
from src.products.vanilla_option import VanillaOption
from src.products.zero_coupon_bond import ZeroCouponBond


def test_black_scholes_uses_market_data_volatility_before_model_default() -> None:
    product = VanillaOption(
        product_id="CALL-ATM",
        option_type="call",
        strike=100.0,
        maturity=1.0,
    )

    model = BlackScholesModel(rate=0.03, volatility=0.20)

    low_vol_price = model.price(
        product,
        MarketData(spot=100.0, rate=0.03, volatility=0.20),
    )
    high_vol_price = model.price(
        product,
        MarketData(spot=100.0, rate=0.03, volatility=0.80),
    )

    assert high_vol_price > low_vol_price


def test_router_uses_market_data_volatility_before_router_default() -> None:
    product = VanillaOption(
        product_id="CALL-ATM",
        option_type="call",
        strike=100.0,
        maturity=1.0,
    )

    router = PricingRouter.with_defaults(rate=0.03, volatility=0.20)

    price_with_default_vol = router.price(
        product,
        MarketData(spot=100.0, rate=0.03, volatility=0.20),
    )
    price_with_line_vol = router.price(
        product,
        MarketData(spot=100.0, rate=0.03, volatility=0.80),
    )

    assert price_with_line_vol > price_with_default_vol


def test_discounting_uses_market_data_rate_before_model_default() -> None:
    product = ZeroCouponBond(
        product_id="ZCB-1Y",
        notional=100.0,
        maturity=1.0,
    )

    model = DiscountingModel(rate=0.03)

    price = model.price(
        product,
        MarketData(rate=0.10),
    )

    assert price == pytest.approx(100.0 * math.exp(-0.10), rel=1e-12)


def test_discounting_dv01_bumps_market_data_rate_when_present() -> None:
    product = ZeroCouponBond(
        product_id="ZCB-5Y",
        notional=100.0,
        maturity=5.0,
    )

    model = DiscountingModel(rate=0.03)
    market_data = MarketData(rate=0.10)

    dv01 = model.dv01(product, market_data)

    base_price = 100.0 * math.exp(-0.10 * 5.0)
    bumped_price = 100.0 * math.exp(-(0.10 + 1e-4) * 5.0)

    assert dv01 == pytest.approx(bumped_price - base_price, rel=1e-12)
    assert dv01 < 0.0


def test_barrier_model_uses_market_data_volatility_before_model_default() -> None:
    product = BarrierOption(
        product_id="CALL-DAO",
        option_type="call",
        strike=100.0,
        maturity=1.0,
        barrier=70.0,
        barrier_type="KO",
        barrier_direction="down",
        initial_spot=100.0,
    )

    model = BarrierModel(rate=0.03, volatility=0.20)

    low_vol_price = model.price(
        product,
        MarketData(spot=100.0, rate=0.03, volatility=0.20),
    )
    high_vol_price = model.price(
        product,
        MarketData(spot=100.0, rate=0.03, volatility=0.80),
    )

    assert high_vol_price != pytest.approx(low_vol_price)


def test_monte_carlo_uses_market_data_volatility_before_model_default() -> None:
    product = VanillaOption(
        product_id="CALL-MC",
        option_type="call",
        strike=100.0,
        maturity=1.0,
    )

    model = MonteCarloGBMModel(
        n_paths=20_000,
        n_steps=100,
        seed=123,
        rate=0.03,
        volatility=0.20,
    )

    low_vol_price = model.price(
        product,
        MarketData(spot=100.0, rate=0.03, volatility=0.20),
    )
    high_vol_price = model.price(
        product,
        MarketData(spot=100.0, rate=0.03, volatility=0.80),
    )

    assert high_vol_price > low_vol_price