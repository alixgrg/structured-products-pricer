from __future__ import annotations

import math
import numpy as np

import pytest

from src.market.market_data import MarketData
from src.models.barrier_model import BarrierModel
from src.models.black_scholes import black_scholes_price_and_greeks
from src.models.discounting_model import DiscountingModel
from src.models.monte_carlo import MonteCarloGBMModel
from src.models.static_replication import StaticReplicationModel
from src.products.autocall import AutocallProduct
from src.products.barrier_option import BarrierOption
from src.products.coupon_bond import CouponBond
from src.products.option_strategies import CallSpread
from src.products.swap import InterestRateSwap
from src.products.vanilla_option import VanillaOption
from src.rates.yield_curve import YieldCurve


def test_static_replication_prices_call_spread_like_two_vanillas() -> None:
    market = MarketData(spot=100.0, rate=0.03, volatility=0.20, dividend_yield=0.0)
    model = StaticReplicationModel(rate=0.03, volatility=0.20)
    spread = CallSpread("CS", maturity=1.0, strike_low=95.0, strike_high=105.0)

    expected = (
        black_scholes_price_and_greeks(option_type="call", spot=100.0, strike=95.0, maturity=1.0, rate=0.03, volatility=0.20).price
        - black_scholes_price_and_greeks(option_type="call", spot=100.0, strike=105.0, maturity=1.0, rate=0.03, volatility=0.20).price
    )

    assert model.price(spread, market) == pytest.approx(expected, rel=1e-12)
    assert model.risk(spread, market)["vega"] != 0.0


def test_discounting_model_prices_coupon_bond() -> None:
    bond = CouponBond("CB", notional=1_000.0, maturity=2.0, coupon_rate=0.05, frequency=2)
    model = DiscountingModel(rate=0.03)

    expected = sum(amount * math.exp(-0.03 * float(t)) for t, amount in bond.get_cash_flows())

    assert model.price(bond) == pytest.approx(expected, rel=1e-12)
    assert model.risk(bond)["dv01"] > 0.0


def test_discounting_model_prices_swap_fixed_minus_float() -> None:
    curve = YieldCurve(
        maturities=np.array([0.25, 0.50, 1.0, 2.0, 5.0, 10.0]),
        zero_rates=np.array([0.03, 0.03, 0.03, 0.03, 0.03, 0.03]),
        interpolation="linear",
        name="flat_3pct",
    )
    swap = InterestRateSwap("IRS", notional=1_000_000.0, maturity=2.0, fixed_rate=0.035, float_index="EURIBOR6M", frequency="6M")
    model = DiscountingModel(yield_curve=curve)

    price = model.price(swap)

    assert price > 0.0
    assert price == pytest.approx(model.fixed_leg_pv(swap) - model.float_leg_pv(swap), rel=1e-12)


def test_barrier_in_out_parity_call_down_and_with_dividend() -> None:
    market = MarketData(spot=100.0, rate=0.03, volatility=0.22, dividend_yield=0.01)
    model = BarrierModel(rate=0.03, volatility=0.22)
    ko = BarrierOption("DO", "call", strike=100.0, maturity=1.5, barrier=80.0, barrier_type="KO", barrier_direction="down")
    ki = BarrierOption("DI", "call", strike=100.0, maturity=1.5, barrier=80.0, barrier_type="KI", barrier_direction="down")

    vanilla = black_scholes_price_and_greeks(
        option_type="call",
        spot=100.0,
        strike=100.0,
        maturity=1.5,
        rate=0.03,
        volatility=0.22,
        dividend_yield=0.01,
    ).price

    assert model.price(ko, market) + model.price(ki, market) == pytest.approx(vanilla, rel=1e-12, abs=1e-12)
    assert 0.0 <= model.price(ko, market) <= vanilla


def test_barrier_in_out_parity_put_up() -> None:
    market = MarketData(spot=100.0, rate=0.02, volatility=0.25)
    model = BarrierModel(rate=0.02, volatility=0.25)
    ko = BarrierOption("UO", "put", strike=100.0, maturity=1.0, barrier=125.0, barrier_type="KO", barrier_direction="up")
    ki = BarrierOption("UI", "put", strike=100.0, maturity=1.0, barrier=125.0, barrier_type="KI", barrier_direction="up")

    vanilla = black_scholes_price_and_greeks(
        option_type="put",
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.02,
        volatility=0.25,
    ).price

    assert model.price(ko, market) + model.price(ki, market) == pytest.approx(vanilla, rel=1e-12, abs=1e-12)


def test_monte_carlo_vanilla_converges_to_black_scholes() -> None:
    market = MarketData(spot=100.0, rate=0.03, volatility=0.20)
    product = VanillaOption("CALL", "call", strike=100.0, maturity=1.0)
    model = MonteCarloGBMModel(n_paths=10_000, n_steps=80, seed=123, antithetic=True)

    result = model.price_with_error(product, market)
    bs = black_scholes_price_and_greeks(option_type="call", spot=100.0, strike=100.0, maturity=1.0, rate=0.03, volatility=0.20).price

    assert abs(result.price - bs) < 4.0 * result.standard_error + 0.20
    assert result.confidence_interval_low < result.price < result.confidence_interval_high


def test_monte_carlo_prices_barrier_and_autocall() -> None:
    market = MarketData(spot=100.0, rate=0.03, volatility=0.20)
    model = MonteCarloGBMModel(n_paths=10_000, n_steps=60, seed=321, antithetic=True)

    barrier = BarrierOption("DOC", "call", strike=100.0, maturity=1.0, barrier=80.0, barrier_type="KO", barrier_direction="down")
    barrier_result = model.price_with_error(barrier, market)
    assert barrier_result.price >= 0.0
    assert barrier_result.standard_error >= 0.0

    autocall = AutocallProduct(
        product_id="AC",
        underlying="SX5E",
        observation_dates=[1.0, 2.0, 3.0],
        trigger_levels=[1.0, 1.0, 1.0],
        coupon_rate=0.05,
        barrier_protection=0.70,
        notional=100.0,
        initial_spot=100.0,
    )
    autocall_market = MarketData(spot=100.0, rate=0.03, volatility=0.20)
    autocall_result = model.price_with_error(autocall, autocall_market)
    assert autocall_result.price > 0.0
