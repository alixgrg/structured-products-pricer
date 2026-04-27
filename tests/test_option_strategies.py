from __future__ import annotations

import pytest

from src.market.market_data import MarketData
from src.models.black_scholes import BlackScholesModel
from src.products.barrier_option import BarrierOption
from src.products.option_strategy import OptionStrategy


def test_call_spread_price_equals_sum_of_legs() -> None:
    strategy = OptionStrategy.call_spread(
        product_id="CS-1",
        maturity=1.0,
        strike_low=95.0,
        strike_high=105.0,
        notional=2.0,
    )
    model = BlackScholesModel()
    market = MarketData(spot=100.0, rate=0.03, volatility=0.20, dividend_yield=0.0)

    strategy_price = strategy.price(model, market)
    decomposition_price = sum(
        leg.quantity * model.price(leg.product, market) for leg in strategy.legs
    ) * strategy.notional

    assert strategy_price == pytest.approx(decomposition_price, rel=1e-12)


def test_put_spread_price_equals_sum_of_legs() -> None:
    strategy = OptionStrategy.put_spread(
        product_id="PS-1",
        maturity=1.0,
        strike_low=90.0,
        strike_high=110.0,
        notional=1.5,
    )
    model = BlackScholesModel()
    market = MarketData(spot=100.0, rate=0.02, volatility=0.25, dividend_yield=0.0)

    strategy_price = strategy.price(model, market)
    decomposition_price = sum(
        leg.quantity * model.price(leg.product, market) for leg in strategy.legs
    ) * strategy.notional

    assert strategy_price == pytest.approx(decomposition_price, rel=1e-12)


def test_butterfly_payoff_equals_sum_of_legs() -> None:
    strategy = OptionStrategy.butterfly(
        product_id="BF-1",
        maturity=1.0,
        strike_low=90.0,
        strike_mid=100.0,
        strike_high=110.0,
    )

    for spot in (80.0, 95.0, 100.0, 105.0, 120.0):
        payoff = strategy.payoff(spot)
        leg_sum = sum(leg.quantity * leg.product.payoff(spot) for leg in strategy.legs)
        assert payoff == pytest.approx(leg_sum, rel=1e-12)


def test_up_and_out_barrier_payoff_limit_case() -> None:
    option = BarrierOption(
        product_id="BO-1",
        option_type="call",
        strike=100.0,
        maturity=1.0,
        barrier=120.0,
        barrier_type="up-and-out",
    )

    assert option.payoff(119.0) == pytest.approx(19.0)
    assert option.payoff(120.0) == pytest.approx(0.0)


def test_down_and_in_barrier_payoff_limit_case() -> None:
    option = BarrierOption(
        product_id="BO-2",
        option_type="put",
        strike=100.0,
        maturity=1.0,
        barrier=85.0,
        barrier_type="down-and-in",
    )

    assert option.payoff(90.0) == pytest.approx(0.0)
    assert option.payoff(85.0) == pytest.approx(15.0)
