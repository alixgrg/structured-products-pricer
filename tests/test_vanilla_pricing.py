from __future__ import annotations

import numpy as np
import pytest

from src.market.market_data import MarketData
from src.models.black_scholes import BlackScholesModel
from src.models.discounting_model import DiscountingModel
from src.products.vanilla_option import VanillaOption
from src.products.zero_coupon_bond import ZeroCouponBond


def test_zero_coupon_bond_price_with_constant_continuous_rate() -> None:
    product = ZeroCouponBond(
        product_id="ZC-001",
        notional=100.0,
        maturity=2.0,
    )
    model = DiscountingModel(rate=0.03)

    price = model.price(product)

    assert price == pytest.approx(100.0 * np.exp(-0.03 * 2.0))


def test_zero_coupon_bond_risk_metrics() -> None:
    maturity = 5.0
    product = ZeroCouponBond(
        product_id="ZC-001",
        notional=100.0,
        maturity=maturity,
    )
    model = DiscountingModel(rate=0.02)

    risk = model.risk(product)
    expected_bumped_duration = (1.0 - np.exp(-1e-4 * maturity)) / 1e-4

    assert risk["price"] > 0.0
    assert risk["duration"] == pytest.approx(expected_bumped_duration)
    assert risk["dv01"] < 0.0
    assert risk["rho"] > 0.0


def test_black_scholes_call_known_value() -> None:
    product = VanillaOption(
        product_id="CALL-ATM",
        option_type="call",
        strike=100.0,
        maturity=1.0,
    )
    market_data = MarketData(
        spot=100.0,
        rate=0.05,
        volatility=0.20,
        dividend_yield=0.0,
    )
    model = BlackScholesModel()

    result = model.price_and_greeks(product, market_data)

    assert result.price == pytest.approx(10.45058357, rel=1e-8)
    assert result.delta == pytest.approx(0.63683065, rel=1e-8)
    assert result.gamma == pytest.approx(0.01876202, rel=2e-7)
    assert result.vega == pytest.approx(37.52403469, rel=1e-8)
    assert result.rho == pytest.approx(53.23248155, rel=1e-8)


def test_black_scholes_put_known_value() -> None:
    product = VanillaOption(
        product_id="PUT-ATM",
        option_type="put",
        strike=100.0,
        maturity=1.0,
    )
    market_data = MarketData(
        spot=100.0,
        rate=0.05,
        volatility=0.20,
        dividend_yield=0.0,
    )
    model = BlackScholesModel()

    result = model.price_and_greeks(product, market_data)

    assert result.price == pytest.approx(5.57352602, rel=1e-8)
    assert result.delta == pytest.approx(-0.36316935, rel=1e-8)
    assert result.gamma == pytest.approx(0.01876202, rel=2e-7)
    assert result.vega == pytest.approx(37.52403469, rel=1e-8)
    assert result.rho == pytest.approx(-41.89046090, rel=1e-8)


def test_put_call_parity_without_dividend() -> None:
    call = VanillaOption(
        product_id="CALL",
        option_type="call",
        strike=100.0,
        maturity=1.0,
    )
    put = VanillaOption(
        product_id="PUT",
        option_type="put",
        strike=100.0,
        maturity=1.0,
    )
    market_data = MarketData(
        spot=100.0,
        rate=0.05,
        volatility=0.20,
        dividend_yield=0.0,
    )
    model = BlackScholesModel()

    call_price = model.price(call, market_data)
    put_price = model.price(put, market_data)

    lhs = call_price - put_price
    rhs = market_data.spot - call.strike * np.exp(-market_data.rate * call.maturity)

    assert lhs == pytest.approx(rhs, rel=1e-10)


def test_put_call_parity_with_continuous_dividend() -> None:
    call = VanillaOption(
        product_id="CALL",
        option_type="call",
        strike=100.0,
        maturity=1.0,
    )
    put = VanillaOption(
        product_id="PUT",
        option_type="put",
        strike=100.0,
        maturity=1.0,
    )
    market_data = MarketData(
        spot=100.0,
        rate=0.05,
        volatility=0.20,
        dividend_yield=0.02,
    )
    model = BlackScholesModel()

    call_price = model.price(call, market_data)
    put_price = model.price(put, market_data)

    lhs = call_price - put_price
    rhs = (
        market_data.spot * np.exp(-market_data.dividend_yield * call.maturity)
        - call.strike * np.exp(-market_data.rate * call.maturity)
    )

    assert lhs == pytest.approx(rhs, rel=1e-10)


def test_greeks_sanity_checks_for_call() -> None:
    product = VanillaOption(
        product_id="CALL",
        option_type="call",
        strike=100.0,
        maturity=1.0,
    )
    market_data = MarketData(
        spot=100.0,
        rate=0.03,
        volatility=0.25,
        dividend_yield=0.01,
    )
    model = BlackScholesModel()

    greeks = model.greeks(product, market_data)

    assert 0.0 < greeks["delta"] < 1.0
    assert greeks["gamma"] > 0.0
    assert greeks["vega"] > 0.0
    assert greeks["rho"] > 0.0


def test_greeks_sanity_checks_for_put() -> None:
    product = VanillaOption(
        product_id="PUT",
        option_type="put",
        strike=100.0,
        maturity=1.0,
    )
    market_data = MarketData(
        spot=100.0,
        rate=0.03,
        volatility=0.25,
        dividend_yield=0.01,
    )
    model = BlackScholesModel()

    greeks = model.greeks(product, market_data)

    assert -1.0 < greeks["delta"] < 0.0
    assert greeks["gamma"] > 0.0
    assert greeks["vega"] > 0.0
    assert greeks["rho"] < 0.0


def test_option_payoff_at_maturity() -> None:
    call = VanillaOption(
        product_id="CALL",
        option_type="call",
        strike=100.0,
        maturity=1.0,
        notional=2.0,
    )
    put = VanillaOption(
        product_id="PUT",
        option_type="put",
        strike=100.0,
        maturity=1.0,
        notional=2.0,
    )

    assert call.payoff(120.0) == pytest.approx(40.0)
    assert put.payoff(80.0) == pytest.approx(40.0)
