from __future__ import annotations

import numpy as np
import pytest

from src.market.market_data import MarketData
from src.models.barrier_model import BarrierModel
from src.models.black_scholes import black_scholes_price_and_greeks
from src.products.barrier_option import BarrierOption
from src.rates.yield_curve import YieldCurve


@pytest.fixture
def yield_curve() -> YieldCurve:
    return YieldCurve(
        maturities=np.array([0.25, 1.0, 2.0, 5.0]),
        zero_rates=np.array([0.020, 0.024, 0.028, 0.032]),
        interpolation="linear",
        name="test_curve",
    )


@pytest.mark.parametrize(
    "option_type,direction,barrier",
    [
        ("call", "down", 80.0),
        ("call", "up", 130.0),
        ("put", "down", 80.0),
        ("put", "up", 130.0),
    ],
)
def test_barrier_knock_out_plus_knock_in_equals_vanilla_with_yield_curve(
    option_type: str,
    direction: str,
    barrier: float,
    yield_curve: YieldCurve,
) -> None:
    maturity = 1.25
    spot = 100.0
    strike = 100.0
    volatility = 0.21
    dividend_yield = 0.01

    market = MarketData(
        spot=spot,
        volatility=volatility,
        dividend_yield=dividend_yield,
    )

    model = BarrierModel(
        yield_curve=yield_curve,
        volatility=volatility,
    )

    ko = BarrierOption(
        product_id="KO",
        option_type=option_type,
        strike=strike,
        maturity=maturity,
        barrier=barrier,
        barrier_type="KO",
        barrier_direction=direction,
        dividend_yield=dividend_yield,
    )

    ki = BarrierOption(
        product_id="KI",
        option_type=option_type,
        strike=strike,
        maturity=maturity,
        barrier=barrier,
        barrier_type="KI",
        barrier_direction=direction,
        dividend_yield=dividend_yield,
    )

    rate = float(yield_curve.zero_rate(maturity))

    vanilla = black_scholes_price_and_greeks(
        option_type=option_type,
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        volatility=volatility,
        dividend_yield=dividend_yield,
    ).price

    assert model.price(ko, market) + model.price(ki, market) == pytest.approx(
        vanilla,
        rel=1e-12,
        abs=1e-12,
    )


def test_barrier_model_uses_yield_curve_before_market_rate(yield_curve: YieldCurve) -> None:
    maturity = 1.25
    spot = 100.0
    strike = 100.0
    barrier = 80.0
    volatility = 0.21
    dividend_yield = 0.01

    # Taux volontairement très différent : il ne doit pas être utilisé
    # car le modèle reçoit yield_curve.
    market = MarketData(
        spot=spot,
        rate=0.99,
        volatility=volatility,
        dividend_yield=dividend_yield,
    )

    product = BarrierOption(
        product_id="CDO",
        option_type="call",
        strike=strike,
        maturity=maturity,
        barrier=barrier,
        barrier_type="KO",
        barrier_direction="down",
        dividend_yield=dividend_yield,
    )

    model_with_curve = BarrierModel(
        yield_curve=yield_curve,
        volatility=volatility,
    )

    model_with_equivalent_rate = BarrierModel(
        rate=float(yield_curve.zero_rate(maturity)),
        volatility=volatility,
    )

    assert model_with_curve.price(product, market) == pytest.approx(
        model_with_equivalent_rate.price(product, market),
        rel=1e-12,
        abs=1e-12,
    )