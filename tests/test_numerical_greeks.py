from __future__ import annotations

import pytest

from src.market.market_data import MarketData
from src.models.barrier_model import BarrierModel
from src.models.black_scholes import BlackScholesModel
from src.products.barrier_option import BarrierOption
from src.products.vanilla_option import VanillaOption
from src.risk.numerical_greeks import NumericalGreeksConfig, NumericalGreeksEngine


def test_numerical_greeks_match_black_scholes_for_vanilla() -> None:
    product = VanillaOption("CALL", "call", 100.0, 1.0)
    market = MarketData(spot=100.0, rate=0.03, volatility=0.20, dividend_yield=0.0)
    model = BlackScholesModel(rate=0.03, volatility=0.20)

    analytical = model.risk(product, market)
    numerical = NumericalGreeksEngine(
        NumericalGreeksConfig(
            spot_relative_bump=1e-4,
            volatility_bump=1e-4,
            rate_bump=1e-4,
            compute_theta=False,
        )
    ).greeks(product, model, market)

    assert numerical["delta"] == pytest.approx(analytical["delta"], rel=1e-4, abs=1e-4)
    assert numerical["gamma"] == pytest.approx(analytical["gamma"], rel=1e-3, abs=1e-4)
    assert numerical["vega"] == pytest.approx(analytical["vega"], rel=1e-4, abs=1e-3)
    assert numerical["rho"] == pytest.approx(analytical["rho"], rel=1e-4, abs=1e-3)


def test_numerical_greeks_fill_barrier_model_zero_greeks() -> None:
    product = BarrierOption(
        product_id="PUT-DO",
        option_type="put",
        strike=100.0,
        maturity=1.0,
        barrier=80.0,
        barrier_type="KO",
        barrier_direction="down",
    )
    market = MarketData(spot=100.0, rate=0.03, volatility=0.20, dividend_yield=0.0)
    model = BarrierModel(rate=0.03, volatility=0.20)

    analytical_placeholder = model.risk(product, market)
    enriched = NumericalGreeksEngine().enrich_metrics(
        product,
        model,
        market,
        analytical_placeholder,
    )

    assert enriched["price"] == pytest.approx(analytical_placeholder["price"])
    assert enriched["numerical_greeks_used"] == pytest.approx(1.0)
    assert abs(enriched["delta"]) > 1e-8 or abs(enriched["vega"]) > 1e-8
