from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.calibration.implied_vol import (
    ImpliedVolSurface,
    calibrate_implied_vol_panel,
    clean_option_panel,
    implied_volatility_from_price,
)
from src.calibration.market_validation import MarketErrorThresholds, reprice_vanilla_market_quotes
from src.models.black_scholes import black_scholes_price_and_greeks


def test_implied_vol_inversion_recovers_known_sigma_call() -> None:
    sigma = 0.24
    price = black_scholes_price_and_greeks(
        option_type="call",
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.03,
        volatility=sigma,
        dividend_yield=0.0,
    ).price

    implied = implied_volatility_from_price(
        option_type="call",
        market_price=price,
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.03,
        dividend_yield=0.0,
    )

    assert implied == pytest.approx(sigma, rel=1e-8)


def test_implied_vol_inversion_is_robust_for_itm_and_otm_quotes() -> None:
    sigma = 0.31

    itm_put_price = black_scholes_price_and_greeks(
        option_type="put",
        spot=90.0,
        strike=110.0,
        maturity=0.75,
        rate=0.02,
        volatility=sigma,
        dividend_yield=0.0,
    ).price

    otm_call_price = black_scholes_price_and_greeks(
        option_type="call",
        spot=90.0,
        strike=120.0,
        maturity=0.75,
        rate=0.02,
        volatility=sigma,
        dividend_yield=0.0,
    ).price

    implied_itm_put = implied_volatility_from_price(
        option_type="put",
        market_price=itm_put_price,
        spot=90.0,
        strike=110.0,
        maturity=0.75,
        rate=0.02,
    )
    implied_otm_call = implied_volatility_from_price(
        option_type="call",
        market_price=otm_call_price,
        spot=90.0,
        strike=120.0,
        maturity=0.75,
        rate=0.02,
    )

    assert implied_itm_put == pytest.approx(sigma, rel=1e-7)
    assert implied_otm_call == pytest.approx(sigma, rel=1e-7)


def test_clean_option_panel_filters_aberrant_quotes() -> None:
    quotes = pd.DataFrame(
        {
            "option_type": ["call", "put", "call", "call"],
            "strike": [100.0, 100.0, 100.0, -10.0],
            "underlying_price": [100.0, 100.0, 100.0, 100.0],
            "time_to_maturity_years": [0.5, 0.5, 0.5, 0.5],
            "bid": [5.0, 4.8, 1.0, 2.0],
            "ask": [5.4, 5.2, 50.0, 2.2],
            "mid": [5.2, 5.0, 25.5, 2.1],
        }
    )

    clean = clean_option_panel(quotes, max_bid_ask_spread_ratio=1.5)

    assert len(clean) == 2
    assert clean["option_type"].isin(["call", "put"]).all()
    assert (clean["strike"] > 0.0).all()


def test_calibrate_panel_returns_finite_iv_and_small_repricing_error() -> None:
    spot = 100.0
    rate = 0.01
    maturity = 1.0

    strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    true_ivs = 0.18 + 0.20 * np.square(np.log(strikes / spot))

    prices = [
        black_scholes_price_and_greeks(
            option_type="call",
            spot=spot,
            strike=float(k),
            maturity=maturity,
            rate=rate,
            volatility=float(iv),
        ).price
        for k, iv in zip(strikes, true_ivs)
    ]

    quotes = pd.DataFrame(
        {
            "option_type": ["call"] * len(strikes),
            "strike": strikes,
            "underlying_price": [spot] * len(strikes),
            "time_to_maturity_years": [maturity] * len(strikes),
            "mid": prices,
            "bid": np.maximum(np.array(prices) - 0.05, 1e-4),
            "ask": np.array(prices) + 0.05,
        }
    )

    calibrated, result = calibrate_implied_vol_panel(
        quotes,
        rate=rate,
        drop_outliers=False,
    )

    assert len(calibrated) == len(strikes)
    assert np.isfinite(calibrated["implied_vol"]).all()
    assert result.objective_value is not None
    assert result.objective_value < 1e-8


def test_surface_interpolation_returns_finite_values() -> None:
    quotes = pd.DataFrame(
        {
            "time_to_maturity_years": [0.5, 0.5, 1.0, 1.0, 2.0, 2.0],
            "log_moneyness": [-0.1, 0.1, -0.1, 0.1, -0.1, 0.1],
            "implied_vol": [0.24, 0.22, 0.21, 0.20, 0.19, 0.18],
        }
    )

    surface = ImpliedVolSurface.from_quotes(quotes)
    value = surface.evaluate(1.25, 0.0)

    assert np.isfinite(value)
    assert 0.15 < value < 0.30


def test_market_repricing_reports_explicit_thresholds() -> None:
    class FlatVolSurface:
        def __init__(self, volatility: float) -> None:
            self.volatility_value = volatility

        def volatility(self, maturity: float | np.ndarray, log_moneyness: float | np.ndarray) -> float | np.ndarray:
            t_arr, k_arr = np.broadcast_arrays(np.asarray(maturity, dtype=float), np.asarray(log_moneyness, dtype=float))
            values = np.full_like(t_arr, self.volatility_value, dtype=float)
            if np.isscalar(maturity) and np.isscalar(log_moneyness):
                return float(values)
            return values

    spot = 100.0
    rate = 0.01
    vol = 0.2

    quotes = pd.DataFrame(
        [
            {
                "option_type": "call",
                "strike": 95.0,
                "underlying_price": spot,
                "time_to_maturity_years": 0.5,
                "log_moneyness": float(np.log(95.0 / spot)),
                "market_price": black_scholes_price_and_greeks(
                    option_type="call",
                    spot=spot,
                    strike=95.0,
                    maturity=0.5,
                    rate=rate,
                    volatility=vol,
                ).price,
                "implied_vol": vol,
            },
            {
                "option_type": "put",
                "strike": 105.0,
                "underlying_price": spot,
                "time_to_maturity_years": 1.0,
                "log_moneyness": float(np.log(105.0 / spot)),
                "market_price": black_scholes_price_and_greeks(
                    option_type="put",
                    spot=spot,
                    strike=105.0,
                    maturity=1.0,
                    rate=rate,
                    volatility=vol,
                ).price,
                "implied_vol": vol,
            },
        ]
    )

    result = reprice_vanilla_market_quotes(
        quotes,
        FlatVolSurface(vol),
        rate=rate,
        thresholds=MarketErrorThresholds(
            abs_price_mae=1e-8,
            abs_price_rmse=1e-8,
            abs_price_max=1e-8,
            abs_relative_price_mae=1e-8,
            abs_vol_mae=1e-8,
        ),
    )

    assert bool(result.summary.iloc[0]["within_thresholds"])
    assert result.calibration_result.parameters["within_thresholds"] == 1.0
