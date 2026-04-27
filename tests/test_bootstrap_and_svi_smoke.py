from __future__ import annotations

import numpy as np
import pandas as pd

from src.calibration.svi import SVIVolSurface
from src.conventions.business_day import BusinessCalendar
from src.rates.bootstrap import bootstrap_yield_curve
from src.rates.market_instruments import BootstrapMarket, DepositQuote, FRAQuote, SwapQuote


def test_bootstrap_curve_smoke() -> None:
    market = BootstrapMarket(
        valuation_date=pd.Timestamp("2026-04-27"),
        calendar=BusinessCalendar(),
        deposits=(DepositQuote("1M", 0.03), DepositQuote("3M", 0.031), DepositQuote("6M", 0.032)),
        fras=(FRAQuote("6M", "9M", 0.033), FRAQuote("9M", "12M", 0.034)),
        swaps=(SwapQuote("2Y", 0.035), SwapQuote("3Y", 0.036), SwapQuote("5Y", 0.037)),
    )

    result = bootstrap_yield_curve(market)

    assert len(result.points) >= 6
    assert result.curve.discount_factor(1.0) > result.curve.discount_factor(5.0)
    assert result.curve.zero_rate(2.0) > 0.0


def test_svi_surface_smoke() -> None:
    rows = []
    for maturity in [0.25, 0.5, 1.0]:
        for k in np.linspace(-0.3, 0.3, 9):
            vol = 0.20 + 0.05 * k * k - 0.03 * k + 0.02 * maturity
            rows.append(
                {
                    "time_to_maturity_years": maturity,
                    "log_moneyness": k,
                    "implied_vol": vol,
                }
            )
    quotes = pd.DataFrame(rows)
    surface = SVIVolSurface.fit_from_quotes(quotes)

    assert surface.volatility(0.5, 0.0) > 0.0
    diagnostics = surface.diagnostics(log_moneyness_grid=np.linspace(-0.3, 0.3, 31))
    assert diagnostics["min_total_variance"] > 0.0
