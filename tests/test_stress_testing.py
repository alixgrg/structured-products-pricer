from __future__ import annotations

import pandas as pd

from src.portfolio.pricing_engine import PortfolioPricingConfig
from src.risk.stress_testing import StressScenario, StressTester


def _inventory() -> dict[str, pd.DataFrame]:
    return {
        "options": pd.DataFrame(
            [
                {
                    "portfolio": "P1",
                    "source_sheet": "options",
                    "product_id": "CALL-1",
                    "product_type": "Call",
                    "underlying": "MSFT",
                    "time_to_maturity_years": 1.0,
                    "strike_1": 100.0,
                    "quantity": 1.0,
                }
            ]
        ),
        "swaps": pd.DataFrame(
            [
                {
                    "portfolio": "P2",
                    "source_sheet": "swaps",
                    "product_id": "IRS-1",
                    "product_type": "Swap",
                    "currency": "EUR",
                    "time_to_maturity_years": 2.0,
                    "notional": 1_000_000.0,
                    "fixed_rate": 0.025,
                    "fixed_leg_frequency": "1Y",
                    "floating_rate_index_1": "EURIBOR6M",
                }
            ]
        ),
    }


def test_stress_tester_runs_and_returns_summary() -> None:
    config = PortfolioPricingConfig(
        default_spot=100.0,
        default_rate=0.03,
        default_volatility=0.20,
        spot_by_underlying={"MSFT": 100.0},
        volatility_by_underlying={"MSFT": 0.20},
    )
    tester = StressTester(base_config=config)

    result = tester.run(
        _inventory(),
        scenarios=[
            StressScenario("central"),
            StressScenario("spot_up_10pct", spot_shock=0.10),
            StressScenario("vol_up_5pts", vol_shock=0.05),
            StressScenario("rates_up_100bps", rate_shock=0.01),
        ],
    )

    assert not result.line_results.empty
    assert not result.scenario_summary.empty
    assert not result.pnl_by_position.empty
    assert {"scenario", "price", "pnl_vs_base"}.issubset(result.scenario_summary.columns)
    assert set(result.scenario_summary["scenario"]) == {
        "central",
        "spot_up_10pct",
        "vol_up_5pts",
        "rates_up_100bps",
    }
