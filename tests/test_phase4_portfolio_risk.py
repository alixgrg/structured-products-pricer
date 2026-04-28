from pathlib import Path

import pandas as pd
import pytest

from src.portfolio.pricing_engine import PortfolioPricingConfig, PortfolioPricingEngine
from src.risk.aggregator import RiskAggregator
from src.risk.report import PortfolioRiskReport, RiskSnapshot


def _mini_inventory() -> dict[str, pd.DataFrame]:
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
                },
                {
                    "portfolio": "P1",
                    "source_sheet": "options",
                    "product_id": "CS-1",
                    "product_type": "Call Spread",
                    "underlying": "MSFT",
                    "time_to_maturity_years": 1.0,
                    "strike_1": 95.0,
                    "strike_2": 110.0,
                    "quantity": 1.0,
                },
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


def test_portfolio_pricing_engine_prices_supported_inventory() -> None:
    engine = PortfolioPricingEngine(
        PortfolioPricingConfig(
            default_spot=100.0,
            default_rate=0.03,
            default_volatility=0.20,
            spot_by_underlying={"MSFT": 100.0},
        )
    )

    priced = engine.price_portfolio(_mini_inventory())

    assert len(priced) == 3
    assert set(priced["status"]) == {"priced"}
    assert {"price", "delta", "gamma", "vega", "theta", "rho", "maturity_bucket", "strike_bucket"}.issubset(priced.columns)
    assert priced["price"].notna().all()
    assert priced.loc[priced["product_id"] == "CALL-1", "model_name"].iloc[0] == "BlackScholesModel"
    assert priced.loc[priced["product_id"] == "CS-1", "model_name"].iloc[0] == "StaticReplicationModel"
    assert priced.loc[priced["product_id"] == "IRS-1", "model_name"].iloc[0] == "DiscountingModel"


def test_risk_aggregator_sums_by_pillars() -> None:
    engine = PortfolioPricingEngine(PortfolioPricingConfig(default_spot=100.0, spot_by_underlying={"MSFT": 100.0}))
    priced = engine.price_portfolio(_mini_inventory())

    aggregated = RiskAggregator().aggregate_by_pillar(priced)

    assert not aggregated.empty
    assert {"underlying", "maturity_bucket", "strike_bucket", "price", "delta", "vega", "line_count"}.issubset(aggregated.columns)
    assert aggregated["price"].sum() == pytest.approx(priced["price"].sum())
    assert int(aggregated["line_count"].sum()) == len(priced)


def test_risk_report_exports_csv(tmp_path: Path) -> None:
    engine = PortfolioPricingEngine(PortfolioPricingConfig(default_spot=100.0, spot_by_underlying={"MSFT": 100.0}))
    priced = engine.price_portfolio(_mini_inventory())

    snapshot = RiskSnapshot.from_priced_portfolio(priced, valuation_date="2026-04-28", name="test_portfolio")
    report = PortfolioRiskReport([snapshot])
    summary = report.to_dataframe()
    exported = report.export_csv(tmp_path / "risk_report.csv")

    assert not summary.empty
    assert summary.loc[0, "line_count"] == len(priced)
    assert exported["summary"].exists()
    assert exported["portfolio_greeks"].exists()
    assert pd.read_csv(exported["summary"]).loc[0, "snapshot"] == "test_portfolio"
