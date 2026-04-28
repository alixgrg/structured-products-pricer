from __future__ import annotations

import pandas as pd
import pytest

from src.portfolio.pricing_engine import PortfolioPricingConfig, PortfolioPricingEngine
from src.risk.aggregator import RiskAggregator
from src.risk.report import build_portfolio_risk_summary


def _sample_inventory() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "portfolio": "BOOK-A",
                "source_sheet": "options",
                "product_id": "OPT-1",
                "product_type": "Call",
                "underlying": "AAPL",
                "quantity": 10.0,
                "strike_1": 100.0,
                "time_to_maturity_years": 1.0,
            },
            {
                "portfolio": "BOOK-A",
                "source_sheet": "options",
                "product_id": "OPT-2",
                "product_type": "Put Spread",
                "underlying": "AAPL",
                "quantity": 5.0,
                "strike_1": 95.0,
                "strike_2": 110.0,
                "time_to_maturity_years": 0.75,
            },
            {
                "portfolio": "BOOK-B",
                "source_sheet": "structured_notes",
                "product_id": "SN-1",
                "product_type": "Capital Protected Note",
                "underlying": "MSFT",
                "quantity": 100.0,
                "participation_rate": 0.8,
                "time_to_maturity_years": 1.5,
            },
        ]
    )


def _engine() -> PortfolioPricingEngine:
    return PortfolioPricingEngine(
        PortfolioPricingConfig(
            default_spot=100.0,
            default_rate=0.03,
            default_volatility=0.20,
            spot_by_underlying={"AAPL": 105.0, "MSFT": 110.0},
            volatility_by_underlying={"AAPL": 0.20, "MSFT": 0.25},
        )
    )


def test_sum_line_prices_equals_portfolio_total() -> None:
    priced = _engine().price_portfolio(_sample_inventory())
    by_portfolio = RiskAggregator().aggregate_by(priced, ["portfolio"])

    supported = priced[priced["status"] == "priced"]
    total_from_lines = supported["price"].sum()
    total_from_portfolio_agg = by_portfolio["price"].sum()

    assert total_from_lines == pytest.approx(total_from_portfolio_agg, rel=1e-12)


def test_aggregations_are_coherent_across_dimensions() -> None:
    priced = _engine().price_portfolio(_sample_inventory())
    aggregator = RiskAggregator()

    total = priced.loc[priced["status"] == "priced", "price"].sum()

    assert aggregator.aggregate_by(priced, ["product_type"])["price"].sum() == pytest.approx(total, rel=1e-12)
    assert aggregator.aggregate_by(priced, ["underlying"])["price"].sum() == pytest.approx(total, rel=1e-12)
    assert aggregator.aggregate_by(priced, ["maturity_bucket"])["price"].sum() == pytest.approx(total, rel=1e-12)


def test_pricing_errors_are_flagged_and_excluded_from_risk_aggregates() -> None:
    inventory = pd.DataFrame(
        [
            {
                "portfolio": "BOOK-X",
                "source_sheet": "unknown",
                "product_id": "BAD-1",
                "product_type": "Mystery Product",
                "underlying": "AAPL",
                "quantity": 1.0,
                "time_to_maturity_years": 1.0,
            },
            {
                "portfolio": "BOOK-X",
                "source_sheet": "options",
                "product_id": "OPT-OK",
                "product_type": "Call",
                "underlying": "AAPL",
                "quantity": 1.0,
                "strike_1": 100.0,
                "time_to_maturity_years": 1.0,
            },
        ]
    )

    priced = _engine().price_portfolio(inventory)
    errors = priced[priced["status"] == "error"]
    supported = priced[priced["status"] == "priced"]
    by_portfolio = RiskAggregator().aggregate_by(priced, ["portfolio"])

    assert len(errors) == 1
    assert errors.iloc[0]["product_id"] == "BAD-1"
    assert pd.isna(errors.iloc[0]["price"])
    assert "Cannot infer product builder" in errors.iloc[0]["error_message"]

    assert len(supported) == 1
    assert int(by_portfolio["line_count"].sum()) == 1


def test_risk_summary_matches_line_level_totals() -> None:
    priced = _engine().price_portfolio(_sample_inventory())
    summary = build_portfolio_risk_summary(priced)
    supported = priced[priced["status"] == "priced"]

    assert summary.by_product["delta"].sum() == pytest.approx(supported["delta"].sum(), rel=1e-12)
    assert summary.by_underlying["vega"].sum() == pytest.approx(supported["vega"].sum(), rel=1e-12)
