from __future__ import annotations

import pandas as pd
import pytest

from src.portfolio.book import PortfolioMarketContext, PortfolioValuationEngine
from src.risk.report import build_portfolio_risk_summary


def _sample_inventory() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "portfolio": "BOOK-A",
                "source_sheet": "options",
                "product_id": "OPT-1",
                "product_type": "call",
                "underlying": "AAPL",
                "quantity": 10.0,
                "strike_1": 100.0,
                "time_to_maturity_years": 1.0,
            },
            {
                "portfolio": "BOOK-A",
                "source_sheet": "options",
                "product_id": "OPT-2",
                "product_type": "put spread",
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
                "product_type": "capital protected note",
                "underlying": "MSFT",
                "quantity": 100.0,
                "participation_rate": 0.8,
                "time_to_maturity_years": 1.5,
            },
        ]
    )


def test_sum_line_prices_equals_portfolio_total() -> None:
    inventory = _sample_inventory()
    engine = PortfolioValuationEngine()
    context = PortfolioMarketContext(
        default_spot=100.0,
        rate=0.03,
        volatility=0.20,
        spot_by_underlying={"AAPL": 105.0, "MSFT": 110.0},
    )

    result = engine.value_inventory(inventory, market=context)

    supported = result.line_valuations[result.line_valuations["status"] == "supported"]
    total_from_lines = supported["price"].sum()
    total_from_portfolio_agg = result.by_portfolio["price"].sum()

    assert total_from_lines == pytest.approx(total_from_portfolio_agg, rel=1e-12)


def test_aggregations_are_coherent_across_dimensions() -> None:
    inventory = _sample_inventory()
    engine = PortfolioValuationEngine()
    context = PortfolioMarketContext(
        default_spot=100.0,
        rate=0.03,
        volatility=0.20,
        spot_by_underlying={"AAPL": 105.0, "MSFT": 110.0},
    )

    result = engine.value_inventory(inventory, market=context)

    total = result.line_valuations.loc[
        result.line_valuations["status"] == "supported", "price"
    ].sum()

    assert result.by_product["price"].sum() == pytest.approx(total, rel=1e-12)
    assert result.by_underlying["price"].sum() == pytest.approx(total, rel=1e-12)
    assert result.by_maturity["price"].sum() == pytest.approx(total, rel=1e-12)


def test_unsupported_products_are_flagged_and_excluded() -> None:
    inventory = pd.DataFrame(
        [
            {
                "portfolio": "BOOK-X",
                "source_sheet": "swaps",
                "product_id": "SWAP-1",
                "product_type": "swap",
                "underlying": "EURIBOR",
                "quantity": 1_000_000.0,
                "time_to_maturity_years": 2.0,
            },
            {
                "portfolio": "BOOK-X",
                "source_sheet": "options",
                "product_id": "OPT-OK",
                "product_type": "call",
                "underlying": "AAPL",
                "quantity": 1.0,
                "strike_1": 100.0,
                "time_to_maturity_years": 1.0,
            },
        ]
    )

    engine = PortfolioValuationEngine()
    result = engine.value_inventory(inventory, market=PortfolioMarketContext())

    unsupported = result.line_valuations[result.line_valuations["status"] == "unsupported"]
    supported = result.line_valuations[result.line_valuations["status"] == "supported"]

    assert len(unsupported) == 1
    assert unsupported.iloc[0]["product_id"] == "SWAP-1"
    assert pd.isna(unsupported.iloc[0]["price"])

    assert len(supported) == 1
    assert result.by_portfolio["line_count"].sum() == 1


def test_risk_summary_matches_line_level_totals() -> None:
    inventory = _sample_inventory()
    engine = PortfolioValuationEngine()
    result = engine.value_inventory(
        inventory,
        market=PortfolioMarketContext(spot_by_underlying={"AAPL": 105.0, "MSFT": 110.0}),
    )

    summary = build_portfolio_risk_summary(result.line_valuations)

    supported = result.line_valuations[result.line_valuations["status"] == "supported"]

    assert summary.by_product["delta"].sum() == pytest.approx(supported["delta"].sum(), rel=1e-12)
    assert summary.by_underlying["vega"].sum() == pytest.approx(supported["vega"].sum(), rel=1e-12)
