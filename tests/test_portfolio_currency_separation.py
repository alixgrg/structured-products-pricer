import pandas as pd
import pytest

from src.portfolio.inventory_loader import normalize_inventory_sheet
from src.portfolio.pricing_engine import PortfolioPricingConfig, PortfolioPricingEngine
from src.risk.aggregator import RiskAggregator


def test_inventory_loader_preserves_portfolio_and_currency() -> None:
    raw = pd.DataFrame(
        {
            "Portefeuille": ["Book_A"],
            "Date Valorisation": ["2026-04-28"],
            "Produit": ["Call"],
            "Quantite": [1.0],
            "Sous Jacent": ["AAPL"],
            "Devise": ["USD"],
            "Maturite": ["2027-04-28"],
            "Strike 1": [100.0],
        }
    )

    normalized = normalize_inventory_sheet("Options", raw)

    assert "portfolio" in normalized.columns
    assert "currency" in normalized.columns
    assert normalized.loc[0, "portfolio"] == "Book_A"
    assert normalized.loc[0, "currency"] == "USD"


def test_inventory_loader_adds_default_portfolio_when_missing() -> None:
    raw = pd.DataFrame(
        {
            "Date Valorisation": ["2026-04-28"],
            "Produit": ["Call"],
            "Quantite": [1.0],
            "Sous Jacent": ["AAPL"],
            "Maturite": ["2027-04-28"],
            "Strike 1": [100.0],
        }
    )

    normalized = normalize_inventory_sheet("Options", raw)

    assert normalized.loc[0, "portfolio"] == "default"
    assert normalized.loc[0, "currency"] == "USD"


def test_pricing_engine_exports_currency_and_risk_underlying_for_equity_option() -> None:
    inventory = pd.DataFrame(
        {
            "source_sheet": ["options"],
            "source_row": [1],
            "portfolio": ["EquityBook"],
            "product_type": ["Call"],
            "underlying": ["AAPL"],
            "currency": ["USD"],
            "quantity": [1.0],
            "time_to_maturity_years": [1.0],
            "strike_1": [100.0],
            "spot": [100.0],
            "rate": [0.03],
            "volatility": [0.20],
        }
    )

    engine = PortfolioPricingEngine()
    priced = engine.price_portfolio(inventory)

    assert priced.loc[0, "portfolio"] == "EquityBook"
    assert priced.loc[0, "currency"] == "USD"
    assert priced.loc[0, "risk_currency"] == "USD"
    assert priced.loc[0, "underlying"] == "AAPL"
    assert priced.loc[0, "risk_underlying"] == "AAPL"


def test_pricing_engine_replaces_missing_underlying_for_rate_products() -> None:
    inventory = pd.DataFrame(
        {
            "source_sheet": ["swaps"],
            "source_row": [1],
            "portfolio": ["RatesBook"],
            "product_type": ["Interest Rate Swap"],
            "currency": ["EUR"],
            "notional": [1_000_000.0],
            "fixed_rate": [0.03],
            "time_to_maturity_years": [2.0],
            "rate": [0.03],
        }
    )

    engine = PortfolioPricingEngine()
    priced = engine.price_portfolio(inventory)

    assert priced.loc[0, "currency"] == "EUR"
    assert priced.loc[0, "risk_currency"] == "EUR"
    assert priced.loc[0, "underlying"] == "EUR_RATE_CURVE"
    assert priced.loc[0, "risk_underlying"] == "EUR_RATE_CURVE"


def test_risk_aggregator_groups_by_portfolio_currency_product_and_risk_underlying() -> None:
    priced = pd.DataFrame(
        {
            "status": ["priced", "priced"],
            "portfolio": ["Book_USD", "Book_EUR"],
            "currency": ["USD", "EUR"],
            "risk_currency": ["USD", "EUR"],
            "product_class": ["VanillaOption", "InterestRateSwap"],
            "underlying": ["AAPL", "EUR_RATE_CURVE"],
            "risk_underlying": ["AAPL", "EUR_RATE_CURVE"],
            "maturity_bucket": ["6M-1Y", "1Y-2Y"],
            "strike_bucket": ["atm", "NA"],
            "price": [10.0, 20.0],
            "delta": [1.0, 0.0],
            "gamma": [0.1, 0.0],
            "vega": [2.0, 0.0],
            "theta": [-0.1, 0.0],
            "rho": [3.0, 4.0],
            "dv01": [0.0, -0.01],
        }
    )

    aggregated = RiskAggregator().aggregate_by_pillar(priced)

    assert set(aggregated["portfolio"]) == {"Book_USD", "Book_EUR"}
    assert set(aggregated["risk_currency"]) == {"USD", "EUR"}
    assert set(aggregated["risk_underlying"]) == {"AAPL", "EUR_RATE_CURVE"}


def test_risk_aggregator_refuses_single_total_across_mixed_currencies() -> None:
    priced = pd.DataFrame(
        {
            "status": ["priced", "priced"],
            "portfolio": ["Book", "Book"],
            "currency": ["USD", "EUR"],
            "risk_currency": ["USD", "EUR"],
            "product_class": ["VanillaOption", "InterestRateSwap"],
            "underlying": ["AAPL", "EUR_RATE_CURVE"],
            "risk_underlying": ["AAPL", "EUR_RATE_CURVE"],
            "maturity_bucket": ["6M-1Y", "1Y-2Y"],
            "strike_bucket": ["atm", "NA"],
            "price": [10.0, 20.0],
        }
    )

    with pytest.raises(ValueError, match="multiple currencies"):
        RiskAggregator().total(priced)

    safe = RiskAggregator().aggregate_safe_totals(priced)
    assert set(safe["risk_currency"]) == {"USD", "EUR"}