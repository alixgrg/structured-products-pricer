from __future__ import annotations

import pandas as pd

from src.portfolio import (
    DEFAULT_PORTFOLIO_TEMPLATES,
    PortfolioPricingConfig,
    PortfolioPricingEngine,
    build_pricing_inventory,
    create_demo_mixed_portfolios,
)


def _neutral_inventory() -> pd.DataFrame:
    valuation_date = pd.Timestamp("2026-04-28")
    return pd.DataFrame(
        [
            {
                "source_sheet": "options",
                "source_row": 1,
                "product_id": "CALL-ATM",
                "product_type": "Call",
                "underlying": "MSFT",
                "time_to_maturity_years": 1.0,
                "quantity": 10.0,
                "strike_1": 100.0,
                "spot": 100.0,
                "rate": 0.03,
                "volatility": 0.20,
            },
            {
                "source_sheet": "options",
                "source_row": 2,
                "product_id": "CALL-SPREAD",
                "product_type": "Call Spread",
                "underlying": "MSFT",
                "time_to_maturity_years": 1.0,
                "quantity": 5.0,
                "strike_1": 95.0,
                "strike_2": 110.0,
                "spot": 100.0,
                "rate": 0.03,
                "volatility": 0.20,
            },
            {
                "source_sheet": "options",
                "source_row": 3,
                "product_id": "DOWN-OUT",
                "product_type": "Put Down-and-Out",
                "barrier_type": "KO",
                "barrier_level": 80.0,
                "underlying": "MSFT",
                "time_to_maturity_years": 1.0,
                "quantity": 2.0,
                "strike_1": 100.0,
                "spot": 100.0,
                "rate": 0.03,
                "volatility": 0.20,
            },
            {
                "source_sheet": "bonds",
                "source_row": 1,
                "product_id": "ZCB-2Y",
                "product_type": "Zero Coupon Bond",
                "currency": "EUR",
                "time_to_maturity_years": 2.0,
                "notional": 100_000.0,
                "rate": 0.03,
            },
            {
                "source_sheet": "structured_notes",
                "source_row": 1,
                "product_id": "CPN-1",
                "product_type": "Capital Protected Note",
                "underlying": "MSFT",
                "time_to_maturity_years": 3.0,
                "quantity": 100_000.0,
                "participation_rate": 0.80,
                "spot_reference": 100.0,
                "spot": 100.0,
                "rate": 0.03,
                "volatility": 0.20,
            },
            {
                "source_sheet": "structured_notes",
                "source_row": 2,
                "product_id": "RC-1",
                "product_type": "Reverse Convertible",
                "underlying": "MSFT",
                "time_to_maturity_years": 1.0,
                "quantity": 100_000.0,
                "coupon_rate": 0.08,
                "barrier_1": 0.70,
                "spot_reference": 100.0,
                "spot": 100.0,
                "rate": 0.03,
                "volatility": 0.20,
            },
            {
                "source_sheet": "autocalls",
                "source_row": 1,
                "product_id": "AUTO-1",
                "product_type": "Autocall",
                "underlying": "MSFT",
                "valuation_date": valuation_date,
                "observation_date": valuation_date + pd.DateOffset(years=1),
                "autocall_trigger_level": 1.0,
                "coupon_rate": 0.07,
                "notional": 100_000.0,
                "spot": 100.0,
                "rate": 0.03,
                "volatility": 0.20,
            },
            {
                "source_sheet": "autocalls",
                "source_row": 2,
                "product_id": "AUTO-1",
                "product_type": "Autocall",
                "underlying": "MSFT",
                "valuation_date": valuation_date,
                "observation_date": valuation_date + pd.DateOffset(years=2),
                "autocall_trigger_level": 0.95,
                "coupon_rate": 0.14,
                "notional": 100_000.0,
                "spot": 100.0,
                "rate": 0.03,
                "volatility": 0.20,
            },
        ]
    )


def test_demo_mixed_portfolios_create_four_priceable_portfolios() -> None:
    demo_inventory = create_demo_mixed_portfolios(_neutral_inventory())
    pricing_inventory = build_pricing_inventory(demo_inventory)

    assert set(demo_inventory["portfolio"]) == {
        template.name for template in DEFAULT_PORTFOLIO_TEMPLATES
    }
    assert demo_inventory.groupby("portfolio")["product_family"].nunique().min() >= 2
    assert "source_product_id" in demo_inventory.columns
    assert "portfolio_strategy" in demo_inventory.columns
    assert pricing_inventory["portfolio"].nunique() == 4
    assert pricing_inventory["product_id"].is_unique

    engine = PortfolioPricingEngine(
        PortfolioPricingConfig(
            default_spot=100.0,
            default_rate=0.03,
            default_volatility=0.20,
            spot_by_underlying={"MSFT": 100.0},
            volatility_by_underlying={"MSFT": 0.20},
            n_paths=2_000,
            n_steps=60,
            seed=7,
        )
    )
    priced = engine.price_portfolio(pricing_inventory)

    assert len(priced) == len(pricing_inventory)
    assert set(priced["portfolio"]) == set(demo_inventory["portfolio"])
    assert priced["status"].eq("priced").all()
