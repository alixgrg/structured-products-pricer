from __future__ import annotations

import pandas as pd

from src.factory.builders import build_product_from_row, create_default_product_registry
from src.factory.pricing_router import PricingRouter
from src.market.market_data import MarketData


def main() -> None:
    registry = create_default_product_registry()
    router = PricingRouter.with_defaults(rate=0.03, volatility=0.20, n_paths=10_000, n_steps=100, seed=42)
    market = MarketData(spot=100.0, rate=0.03, volatility=0.20, dividend_yield=0.0)

    rows = [
        {
            "product_id": "CALL-1",
            "product_type": "Call",
            "underlying": "AAPL",
            "time_to_maturity_years": 1.0,
            "strike_1": 100.0,
            "quantity": 1.0,
        },
        {
            "product_id": "CS-1",
            "product_type": "Call Spread",
            "underlying": "AAPL",
            "time_to_maturity_years": 1.0,
            "strike_1": 95.0,
            "strike_2": 110.0,
            "quantity": 1.0,
        },
        {
            "product_id": "BARRIER-1",
            "product_type": "Put Down-and-Out",
            "underlying": "AAPL",
            "time_to_maturity_years": 1.0,
            "strike_1": 100.0,
            "barrier_level": 75.0,
            "quantity": 1.0,
        },
        {
            "source_sheet": "structured_notes",
            "product_id": "SN-1",
            "product_type": "Capital Protected Note",
            "underlying": "AAPL",
            "time_to_maturity_years": 2.0,
            "quantity": 100.0,
            "participation_rate": 1.0,
            "spot_reference": 100.0,
        },
        {
            "source_sheet": "swaps",
            "product_id": "IRS-1",
            "notional": 1_000_000.0,
            "time_to_maturity_years": 3.0,
            "fixed_rate": 0.03,
            "fixed_leg_frequency": "6M",
            "floating_rate_index_1": "EURIBOR6M",
        },
    ]

    print("PHASE 3 VALIDATION")
    print("==================")
    for raw in rows:
        product = build_product_from_row(pd.Series(raw), registry=registry)
        model = router.model_for(product)
        price = router.price(product, market)
        risk = router.risk(product, market)
        print(f"{raw.get('product_id', 'NA'):<12} {type(product).__name__:<28} -> {type(model).__name__:<25} price={price:,.6f} delta={risk.get('delta', 0.0):,.6f}")

    print("\nOK - Phase 3 factory + pricing router validated.")


if __name__ == "__main__":
    main()
