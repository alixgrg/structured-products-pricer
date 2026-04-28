from __future__ import annotations

import pandas as pd
import pytest

from src.portfolio.inventory_loader import build_pricing_inventory
from src.factory.builders import build_product_from_row
from src.market.market_data import MarketData
from src.models.discounting_model import DiscountingModel
from src.products.autocall import AutocallProduct
from src.products.basis_swap import BasisSwap


def test_build_pricing_inventory_groups_autocall_and_uses_real_maturity() -> None:
    frame = pd.DataFrame(
        {
            "source_sheet": ["autocalls", "autocalls", "autocalls"],
            "source_row": [1, 2, 3],
            "valuation_date": pd.to_datetime(["2026-02-27"] * 3),
            "product_id": [1, 1, 1],
            "underlying": ["AAPL", "AAPL", "AAPL"],
            "observation_date": pd.to_datetime(["2027-02-27", "2028-02-27", "2029-02-27"]),
            "autocall_trigger_level": [1.0, 1.0, 1.0],
            "coupon_rate": [0.07, 0.14, 0.21],
        }
    )

    pricing = build_pricing_inventory({"autocalls": frame})

    assert len(pricing) == 1
    assert pricing.iloc[0]["product_type"] == "Autocall"
    assert pricing.iloc[0]["time_to_maturity_years"] == pytest.approx(3.000684, rel=1e-3)

    product = build_product_from_row(pricing.iloc[0])
    assert isinstance(product, AutocallProduct)
    assert product.maturity == pytest.approx(float(pricing.iloc[0]["time_to_maturity_years"]))


def test_build_pricing_inventory_detects_basis_swap() -> None:
    swaps = pd.DataFrame(
        {
            "source_sheet": ["swaps"],
            "source_row": [1],
            "valuation_date": pd.to_datetime(["2026-02-27"]),
            "maturity_date": pd.to_datetime(["2032-07-31"]),
            "time_to_maturity_years": [6.42],
            "currency": ["EUR"],
            "notional": [5_000_000.0],
            "fixed_rate": [pd.NA],
            "floating_rate_index_1": ["6M"],
            "floating_rate_index_2": ["3M"],
        }
    )

    pricing = build_pricing_inventory({"swaps": swaps})
    assert pricing.iloc[0]["product_type"] == "Basis Swap"

    product = build_product_from_row(pricing.iloc[0])
    assert isinstance(product, BasisSwap)

    price = DiscountingModel(rate=0.03).price(product, MarketData(rate=0.03))
    assert isinstance(price, float)


def test_negative_quantity_is_positive_product_notional_but_negative_position_sign() -> None:
    options = pd.DataFrame(
        {
            "source_sheet": ["options"],
            "source_row": [1],
            "valuation_date": pd.to_datetime(["2026-02-27"]),
            "product_type": ["Put Spread"],
            "underlying": ["AAPL"],
            "maturity_date": pd.to_datetime(["2026-09-30"]),
            "time_to_maturity_years": [0.588638],
            "quantity": [-100000.0],
            "strike_1": [200.0],
            "strike_2": [230.0],
        }
    )

    pricing = build_pricing_inventory({"options": options})
    assert pricing.iloc[0]["position_sign"] == -1.0
    assert pricing.iloc[0]["quantity"] == 100000.0

    product = build_product_from_row(pricing.iloc[0])
    assert product.notional == pytest.approx(100000.0)
