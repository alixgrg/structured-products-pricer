from __future__ import annotations

import pytest

from src.market.market_data import MarketData
from src.models.discounting_model import DiscountingModel
from src.products.basis_swap import BasisSwap


def test_basis_swap_prices_and_has_rate_risk() -> None:
    product = BasisSwap(
        product_id="BASIS-6M-3M",
        notional=1_000_000.0,
        maturity=2.0,
        receive_index="6M",
        pay_index="3M",
        spread=0.001,
    )
    model = DiscountingModel(rate=0.03)

    price = model.price(product, MarketData(rate=0.03))
    risk = model.risk(product, MarketData(rate=0.03))

    assert isinstance(price, float)
    assert risk["price"] == pytest.approx(price)
    assert "dv01" in risk
    assert "rho" in risk
