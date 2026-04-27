from __future__ import annotations

import pandas as pd
import pytest

from src.factory.structured_note_factory import StructuredNoteFactory
from src.market.market_data import MarketData
from src.models.black_scholes import BlackScholesModel
from src.models.discounting_model import DiscountingModel
from src.products.structured_note import (
    CappedCapitalProtectedNote,
    CapitalProtectedNote,
    ReverseConvertible,
)


def test_capital_protected_note_payoff_floor_is_preserved() -> None:
    note = CapitalProtectedNote(
        product_id="CPN-1",
        notional=100.0,
        maturity=1.0,
        spot_reference=100.0,
        participation_rate=1.0,
    )

    assert note.payoff(70.0) == pytest.approx(100.0)
    assert note.payoff(140.0) == pytest.approx(140.0)


def test_capped_capital_protected_note_payoff_matches_decomposition() -> None:
    note = CappedCapitalProtectedNote(
        product_id="CCPN-1",
        notional=100.0,
        maturity=1.0,
        spot_reference=100.0,
        participation_rate=1.0,
        cap_level=1.20,
    )

    for spot in (80.0, 100.0, 110.0, 130.0):
        direct = note.payoff(spot)
        decomposed = sum(leg.quantity * leg.product.payoff(spot) for leg in note.decomposition())
        assert direct == pytest.approx(decomposed, rel=1e-12)


def test_reverse_convertible_payoff_matches_decomposition() -> None:
    note = ReverseConvertible(
        product_id="RC-1",
        notional=100.0,
        maturity=1.0,
        spot_reference=100.0,
        coupon_rate=0.10,
    )

    for spot in (70.0, 100.0, 130.0):
        direct = note.payoff(spot)
        decomposed = sum(leg.quantity * leg.product.payoff(spot) for leg in note.decomposition())
        assert direct == pytest.approx(decomposed, rel=1e-12)


def test_structured_note_price_equals_sum_of_brick_prices() -> None:
    note = CapitalProtectedNote(
        product_id="CPN-2",
        notional=100.0,
        maturity=1.0,
        spot_reference=100.0,
        participation_rate=0.8,
    )

    bs = BlackScholesModel()
    disc = DiscountingModel(rate=0.03)
    market = MarketData(spot=100.0, rate=0.03, volatility=0.20, dividend_yield=0.0)

    direct_price = note.price(bs, disc, market)
    decomposition_price = 0.0
    for leg in note.decomposition():
        if hasattr(leg.product, "option_type"):
            leg_price = bs.price(leg.product, market)
        else:
            leg_price = disc.price(leg.product, market)
        decomposition_price += leg.quantity * leg_price

    assert direct_price == pytest.approx(decomposition_price, rel=1e-12)


def test_structured_note_factory_maps_inventory_rows() -> None:
    factory = StructuredNoteFactory.with_defaults()

    inventory = pd.DataFrame(
        [
            {
                "product_id": "SN-CPN",
                "product_type": "capital protected note",
                "quantity": 100.0,
                "participation_rate": 0.9,
                "time_to_maturity_years": 1.0,
            },
            {
                "product_id": "SN-CCPN",
                "product_type": "capped capital protected note",
                "quantity": 100.0,
                "participation_rate": 0.8,
                "cap": 1.25,
                "time_to_maturity_years": 1.0,
            },
            {
                "product_id": "SN-RC",
                "product_type": "reverse convertible",
                "quantity": 100.0,
                "coupon_rate": 0.12,
                "time_to_maturity_years": 1.0,
            },
        ]
    )

    products = factory.build_many(inventory, spot_reference=100.0)

    assert isinstance(products[0], CapitalProtectedNote)
    assert isinstance(products[1], CappedCapitalProtectedNote)
    assert isinstance(products[2], ReverseConvertible)
