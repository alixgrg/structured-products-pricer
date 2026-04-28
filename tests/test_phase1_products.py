from __future__ import annotations

import pytest

from src.products.autocall import AutocallProduct
from src.products.barrier_option import BarrierOption
from src.products.coupon_bond import CouponBond
from src.products.option_strategies import Butterfly, CallSpread, PutSpread, Straddle
from src.products.structured_notes import (
    CapitalProtectedNote,
    CappedCapitalProtectedNote,
    ReverseConvertible,
    build_structured_note_from_inventory_row,
)
from src.products.swap import InterestRateSwap
from src.products.vanilla_option import VanillaOption


def test_coupon_bond_cash_flows_and_payoff() -> None:
    bond = CouponBond("CB-1", notional=1_000.0, maturity=2.0, coupon_rate=0.05, frequency=2, currency="EUR")

    cash_flows = bond.get_cash_flows()

    assert len(cash_flows) == 4
    assert cash_flows[0] == pytest.approx((0.5, 25.0))
    assert cash_flows[-1] == pytest.approx((2.0, 1_025.0))
    assert bond.payoff(None) == pytest.approx(1_000.0)
    assert bond.get_risk_factors() == ["rate"]


def test_swap_fixed_and_float_legs() -> None:
    swap = InterestRateSwap("IRS-1", notional=1_000_000.0, maturity=1.0, fixed_rate=0.03, float_index="EURIBOR6M", frequency="6M")

    fixed = swap.fixed_leg_cash_flows()
    floating = swap.float_leg_cash_flows(0.025)

    assert len(fixed) == 2
    assert fixed[0][1] == pytest.approx(15_000.0)
    assert floating[0][1] == pytest.approx(12_500.0)
    assert swap.payoff({"forward_rate": 0.025}) == pytest.approx(5_000.0)
    assert swap.get_risk_factors() == ["rate"]


def test_option_strategy_payoffs_match_static_replication() -> None:
    spot = {"spot": 115.0}
    call_spread = CallSpread("CS-1", maturity=1.0, strike_low=100.0, strike_high=120.0, notional=2.0)
    put_spread = PutSpread("PS-1", maturity=1.0, strike_low=90.0, strike_high=110.0)
    butterfly = Butterfly("BF-1", maturity=1.0, strike_low=90.0, strike_mid=100.0, strike_high=110.0)
    straddle = Straddle("ST-1", maturity=1.0, strike=100.0)

    assert call_spread.payoff(spot) == pytest.approx(2.0 * (max(115 - 100, 0) - max(115 - 120, 0)))
    assert put_spread.payoff({"spot": 85.0}) == pytest.approx(max(110 - 85, 0) - max(90 - 85, 0))
    assert butterfly.payoff({"spot": 100.0}) == pytest.approx(10.0)
    assert straddle.payoff({"spot": 115.0}) == pytest.approx(15.0)

    legs = call_spread.get_legs()
    assert len(legs) == 2
    assert isinstance(legs[0][0], VanillaOption)
    assert [qty for _, qty in legs] == [1.0, -1.0]


def test_barrier_knock_out_plus_knock_in_equals_vanilla_payoff() -> None:
    path = {"path": [100.0, 92.0, 88.0, 112.0]}
    ko = BarrierOption("B-KO", "call", 100.0, 1.0, barrier=90.0, barrier_type="KO", barrier_direction="down")
    ki = BarrierOption("B-KI", "call", 100.0, 1.0, barrier=90.0, barrier_type="KI", barrier_direction="down")
    vanilla = VanillaOption("V", "call", 100.0, 1.0)

    assert ko.barrier_touched(path)
    assert ko.payoff(path) + ki.payoff(path) == pytest.approx(vanilla.payoff({"spot": 112.0}))


def test_barrier_accepts_legacy_type_names() -> None:
    barrier = BarrierOption("LEGACY", "put", 100.0, 1.0, barrier=120.0, barrier_type="up-and-out")

    assert barrier.barrier_type == "KO"
    assert barrier.barrier_direction == "up"
    assert barrier.payoff({"path": [100.0, 121.0, 90.0]}) == pytest.approx(0.0)


def test_structured_notes_payoffs_and_decomposition() -> None:
    cpn = CapitalProtectedNote("CPN", 100.0, 1.0, spot_reference=100.0, participation_rate=0.8)
    capped = CappedCapitalProtectedNote("CCPN", 100.0, 1.0, spot_reference=100.0, participation_rate=1.0, cap_level=1.20)
    reverse = ReverseConvertible("RC", 100.0, 1.0, spot_reference=100.0, coupon_rate=0.10)

    assert cpn.payoff({"spot": 125.0}) == pytest.approx(120.0)
    assert len(cpn.decomposition()) == 2
    assert capped.payoff({"spot": 150.0}) == pytest.approx(120.0)
    assert len(capped.decomposition()) == 3
    assert reverse.payoff({"spot": 80.0}) == pytest.approx(90.0)
    assert len(reverse.decomposition()) == 2


def test_structured_note_builder_from_inventory_row() -> None:
    row = {
        "source_row": 1,
        "sspa_code": 1205,
        "quantity": 100.0,
        "participation_rate": 1.0,
        "cap": 130.0,
        "time_to_maturity_years": 2.0,
        "underlying": "SX5E",
    }

    note = build_structured_note_from_inventory_row(row, spot_reference=100.0)

    assert isinstance(note, CappedCapitalProtectedNote)
    assert note.cap_level == pytest.approx(1.30)
    assert note.underlying == "SX5E"


def test_autocall_payoff_early_redemption_and_protection() -> None:
    product = AutocallProduct(
        product_id="AC-1",
        underlying="SX5E",
        observation_dates=[1.0, 2.0, 3.0],
        trigger_levels=[1.0, 0.95, 0.90],
        coupon_rate=0.06,
        barrier_protection=0.70,
        notional=100.0,
        initial_spot=100.0,
    )

    assert product.requires_monte_carlo is True
    assert product.payoff({"path": [90.0, 96.0, 80.0]}) == pytest.approx(112.0)
    assert product.payoff({"path": [80.0, 75.0, 65.0]}) == pytest.approx(65.0)
    assert product.get_risk_factors() == ["spot", "rate", "volatility"]
