"""Financial product definitions for the structured-products pricer."""

from src.products.zero_coupon_bond import ZeroCouponBond
from src.products.vanilla_option import VanillaOption
from src.products.coupon_bond import CouponBond
from src.products.swap import InterestRateSwap
from src.products.option_strategies import (
    OptionLeg,
    OptionStrategy,
    CallSpread,
    PutSpread,
    Butterfly,
    Straddle,
)
from src.products.barrier_option import BarrierOption
from src.products.structured_notes import (
    ReplicationLeg,
    CapitalProtectedNote,
    CappedCapitalProtectedNote,
    ReverseConvertible,
    build_structured_note_from_inventory_row,
)
from src.products.autocall import AutocallProduct

__all__ = [
    "ZeroCouponBond",
    "VanillaOption",
    "CouponBond",
    "InterestRateSwap",
    "OptionLeg",
    "OptionStrategy",
    "CallSpread",
    "PutSpread",
    "Butterfly",
    "Straddle",
    "BarrierOption",
    "ReplicationLeg",
    "CapitalProtectedNote",
    "CappedCapitalProtectedNote",
    "ReverseConvertible",
    "build_structured_note_from_inventory_row",
    "AutocallProduct",
]
