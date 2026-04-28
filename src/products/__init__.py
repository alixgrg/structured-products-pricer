"""Financial product definitions for the structured-products pricer."""

from src.products.autocall import AutocallProduct
from src.products.barrier_option import BarrierOption
from src.products.base_product import Product
from src.products.basis_swap import BasisSwap
from src.products.coupon_bond import CouponBond
from src.products.option_strategies import (
    Butterfly,
    CallSpread,
    OptionLeg,
    OptionStrategy,
    PutSpread,
    Straddle,
)
from src.products.structured_notes import (
    CapitalProtectedNote,
    CappedCapitalProtectedNote,
    ReplicationLeg,
    ReverseConvertible,
    build_structured_note_from_inventory_row,
)
from src.products.swap import InterestRateSwap
from src.products.vanilla_option import VanillaOption
from src.products.zero_coupon_bond import ZeroCouponBond

__all__ = [
    "AutocallProduct",
    "BarrierOption",
    "BasisSwap",
    "Butterfly",
    "CallSpread",
    "CapitalProtectedNote",
    "CappedCapitalProtectedNote",
    "CouponBond",
    "InterestRateSwap",
    "OptionLeg",
    "OptionStrategy",
    "Product",
    "PutSpread",
    "ReplicationLeg",
    "ReverseConvertible",
    "Straddle",
    "VanillaOption",
    "ZeroCouponBond",
    "build_structured_note_from_inventory_row",
]
