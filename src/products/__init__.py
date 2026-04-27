"""Products package exports."""

from src.products.base_product import Product
from src.products.barrier_option import BarrierOption
from src.products.option_strategy import OptionStrategy, OptionStrategyLeg
from src.products.structured_note import (
    CappedCapitalProtectedNote,
    CapitalProtectedNote,
    ReverseConvertible,
    StructuredNoteLeg,
)
from src.products.vanilla_option import VanillaOption
from src.products.zero_coupon_bond import ZeroCouponBond

__all__ = [
    "Product",
    "BarrierOption",
    "OptionStrategy",
    "OptionStrategyLeg",
    "CapitalProtectedNote",
    "CappedCapitalProtectedNote",
    "ReverseConvertible",
    "StructuredNoteLeg",
    "VanillaOption",
    "ZeroCouponBond",
]