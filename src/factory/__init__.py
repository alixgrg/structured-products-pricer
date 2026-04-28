"""Factory layer public API."""

from src.factory.builders import (
    build_autocall,
    build_autocalls_from_frame,
    build_barrier_option,
    build_basis_swap,
    build_butterfly,
    build_call_spread,
    build_coupon_bond,
    build_interest_rate_swap,
    build_product_from_row,
    build_put_spread,
    build_straddle,
    build_structured_note,
    build_vanilla_option,
    build_zero_coupon_bond,
    create_default_product_registry,
    infer_product_type_key,
)
from src.factory.pricing_router import PricingRouter
from src.factory.registry import ProductFactoryRegistry
from src.factory.structured_note_factory import StructuredNoteFactory

__all__ = [
    "ProductFactoryRegistry",
    "PricingRouter",
    "StructuredNoteFactory",
    "build_autocall",
    "build_autocalls_from_frame",
    "build_barrier_option",
    "build_basis_swap",
    "build_butterfly",
    "build_call_spread",
    "build_coupon_bond",
    "build_interest_rate_swap",
    "build_product_from_row",
    "build_put_spread",
    "build_straddle",
    "build_structured_note",
    "build_vanilla_option",
    "build_zero_coupon_bond",
    "create_default_product_registry",
    "infer_product_type_key",
]
