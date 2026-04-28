"""Booking and product-size conventions for portfolio inventories.

This layer translates raw inventory quantities into pricing quantities/notionals.
It is intentionally separated from:
- io_utils.py: technical parsing of dates/numbers;
- inventory_loader.py: sheet and column normalization;
- pricing_engine.py: pricing and risk calculation.

The goal is to make financial size conventions explicit before pricing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd


PriceUnit = Literal["amount", "percent_notional", "unit"]


@dataclass(frozen=True, slots=True)
class BookingConvention:
    """Sizing convention for one product family."""

    contract_multiplier: float = 1.0
    default_notional: float | None = None
    price_unit: PriceUnit = "amount"


DEFAULT_BOOKING_CONVENTIONS: dict[str, BookingConvention] = {
    # Listed equity options are often quoted per share with 100 shares per contract.
    # In this student project, set this to 1.0 if quantity already represents notional.
    "vanilla_option": BookingConvention(contract_multiplier=1.0, price_unit="amount"),
    "option_strategy": BookingConvention(contract_multiplier=1.0, price_unit="amount"),
    "barrier": BookingConvention(contract_multiplier=1.0, price_unit="amount"),

    # Structured products are usually quoted for a nominal amount.
    "structured_note": BookingConvention(contract_multiplier=1.0, default_notional=100.0, price_unit="amount"),
    "capital_protected": BookingConvention(contract_multiplier=1.0, default_notional=100.0, price_unit="amount"),
    "structured_yield": BookingConvention(contract_multiplier=1.0, default_notional=100.0, price_unit="amount"),

    # Autocall payoff should be expressed on the same nominal basis as notes.
    "autocall": BookingConvention(contract_multiplier=1.0, default_notional=100.0, price_unit="amount"),

    # Rates already use notional in the workbook.
    "rates": BookingConvention(contract_multiplier=1.0, price_unit="amount"),
    "other": BookingConvention(contract_multiplier=1.0, price_unit="amount"),
}


def apply_booking_conventions(
    inventory: pd.DataFrame,
    *,
    conventions: dict[str, BookingConvention] | None = None,
    equity_option_contract_multiplier: float | None = None,
) -> pd.DataFrame:
    """Apply product sizing conventions to a normalized inventory dataframe.

    Adds:
    - product_family_for_booking
    - contract_multiplier
    - booking_notional
    - price_unit

    Also updates notional where it is missing, so downstream product builders
    receive coherent notionals.
    """
    if inventory is None or inventory.empty:
        return pd.DataFrame()

    rules = dict(DEFAULT_BOOKING_CONVENTIONS)
    if conventions:
        rules.update(conventions)

    if equity_option_contract_multiplier is not None:
        for family in ("vanilla_option", "option_strategy", "barrier"):
            old = rules[family]
            rules[family] = BookingConvention(
                contract_multiplier=float(equity_option_contract_multiplier),
                default_notional=old.default_notional,
                price_unit=old.price_unit,
            )

    data = inventory.copy()
    data["product_family_for_booking"] = data.apply(_classify_booking_family, axis=1)

    data["contract_multiplier"] = data["product_family_for_booking"].map(
        lambda family: rules.get(str(family), rules["other"]).contract_multiplier
    ).astype(float)

    data["price_unit"] = data["product_family_for_booking"].map(
        lambda family: rules.get(str(family), rules["other"]).price_unit
    )

    if "quantity" in data.columns:
        quantity = pd.to_numeric(data["quantity"], errors="coerce")
    else:
        quantity = pd.Series(1.0, index=data.index)

    if "notional" in data.columns:
        notional = pd.to_numeric(data["notional"], errors="coerce")
    else:
        notional = pd.Series(pd.NA, index=data.index, dtype="Float64")

    default_notional = data["product_family_for_booking"].map(
        lambda family: rules.get(str(family), rules["other"]).default_notional
    )
    default_notional = pd.to_numeric(default_notional, errors="coerce")

    # Main convention:
    # - if explicit notional exists, keep it;
    # - otherwise use abs(quantity) * contract_multiplier;
    # - if quantity is absent and product has a default nominal, use the default.
    booking_notional = notional.copy()
    booking_notional = booking_notional.where(booking_notional.notna(), quantity.abs() * data["contract_multiplier"])
    booking_notional = booking_notional.where(booking_notional.notna(), default_notional)
    booking_notional = booking_notional.fillna(1.0)

    data["booking_notional"] = booking_notional.astype(float)

    if "notional" not in data.columns:
        data["notional"] = data["booking_notional"]
    else:
        data["notional"] = pd.to_numeric(data["notional"], errors="coerce")
        data["notional"] = data["notional"].fillna(data["booking_notional"])

    return data


def _classify_booking_family(row: pd.Series) -> str:
    source_sheet = str(row.get("source_sheet", "")).strip().lower()
    product_type = str(row.get("product_type", "")).strip().lower()
    product_family = str(row.get("product_family", "")).strip().lower()

    if product_family:
        return product_family

    if source_sheet in {"swaps", "swap", "bonds", "bond", "rates"}:
        return "rates"

    if source_sheet in {"autocalls", "autocall"}:
        return "autocall"

    if source_sheet in {"structured_notes", "notes_structurees"}:
        if "reverse" in product_type or "convertible" in product_type:
            return "structured_yield"
        return "capital_protected"

    if source_sheet in {"options", "option"}:
        has_barrier = pd.notna(row.get("barrier_level", pd.NA)) or pd.notna(row.get("barrier_type", pd.NA))
        if has_barrier or "barrier" in product_type or "knock" in product_type:
            return "barrier"
        if "spread" in product_type or "butterfly" in product_type or "straddle" in product_type:
            return "option_strategy"
        if "call" in product_type or "put" in product_type:
            return "vanilla_option"

    return "other"


__all__ = [
    "BookingConvention",
    "DEFAULT_BOOKING_CONVENTIONS",
    "apply_booking_conventions",
]