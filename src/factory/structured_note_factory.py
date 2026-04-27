"""Factory mapping normalized inventory rows to structured-note objects."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.factory.registry import ProductFactoryRegistry
from src.products.structured_note import (
    CapitalProtectedNote,
    CappedCapitalProtectedNote,
    ReverseConvertible,
)


@dataclass(slots=True)
class StructuredNoteFactory:
    """Build structured-note objects from normalized inventory rows."""

    registry: ProductFactoryRegistry

    @classmethod
    def with_defaults(cls) -> "StructuredNoteFactory":
        registry = ProductFactoryRegistry()
        registry.register("capital_protected_note", CapitalProtectedNote)
        registry.register("capped_capital_protected_note", CappedCapitalProtectedNote)
        registry.register("reverse_convertible", ReverseConvertible)
        return cls(registry=registry)

    def build_from_inventory_row(
        self,
        row: dict | pd.Series,
        *,
        spot_reference: float,
        valuation_date: pd.Timestamp | None = None,
    ):
        data = dict(row)

        product_type = _infer_product_type(data)
        product_id = str(data.get("product_id") or f"SN-{data.get('source_row', 'X')}")

        maturity = _infer_maturity_years(data, valuation_date=valuation_date)
        participation_rate = float(_get_default(data, "participation_rate", 1.0))
        cap_level = float(_get_default(data, "cap", 1.30))
        coupon_rate = float(_get_default(data, "coupon_rate", 0.08))

        if product_type == "capital_protected_note":
            return self.registry.build(
                product_type,
                product_id=product_id,
                notional=float(_get_default(data, "quantity", 100.0)),
                maturity=maturity,
                spot_reference=spot_reference,
                participation_rate=participation_rate,
            )

        if product_type == "capped_capital_protected_note":
            return self.registry.build(
                product_type,
                product_id=product_id,
                notional=float(_get_default(data, "quantity", 100.0)),
                maturity=maturity,
                spot_reference=spot_reference,
                participation_rate=participation_rate,
                cap_level=cap_level,
            )

        return self.registry.build(
            "reverse_convertible",
            product_id=product_id,
            notional=float(_get_default(data, "quantity", 100.0)),
            maturity=maturity,
            spot_reference=spot_reference,
            coupon_rate=coupon_rate,
        )

    def build_many(
        self,
        inventory_frame: pd.DataFrame,
        *,
        spot_reference: float,
        valuation_date: pd.Timestamp | None = None,
    ) -> list:
        return [
            self.build_from_inventory_row(
                row,
                spot_reference=spot_reference,
                valuation_date=valuation_date,
            )
            for _, row in inventory_frame.iterrows()
        ]


def _get_default(data: dict, key: str, default):
    value = data.get(key)
    if value is None:
        return default
    if isinstance(value, float) and pd.isna(value):
        return default
    return value


def _infer_product_type(data: dict) -> str:
    text = str(data.get("product_type", "")).strip().lower()

    if "reverse" in text or "convertible" in text:
        return "reverse_convertible"
    if "capped" in text:
        return "capped_capital_protected_note"
    if "capital" in text or "protected" in text:
        return "capital_protected_note"

    sspa_code = _get_default(data, "sspa_code", None)
    if sspa_code is not None:
        try:
            code = int(float(sspa_code))
        except (ValueError, TypeError):
            code = None

        if code is not None:
            if 1200 <= code < 1300:
                return "capped_capital_protected_note"
            if 2200 <= code < 2300:
                return "reverse_convertible"
            if 1100 <= code < 1200:
                return "capital_protected_note"

    if pd.notna(data.get("cap")):
        return "capped_capital_protected_note"

    if pd.notna(data.get("barrier_1")) or pd.notna(data.get("barrier_2")):
        return "reverse_convertible"

    return "capital_protected_note"


def _infer_maturity_years(
    data: dict,
    *,
    valuation_date: pd.Timestamp | None,
) -> float:
    direct = data.get("time_to_maturity_years")
    if direct is not None and not (isinstance(direct, float) and pd.isna(direct)):
        maturity = float(direct)
        if maturity >= 0.0:
            return maturity

    if valuation_date is not None and pd.notna(data.get("maturity_date")):
        maturity_date = pd.to_datetime(data["maturity_date"])
        maturity = float((maturity_date - pd.to_datetime(valuation_date)).days / 365.25)
        return max(maturity, 0.0)

    return 1.0


__all__ = ["StructuredNoteFactory"]
