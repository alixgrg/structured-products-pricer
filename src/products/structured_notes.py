"""Structured-note products expressed through static replication."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.products._helpers import extract_spot, normalize_non_negative_float, normalize_positive_float
from src.products.base_product import Product
from src.products.vanilla_option import VanillaOption
from src.products.zero_coupon_bond import ZeroCouponBond


@dataclass(slots=True)
class ReplicationLeg:
    """Single leg of a static replication decomposition."""

    product: object
    quantity: float = 1.0

    def __post_init__(self) -> None:
        self.quantity = float(self.quantity)
        if self.quantity == 0.0:
            raise ValueError("ReplicationLeg quantity cannot be zero.")


@dataclass(slots=True)
class _StructuredNoteBase(Product):
    product_id: str
    notional: float
    maturity: float
    spot_reference: float
    underlying: str = ""
    currency: str = "EUR"

    def __post_init__(self) -> None:
        self.notional = normalize_positive_float(self.notional, "notional")
        self.maturity = normalize_non_negative_float(self.maturity, "maturity")
        self.spot_reference = normalize_positive_float(self.spot_reference, "spot_reference")
        self.underlying = str(self.underlying).strip().upper()
        self.currency = str(self.currency).strip().upper()

    def decomposition(self) -> list[ReplicationLeg]:
        raise NotImplementedError

    def price(self, option_model, discount_model, market_data=None) -> float:
        total = 0.0
        for leg in self.decomposition():
            if isinstance(leg.product, VanillaOption):
                total += leg.quantity * option_model.price(leg.product, market_data)
            elif isinstance(leg.product, ZeroCouponBond):
                total += leg.quantity * discount_model.price(leg.product, market_data)
            else:
                raise TypeError(f"Unsupported replication leg: {type(leg.product)!r}")
        return float(total)

    def get_risk_factors(self) -> list[str]:
        return ["spot", "rate", "volatility"]


@dataclass(slots=True)
class CapitalProtectedNote(_StructuredNoteBase):
    """Capital protected note = ZCB + participation * ATM call."""

    participation_rate: float = 1.0

    def __post_init__(self) -> None:
        _StructuredNoteBase.__post_init__(self)
        self.participation_rate = float(self.participation_rate)
        if self.participation_rate < 0.0:
            raise ValueError("participation_rate must be non-negative.")

    def decomposition(self) -> list[ReplicationLeg]:
        option_units = self.notional * self.participation_rate / self.spot_reference
        return [
            ReplicationLeg(ZeroCouponBond(f"{self.product_id}-ZCB", self.notional, self.maturity, self.currency), 1.0),
            ReplicationLeg(
                VanillaOption(
                    f"{self.product_id}-CALL",
                    "call",
                    self.spot_reference,
                    self.maturity,
                    notional=option_units,
                    underlying=self.underlying,
                    currency=self.currency,
                ),
                1.0,
            ),
        ]

    def payoff(self, market_data) -> float:
        spot = extract_spot(market_data)
        performance = max(spot / self.spot_reference - 1.0, 0.0)
        return float(self.notional * (1.0 + self.participation_rate * performance))


@dataclass(slots=True)
class CappedCapitalProtectedNote(_StructuredNoteBase):
    """Capital protected note with capped upside = ZCB + call spread."""

    participation_rate: float = 1.0
    cap_level: float = 1.30

    def __post_init__(self) -> None:
        _StructuredNoteBase.__post_init__(self)
        self.participation_rate = float(self.participation_rate)
        if self.participation_rate < 0.0:
            raise ValueError("participation_rate must be non-negative.")
        self.cap_level = _normalize_level(self.cap_level, default=1.30)
        if self.cap_level <= 1.0:
            raise ValueError("cap_level must be greater than 1.0, e.g. 1.30 for 130%.")

    def decomposition(self) -> list[ReplicationLeg]:
        option_units = self.notional * self.participation_rate / self.spot_reference
        cap_strike = self.spot_reference * self.cap_level
        return [
            ReplicationLeg(ZeroCouponBond(f"{self.product_id}-ZCB", self.notional, self.maturity, self.currency), 1.0),
            ReplicationLeg(
                VanillaOption(f"{self.product_id}-CALL-K1", "call", self.spot_reference, self.maturity, option_units, self.underlying, self.currency),
                1.0,
            ),
            ReplicationLeg(
                VanillaOption(f"{self.product_id}-CALL-K2", "call", cap_strike, self.maturity, option_units, self.underlying, self.currency),
                -1.0,
            ),
        ]

    def payoff(self, market_data) -> float:
        spot = extract_spot(market_data)
        capped_performance = min(max(spot / self.spot_reference - 1.0, 0.0), self.cap_level - 1.0)
        return float(self.notional * (1.0 + self.participation_rate * capped_performance))


@dataclass(slots=True)
class ReverseConvertible(_StructuredNoteBase):
    """Reverse convertible = coupon bond proxy + short put."""

    coupon_rate: float = 0.08
    barrier_level: float | None = None

    def __post_init__(self) -> None:
        _StructuredNoteBase.__post_init__(self)
        self.coupon_rate = float(self.coupon_rate)
        if self.barrier_level is not None:
            self.barrier_level = _normalize_level(self.barrier_level, default=0.70)

    def decomposition(self) -> list[ReplicationLeg]:
        redemption_plus_coupon = self.notional * (1.0 + self.coupon_rate * self.maturity)
        option_units = self.notional / self.spot_reference
        return [
            ReplicationLeg(ZeroCouponBond(f"{self.product_id}-BOND", redemption_plus_coupon, self.maturity, self.currency), 1.0),
            ReplicationLeg(
                VanillaOption(f"{self.product_id}-SHORT-PUT", "put", self.spot_reference, self.maturity, option_units, self.underlying, self.currency),
                -1.0,
            ),
        ]

    def payoff(self, market_data) -> float:
        spot = extract_spot(market_data)
        coupon = self.notional * self.coupon_rate * self.maturity
        loss = self.notional * max(1.0 - spot / self.spot_reference, 0.0)
        return float(self.notional + coupon - loss)


def build_structured_note_from_inventory_row(
    row: dict | pd.Series,
    *,
    spot_reference: float,
    valuation_date: str | pd.Timestamp | None = None,
) -> _StructuredNoteBase:
    """Build a structured note from normalized inventory columns.

    Expected columns are compatible with the inventory loader:
    product_type, sspa_code, participation_rate, barrier_1, cap, quantity,
    maturity_date, time_to_maturity_years, underlying.
    """
    data = dict(row)
    product_type = _infer_product_type(data)
    product_id = str(_get_default(data, "product_id", f"SN-{_get_default(data, 'source_row', 'X')}"))
    notional = float(_get_default(data, "notional", _get_default(data, "quantity", 100.0)))
    maturity = _infer_maturity_years(data, valuation_date=valuation_date)
    underlying = str(_get_default(data, "underlying", "")).strip().upper()
    currency = str(_get_default(data, "currency", _get_default(data, "rate_currency", "EUR"))).strip().upper()

    if product_type == "capital_protected_note":
        return CapitalProtectedNote(
            product_id=product_id,
            notional=notional,
            maturity=maturity,
            spot_reference=spot_reference,
            underlying=underlying,
            currency=currency,
            participation_rate=float(_get_default(data, "participation_rate", 1.0)),
        )
    if product_type == "capped_capital_protected_note":
        return CappedCapitalProtectedNote(
            product_id=product_id,
            notional=notional,
            maturity=maturity,
            spot_reference=spot_reference,
            underlying=underlying,
            currency=currency,
            participation_rate=float(_get_default(data, "participation_rate", 1.0)),
            cap_level=float(_get_default(data, "cap", 1.30)),
        )
    return ReverseConvertible(
        product_id=product_id,
        notional=notional,
        maturity=maturity,
        spot_reference=spot_reference,
        underlying=underlying,
        currency=currency,
        coupon_rate=float(_get_default(data, "coupon_rate", 0.08)),
        barrier_level=_get_optional_float(data, "barrier_1"),
    )


def _get_default(data: dict, key: str, default: Any) -> Any:
    value = data.get(key, default)
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    return value


def _get_optional_float(data: dict, key: str) -> float | None:
    value = _get_default(data, key, None)
    if value is None:
        return None
    return float(value)


def _infer_product_type(data: dict) -> str:
    text = str(data.get("product_type", "")).strip().lower()

    if "reverse" in text or "convertible" in text:
        return "reverse_convertible"

    # Important : ne pas tester simplement "cap" in text,
    # car "capital" commence par "cap".
    if (
        "capped" in text
        or "capped capital" in text
        or "capital protected capped" in text
        or "capital protected note capped" in text
    ):
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

    # Ici on teste la colonne cap, pas le mot "cap" dans product_type.
    cap_value = _get_default(data, "cap", None)
    if cap_value is not None:
        return "capped_capital_protected_note"

    if _get_default(data, "barrier_1", None) is not None:
        return "reverse_convertible"

    return "capital_protected_note"

def _infer_maturity_years(data: dict, *, valuation_date: str | pd.Timestamp | None) -> float:
    direct = _get_default(data, "time_to_maturity_years", None)
    if direct is not None:
        return max(float(direct), 0.0)
    if valuation_date is not None and _get_default(data, "maturity_date", None) is not None:
        maturity_date = pd.Timestamp(data["maturity_date"])
        val_date = pd.Timestamp(valuation_date)
        return max(float((maturity_date - val_date).days / 365.25), 0.0)
    return 1.0


def _normalize_level(value: float, *, default: float) -> float:
    if value is None:
        return default
    level = float(value)
    if level > 10.0:
        level = level / 100.0
    return level


__all__ = [
    "ReplicationLeg",
    "CapitalProtectedNote",
    "CappedCapitalProtectedNote",
    "ReverseConvertible",
    "build_structured_note_from_inventory_row",
]
