"""Product builders from normalized inventory rows.

Phase 3 objective
-----------------
Centralize all row -> product construction logic so the portfolio engine does
not have to know product-specific constructor details.

The builders are intentionally tolerant: they accept ``pd.Series`` or ``dict``
and support both the normalized inventory columns produced by
``inventory_loader`` and a few common aliases used in tests/notebooks.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.factory.registry import ProductFactoryRegistry
from src.products.autocall import AutocallProduct
from src.products.barrier_option import BarrierOption
from src.products.coupon_bond import CouponBond
from src.products.option_strategies import Butterfly, CallSpread, PutSpread, Straddle
from src.products.structured_notes import build_structured_note_from_inventory_row
from src.products.swap import InterestRateSwap
from src.products.vanilla_option import VanillaOption
from src.products.zero_coupon_bond import ZeroCouponBond

RowLike = pd.Series | dict[str, Any]
Builder = Callable[..., object]


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------


def build_vanilla_option(row: RowLike, **_: Any) -> VanillaOption:
    option_type = _option_type(row)
    return VanillaOption(
        product_id=_product_id(row, prefix=f"{option_type.upper()}"),
        option_type=option_type,
        strike=_strike(row),
        maturity=_maturity(row),
        notional=_notional(row, default=1.0),
        underlying=_underlying(row),
        currency=_currency(row),
        dividend_yield=_optional_float(_get(row, "dividend_yield", "q")),
    )


def build_call_spread(row: RowLike, **_: Any) -> CallSpread:
    k1, k2 = _two_strikes(row)
    return CallSpread(
        product_id=_product_id(row, prefix="CS"),
        maturity=_maturity(row),
        strike_low=min(k1, k2),
        strike_high=max(k1, k2),
        underlying=_underlying(row),
        notional=_notional(row, default=1.0),
        currency=_currency(row),
    )


def build_put_spread(row: RowLike, **_: Any) -> PutSpread:
    k1, k2 = _two_strikes(row)
    return PutSpread(
        product_id=_product_id(row, prefix="PS"),
        maturity=_maturity(row),
        strike_low=min(k1, k2),
        strike_high=max(k1, k2),
        underlying=_underlying(row),
        notional=_notional(row, default=1.0),
        currency=_currency(row),
    )


def build_butterfly(row: RowLike, **_: Any) -> Butterfly:
    strikes = sorted([_float_required(_get(row, "strike_1"), "strike_1"), _float_required(_get(row, "strike_2"), "strike_2"), _float_required(_get(row, "strike_3"), "strike_3")])
    return Butterfly(
        product_id=_product_id(row, prefix="BF"),
        maturity=_maturity(row),
        strike_low=strikes[0],
        strike_mid=strikes[1],
        strike_high=strikes[2],
        underlying=_underlying(row),
        notional=_notional(row, default=1.0),
        currency=_currency(row),
    )


def build_straddle(row: RowLike, **_: Any) -> Straddle:
    return Straddle(
        product_id=_product_id(row, prefix="STD"),
        maturity=_maturity(row),
        strike=_strike(row),
        underlying=_underlying(row),
        notional=_notional(row, default=1.0),
        currency=_currency(row),
    )


def build_barrier_option(row: RowLike, **_: Any) -> BarrierOption:
    product_type = str(_get(row, "product_type", default="")).strip().lower()
    barrier_type_raw = str(_get(row, "barrier_type", default="")).strip().lower()
    combined = f"{product_type} {barrier_type_raw}".replace("_", "-")

    option_type = _option_type(row)

    strike = _float_required(
        _get(row, "strike", "strike_1", default=None),
        "strike",
    )

    barrier = _float_required(
        _get(row, "barrier_level", "barrier", "barrier_1", default=None),
        "barrier",
    )

    direction = _barrier_direction(product_type, barrier_type_raw)
    if direction is None:
        direction = "down" if float(barrier) < float(strike) else "up"

    barrier_type = _barrier_kind(product_type, barrier_type_raw)

    return BarrierOption(
        product_id=_product_id(row, prefix=f"{option_type.upper()}-BARRIER"),
        option_type=option_type,
        strike=float(strike),
        maturity=_maturity(row),
        barrier=float(barrier),
        barrier_type=barrier_type,
        barrier_direction=direction,
        notional=_notional(row, default=1.0),
        underlying=_underlying(row),
        currency=_currency(row),
        dividend_yield=_optional_float(_get(row, "dividend_yield", "q", default=None)),
    )


def build_zero_coupon_bond(row: RowLike, **_: Any) -> ZeroCouponBond:
    return ZeroCouponBond(
        product_id=_product_id(row, prefix="ZCB"),
        notional=_notional(row, default=100.0),
        maturity=_maturity(row),
        currency=_currency(row),
    )


def build_coupon_bond(row: RowLike, **_: Any) -> CouponBond:
    return CouponBond(
        product_id=_product_id(row, prefix="BOND"),
        notional=_notional(row, default=100.0),
        maturity=_maturity(row),
        coupon_rate=_float_required(_get(row, "coupon_rate", "fixed_rate", "coupon"), "coupon_rate"),
        frequency=_get(row, "frequency", "fixed_leg_frequency", default="1Y"),
        currency=_currency(row),
    )


def build_interest_rate_swap(row: RowLike, **_: Any) -> InterestRateSwap:
    float_index = _get(row, "float_index", "floating_rate_index", "floating_rate_index_1", "taux_variable_1", default="EURIBOR6M")
    return InterestRateSwap(
        product_id=_product_id(row, prefix="IRS"),
        notional=_notional(row, default=1_000_000.0),
        maturity=_maturity(row),
        fixed_rate=_float_required(_get(row, "fixed_rate", "taux_fixe"), "fixed_rate"),
        float_index=str(float_index),
        frequency=_get(row, "frequency", "fixed_leg_frequency", default="6M"),
        currency=_currency(row),
    )


def build_structured_note(row: RowLike, *, spot_reference: float | None = None, **_: Any):
    spot_ref = _resolve_spot_reference(row, spot_reference)
    valuation_date = _get(row, "valuation_date", "date_valorisation", default=None)
    return build_structured_note_from_inventory_row(
        row,
        spot_reference=spot_ref,
        valuation_date=valuation_date,
    )


def build_autocall(row: RowLike, **_: Any) -> AutocallProduct:
    observation_dates = _list_value(_get(row, "observation_dates", default=None))
    if not observation_dates:
        observation_dates = [_get(row, "observation_date", "date_observation", default=1.0)]

    trigger_levels = _list_value(_get(row, "trigger_levels", default=None))
    if not trigger_levels:
        trigger_levels = [_float_required(_get(row, "autocall_trigger_level", "trigger_level", "niveau_de_rappel"), "autocall_trigger_level")]

    return AutocallProduct(
        product_id=_product_id(row, prefix="AUTO"),
        underlying=_underlying(row),
        observation_dates=observation_dates,
        trigger_levels=[float(x) for x in trigger_levels],
        coupon_rate=_float_required(_get(row, "coupon_rate", "coupon"), "coupon_rate"),
        barrier_protection=_float_required(_get(row, "barrier_protection", "protection_barrier", "barrier_1", default=70.0), "barrier_protection"),
        notional=_notional(row, default=100.0),
        initial_spot=_resolve_spot_reference(row, None),
        currency=_currency(row),
    )


def build_autocalls_from_frame(frame: pd.DataFrame) -> list[AutocallProduct]:
    """Build one autocall per product_id from a normalized autocall inventory sheet.

    The normalized sheet usually has one row per observation date. This helper
    groups those rows into one product each.
    """
    if frame.empty:
        return []

    group_key = "product_id" if "product_id" in frame.columns else None
    if group_key is None:
        return [build_autocall(row) for _, row in frame.iterrows()]

    products: list[AutocallProduct] = []
    for _, group in frame.groupby(group_key, dropna=False, sort=False):
        first = group.iloc[0].copy()
        first["observation_dates"] = group.get("observation_date", pd.Series(range(1, len(group) + 1))).tolist()
        first["trigger_levels"] = group.get("autocall_trigger_level", pd.Series([first.get("trigger_level", 1.0)] * len(group))).tolist()
        products.append(build_autocall(first))
    return products


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------


def create_default_product_registry() -> ProductFactoryRegistry:
    registry = ProductFactoryRegistry()

    _register_aliases(registry, build_vanilla_option, ["call", "put", "vanilla call", "vanilla put", "vanilla option", "european option"])
    _register_aliases(registry, build_call_spread, ["call spread", "bull spread", "call_spread"])
    _register_aliases(registry, build_put_spread, ["put spread", "bear spread", "put_spread"])
    _register_aliases(registry, build_butterfly, ["butterfly", "butterfly spread"])
    _register_aliases(registry, build_straddle, ["straddle"])
    _register_aliases(registry, build_barrier_option, ["barrier", "barrier option", "knock-out", "knock-in", "ko", "ki", "up-and-out", "down-and-out", "up-and-in", "down-and-in"])
    _register_aliases(registry, build_zero_coupon_bond, ["zero coupon bond", "zero-coupon bond", "zcb", "zero_coupon_bond"])
    _register_aliases(registry, build_coupon_bond, ["coupon bond", "fixed coupon bond", "coupon_bond"])
    _register_aliases(registry, build_interest_rate_swap, ["swap", "interest rate swap", "irs", "interest_rate_swap"])
    _register_aliases(registry, build_structured_note, ["structured note", "structured_note", "capital protected", "capital protected note", "capped capital protected", "capped capital protected note", "reverse convertible"])
    _register_aliases(registry, build_autocall, ["autocall", "autocallable", "autocall product"])

    return registry


def build_product_from_row(
    row: RowLike,
    *,
    registry: ProductFactoryRegistry | None = None,
    spot_reference: float | None = None,
) -> object:
    registry = registry or create_default_product_registry()
    product_key = infer_product_type_key(row)
    return registry.build(product_key, row=row, spot_reference=spot_reference)


def infer_product_type_key(row: RowLike) -> str:
    source_sheet = str(_get(row, "source_sheet", default="")).strip().lower()
    product_type = str(_get(row, "product_type", default="")).strip().lower()

    if source_sheet in {"swaps", "swap"}:
        return "interest rate swap"
    if source_sheet in {"autocalls", "autocall"}:
        return "autocall"
    if source_sheet in {"structured_notes", "structured notes", "notes_structurees"}:
        return "structured note"
    if source_sheet in {"bonds", "bond"}:
        if "coupon" in product_type:
            return "coupon bond"
        return "zero coupon bond"

    text = product_type.replace("_", " ").replace("-", " ")

    if _is_barrier_row(row):
        return "barrier option"

    if "call spread" in text:
        return "call spread"
    if "put spread" in text:
        return "put spread"
    if "butterfly" in text:
        return "butterfly"
    if "straddle" in text:
        return "straddle"
    if "reverse" in text or "convertible" in text or "capital" in text or "protected" in text or "structured" in text:
        return "structured note"
    if "autocall" in text:
        return "autocall"
    if "swap" in text or "irs" in text:
        return "interest rate swap"
    if "coupon" in text and "bond" in text:
        return "coupon bond"
    if "zero" in text or "zcb" in text:
        return "zero coupon bond"
    if "put" in text:
        return "put"
    if "call" in text:
        return "call"

    raise KeyError(
        f"Cannot infer product builder from row product_type={product_type!r}, "
        f"source_sheet={source_sheet!r}."
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _register_aliases(registry: ProductFactoryRegistry, builder: Builder, aliases: Iterable[str]) -> None:
    for alias in aliases:
        registry.register(alias, builder)


def _as_dict(row: RowLike) -> dict[str, Any]:
    if isinstance(row, pd.Series):
        return row.to_dict()
    return dict(row)


def _get(row: RowLike, *keys: str, default: Any = None) -> Any:
    data = _as_dict(row)
    for key in keys:
        if key in data and _has_value(data[key]):
            return data[key]
    return default


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    try:
        return not bool(pd.isna(value))
    except (TypeError, ValueError):
        return True


def _optional_float(value: Any) -> float | None:
    if not _has_value(value):
        return None
    return float(value)


def _float_required(value: Any, field_name: str) -> float:
    if not _has_value(value):
        raise ValueError(f"{field_name} is required.")
    return float(value)


def _product_id(row: RowLike, *, prefix: str) -> str:
    value = _get(row, "product_id", "id_produit", default=None)
    if _has_value(value):
        return str(value)
    source_row = _get(row, "source_row", default="X")
    return f"{prefix}-{source_row}"


def _maturity(row: RowLike) -> float:
    direct = _get(row, "time_to_maturity_years", "maturity", default=None)
    if _has_value(direct):
        return max(float(direct), 0.0)

    valuation_date = _get(row, "valuation_date", "date_valorisation", default=None)
    maturity_date = _get(row, "maturity_date", "maturite", default=None)
    if _has_value(valuation_date) and _has_value(maturity_date):
        return max(float((pd.Timestamp(maturity_date) - pd.Timestamp(valuation_date)).days / 365.25), 0.0)

    raise ValueError("maturity requires time_to_maturity_years or valuation_date + maturity_date.")


def _notional(row: RowLike, *, default: float) -> float:
    return max(float(_get(row, "notional", "quantity", "nominal", "quantite", default=default)), 1e-12)


def _currency(row: RowLike) -> str:
    return str(_get(row, "currency", "rate_currency", "devise", default="EUR")).strip().upper()


def _underlying(row: RowLike) -> str:
    return str(_get(row, "underlying", "sous_jacent", "ticker", default="")).strip().upper()


def _option_type(row: RowLike) -> str:
    raw = str(_get(row, "option_type", default="")).strip().lower()
    product_type = str(_get(row, "product_type", default="")).strip().lower()
    text = f"{raw} {product_type}"
    if "put" in text or raw == "p":
        return "put"
    if "call" in text or raw == "c":
        return "call"
    raise ValueError("Cannot infer option_type. Provide option_type or product_type containing call/put.")


def _strike(row: RowLike) -> float:
    return _float_required(_get(row, "strike", "strike_1", default=None), "strike")


def _two_strikes(row: RowLike) -> tuple[float, float]:
    k1 = _float_required(_get(row, "strike_low", "strike_1", "strike", default=None), "strike_1")
    k2 = _float_required(_get(row, "strike_high", "strike_2", default=None), "strike_2")
    return k1, k2


def _barrier_kind(product_type: str, barrier_type: Any) -> str:
    text = f"{product_type} {barrier_type}".lower().replace("_", "-")
    if "out" in text or "ko" in text or "knock-out" in text:
        return "KO"
    if "in" in text or "ki" in text or "knock-in" in text:
        return "KI"
    return "KO"


def _barrier_direction(product_type: str, barrier_type: Any) -> str | None:
    text = f"{product_type} {barrier_type}".lower().replace("_", "-")
    if "up" in text:
        return "up"
    if "down" in text:
        return "down"
    return None

def _is_barrier_row(row: RowLike) -> bool:
    product_type = str(_get(row, "product_type", default="")).strip().lower()
    barrier_type = str(_get(row, "barrier_type", default="")).strip().lower()
    text = f"{product_type} {barrier_type}".replace("_", "-")

    has_barrier_level = _has_value(_get(row, "barrier_level", default=None))
    has_barrier_column = _has_value(_get(row, "barrier", default=None))
    has_barrier_1 = _has_value(_get(row, "barrier_1", default=None))

    keywords = (
        "barrier",
        "barrière",
        "down-and-out",
        "up-and-out",
        "down-and-in",
        "up-and-in",
        "down out",
        "up out",
        "down in",
        "up in",
        "knock-out",
        "knock-in",
        "knockout",
        "knockin",
        "ko",
        "ki",
    )

    return (
        has_barrier_level
        or has_barrier_column
        or has_barrier_1
        or any(keyword in text for keyword in keywords)
    )


def _resolve_spot_reference(row: RowLike, explicit: float | None) -> float:
    if explicit is not None:
        return float(explicit)
    return float(_get(row, "spot_reference", "initial_spot", "underlying_price", "spot", default=100.0))


def _list_value(value: Any) -> list[Any]:
    if not _has_value(value):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, str) and "," in value:
        return [item.strip() for item in value.split(",") if item.strip()]
    return [value]


__all__ = [
    "build_autocall",
    "build_autocalls_from_frame",
    "build_barrier_option",
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
