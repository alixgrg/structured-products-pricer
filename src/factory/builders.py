"""Product builders from normalized inventory rows."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.factory.registry import ProductFactoryRegistry
from src.products.autocall import AutocallProduct
from src.products.barrier_option import BarrierOption
from src.products.basis_swap import BasisSwap
from src.products.coupon_bond import CouponBond
from src.products.option_strategies import Butterfly, CallSpread, PutSpread, Straddle
from src.products.structured_notes import build_structured_note_from_inventory_row
from src.products.swap import InterestRateSwap
from src.products.vanilla_option import VanillaOption
from src.products.zero_coupon_bond import ZeroCouponBond

RowLike = pd.Series | dict[str, Any]
Builder = Callable[..., object]


def build_vanilla_option(row: RowLike, **_: Any) -> VanillaOption:
    """Build a plain vanilla option from an inventory row."""
    option_type = _option_type(row)
    return VanillaOption(
        product_id=_product_id(row, prefix=f"{option_type.upper()}"),
        option_type=option_type,
        strike=_strike(row),
        maturity=_maturity(row),
        notional=_positive_notional(row, default=1.0),
        underlying=_underlying(row),
        currency=_currency(row),
        dividend_yield=_optional_float(_get(row, "dividend_yield", "q", default=None)),
    )


def build_call_spread(row: RowLike, **_: Any) -> CallSpread:
    """Build a call spread from an inventory row."""
    k1, k2 = _two_strikes(row)
    return CallSpread(
        product_id=_product_id(row, prefix="CS"),
        maturity=_maturity(row),
        strike_low=min(k1, k2),
        strike_high=max(k1, k2),
        underlying=_underlying(row),
        notional=_positive_notional(row, default=1.0),
        currency=_currency(row),
    )


def build_put_spread(row: RowLike, **_: Any) -> PutSpread:
    """Build a put spread from an inventory row."""
    k1, k2 = _two_strikes(row)
    return PutSpread(
        product_id=_product_id(row, prefix="PS"),
        maturity=_maturity(row),
        strike_low=min(k1, k2),
        strike_high=max(k1, k2),
        underlying=_underlying(row),
        notional=_positive_notional(row, default=1.0),
        currency=_currency(row),
    )


def build_butterfly(row: RowLike, **_: Any) -> Butterfly:
    """Build a butterfly strategy from an inventory row."""
    strikes = sorted(
        [
            _float_required(_get(row, "strike_1", default=None), "strike_1"),
            _float_required(_get(row, "strike_2", default=None), "strike_2"),
            _float_required(_get(row, "strike_3", default=None), "strike_3"),
        ]
    )
    return Butterfly(
        product_id=_product_id(row, prefix="BF"),
        maturity=_maturity(row),
        strike_low=strikes[0],
        strike_mid=strikes[1],
        strike_high=strikes[2],
        underlying=_underlying(row),
        notional=_positive_notional(row, default=1.0),
        currency=_currency(row),
    )


def build_straddle(row: RowLike, **_: Any) -> Straddle:
    """Build a straddle from an inventory row."""
    return Straddle(
        product_id=_product_id(row, prefix="STD"),
        maturity=_maturity(row),
        strike=_strike(row),
        underlying=_underlying(row),
        notional=_positive_notional(row, default=1.0),
        currency=_currency(row),
    )


def build_barrier_option(row: RowLike, **_: Any) -> BarrierOption:
    """Build a barrier option from an inventory row."""
    product_type = str(_get(row, "product_type", default="")).strip().lower()
    barrier_type_raw = str(_get(row, "barrier_type", default="")).strip().lower()

    option_type = _option_type(row)
    strike = _float_required(_get(row, "strike", "strike_1", default=None), "strike")
    barrier = _float_required(_get(row, "barrier_level", "barrier", "barrier_1", default=None), "barrier")

    direction = _barrier_direction(product_type, barrier_type_raw)
    if direction is None:
        direction = "down" if float(barrier) < float(strike) else "up"

    return BarrierOption(
        product_id=_product_id(row, prefix=f"{option_type.upper()}-BARRIER"),
        option_type=option_type,
        strike=float(strike),
        maturity=_maturity(row),
        barrier=float(barrier),
        barrier_type=_barrier_kind(product_type, barrier_type_raw),
        barrier_direction=direction,
        notional=_positive_notional(row, default=1.0),
        underlying=_underlying(row),
        currency=_currency(row),
        dividend_yield=_optional_float(_get(row, "dividend_yield", "q", default=None)),
    )


def build_zero_coupon_bond(row: RowLike, **_: Any) -> ZeroCouponBond:
    """Build a zero-coupon bond from an inventory row."""
    return ZeroCouponBond(
        product_id=_product_id(row, prefix="ZCB"),
        notional=_positive_notional(row, default=100.0),
        maturity=_maturity(row),
        currency=_currency(row),
    )


def build_coupon_bond(row: RowLike, **_: Any) -> CouponBond:
    """Build a fixed-coupon bond from an inventory row."""
    return CouponBond(
        product_id=_product_id(row, prefix="BOND"),
        notional=_positive_notional(row, default=100.0),
        maturity=_maturity(row),
        coupon_rate=_float_required(_get(row, "coupon_rate", "fixed_rate", "coupon", default=None), "coupon_rate"),
        frequency=_get(row, "frequency", "fixed_leg_frequency", default="1Y"),
        currency=_currency(row),
    )


def build_interest_rate_swap(row: RowLike, **_: Any) -> InterestRateSwap:
    """Build an interest rate swap from an inventory row."""
    float_index = _get(row, "float_index", "floating_rate_index", "floating_rate_index_1", "taux_variable_1", default="EURIBOR6M")
    return InterestRateSwap(
        product_id=_product_id(row, prefix="IRS"),
        notional=_positive_notional(row, default=1_000_000.0),
        maturity=_maturity(row),
        fixed_rate=_float_required(_get(row, "fixed_rate", "taux_fixe", default=None), "fixed_rate"),
        float_index=str(float_index),
        frequency=_get(row, "frequency", "fixed_leg_frequency", default="6M"),
        currency=_currency(row),
    )


def build_basis_swap(row: RowLike, **_: Any) -> BasisSwap:
    """Build a basis swap from an inventory row."""
    receive_index = _get(row, "receive_index", "floating_rate_index_1", "taux_variable_1", default=None)
    pay_index = _get(row, "pay_index", "floating_rate_index_2", "taux_variable_2", default=None)

    if not _has_value(receive_index) or not _has_value(pay_index):
        raise ValueError("BasisSwap requires two floating indices.")

    return BasisSwap(
        product_id=_product_id(row, prefix="BASIS"),
        notional=_positive_notional(row, default=1_000_000.0),
        maturity=_maturity(row),
        receive_index=str(receive_index),
        pay_index=str(pay_index),
        receive_frequency=_get(row, "receive_frequency", default=None),
        pay_frequency=_get(row, "pay_frequency", default=None),
        spread=float(_get(row, "spread", "basis_spread", default=0.0)),
        currency=_currency(row),
    )


def build_structured_note(row: RowLike, *, spot_reference: float | None = None, **_: Any):
    """Build a structured note by delegating to the note factory."""
    spot_ref = _resolve_spot_reference(row, spot_reference)
    valuation_date = _get(row, "valuation_date", "date_valorisation", default=None)
    return build_structured_note_from_inventory_row(
        row,
        spot_reference=spot_ref,
        valuation_date=valuation_date,
    )


def build_autocall(row: RowLike, **_: Any) -> AutocallProduct:
    """Build an autocall product from an inventory row."""
    observation_dates = _list_value(_get(row, "observation_dates", default=None))

    if not observation_dates:
        obs = _get(row, "observation_date", "date_observation", default=None)
        valuation = _get(row, "valuation_date", "date_valorisation", default=None)

        if _has_value(obs) and _has_value(valuation):
            observation_dates = [
                max(
                    (pd.Timestamp(obs) - pd.Timestamp(valuation)).days / 365.25,
                    1e-12,
                )
            ]
        elif _has_value(obs):
            observation_dates = [obs]
        else:
            observation_dates = [1.0]

    if observation_dates and not all(_is_number_like(x) for x in observation_dates):
        valuation = _get(row, "valuation_date", "date_valorisation", default=None)
        if not _has_value(valuation):
            raise ValueError("valuation_date is required to convert autocall observation dates.")
        valuation = pd.Timestamp(valuation)
        observation_dates = [
            max((pd.Timestamp(x) - valuation).days / 365.25, 1e-12)
            for x in observation_dates
        ]

    trigger_levels = _list_value(_get(row, "trigger_levels", default=None))
    if not trigger_levels:
        trigger_levels = [
            _float_required(
                _get(row, "autocall_trigger_level", "trigger_level", "niveau_de_rappel", default=None),
                "autocall_trigger_level",
            )
        ]

    maturity = _optional_float(_get(row, "time_to_maturity_years", "maturity", default=None))
    if maturity is None:
        maturity = max(float(x) for x in observation_dates)

    return AutocallProduct(
        product_id=_product_id(row, prefix="AUTO"),
        underlying=_underlying(row),
        observation_dates=[float(x) for x in observation_dates],
        trigger_levels=[float(x) for x in trigger_levels],
        coupon_rate=_float_required(_get(row, "coupon_rate", "coupon", default=None), "coupon_rate"),
        barrier_protection=_float_required(
            _get(row, "barrier_protection", "protection_barrier", "barrier_1", default=70.0),
            "barrier_protection",
        ),
        notional=_positive_notional(row, default=100.0),
        initial_spot=_resolve_spot_reference(row, None),
        currency=_currency(row),
        maturity=float(maturity),
    )



def build_autocalls_from_frame(frame: pd.DataFrame) -> list[AutocallProduct]:
    if frame.empty:
        return []

    group_key = "product_id" if "product_id" in frame.columns else None
    if group_key is None:
        return [build_autocall(row) for _, row in frame.iterrows()]

    products: list[AutocallProduct] = []

    for _, group in frame.groupby(group_key, dropna=False, sort=False):
        group = group.sort_values("observation_date").copy()
        first = group.iloc[0].copy()

        valuation_date = _get(first, "valuation_date", default=None)
        if _has_value(valuation_date) and "observation_date" in group.columns:
            valuation = pd.Timestamp(valuation_date)
            obs_dates = pd.to_datetime(group["observation_date"], errors="coerce")
            obs_times = (obs_dates - valuation).dt.days / 365.25
            obs_times = obs_times.astype(float).clip(lower=1e-12)

            first["observation_dates"] = obs_times.tolist()
            first["time_to_maturity_years"] = float(obs_times.max())
        else:
            first["observation_dates"] = group.get(
                "observation_date",
                pd.Series(range(1, len(group) + 1)),
            ).tolist()

        first["trigger_levels"] = group.get(
            "autocall_trigger_level",
            pd.Series([first.get("trigger_level", 1.0)] * len(group)),
        ).tolist()

        if "coupon_rate" in group.columns and pd.to_numeric(group["coupon_rate"], errors="coerce").notna().any():
            maturity = max(float(first["time_to_maturity_years"]), 1e-12)
            final_coupon = float(pd.to_numeric(group["coupon_rate"], errors="coerce").dropna().iloc[-1])
            first["coupon_rate"] = final_coupon / maturity

        products.append(build_autocall(first))

    return products


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
    _register_aliases(registry, build_basis_swap, ["basis swap", "basis_swap", "float float swap", "floating floating swap"])
    _register_aliases(registry, build_structured_note, ["structured note", "structured_note", "capital protected", "capital protected note", "capped capital protected", "capped capital protected note", "reverse convertible"])
    _register_aliases(registry, build_autocall, ["autocall", "autocallable", "autocall product"])

    return registry


def build_product_from_row(
    row: RowLike,
    *,
    registry: ProductFactoryRegistry | None = None,
    spot_reference: float | None = None,
) -> object:
    """Build a product from a normalized inventory row."""
    registry = registry or create_default_product_registry()
    product_key = infer_product_type_key(row)
    return registry.build(product_key, row=row, spot_reference=spot_reference)


def infer_product_type_key(row: RowLike) -> str:
    """Infer the registry key for a normalized inventory row."""
    source_sheet = str(_get(row, "source_sheet", default="")).strip().lower()
    product_type = str(_get(row, "product_type", default="")).strip().lower()
    text = product_type.replace("_", " ").replace("-", " ")

    if source_sheet in {"swaps", "swap"}:
        if "basis" in text:
            return "basis swap"
        if not _has_value(_get(row, "fixed_rate", "taux_fixe", default=None)) and _has_value(_get(row, "floating_rate_index_1", default=None)) and _has_value(_get(row, "floating_rate_index_2", default=None)):
            return "basis swap"
        return "interest rate swap"

    if source_sheet in {"autocalls", "autocall"}:
        return "autocall"
    if source_sheet in {"structured_notes", "structured notes", "notes_structurees"}:
        return "structured note"
    if source_sheet in {"bonds", "bond"}:
        if "zero" in text or "zcb" in text:
            return "zero coupon bond"
        if "coupon" in text:
            return "coupon bond"
        return "zero coupon bond"

    if _is_barrier_row(row):
        return "barrier option"
    if "basis" in text:
        return "basis swap"
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
    if "zero" in text or "zcb" in text:
        return "zero coupon bond"
    if "coupon" in text and "bond" in text:
        return "coupon bond"
    if "put" in text:
        return "put"
    if "call" in text:
        return "call"

    raise KeyError(
        f"Cannot infer product builder from row product_type={product_type!r}, source_sheet={source_sheet!r}."
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


def _is_number_like(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


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


def _positive_notional(row: RowLike, default: float = 1.0) -> float:
    value = _get(
        row,
        "booking_notional",
        "notional",
        "position_size",
        "quantity",
        default=default,
    )
    return abs(_float_required(value, "notional"))


def _position_sign(row: RowLike) -> float:
    explicit = _get(row, "position_sign", default=None)
    if _has_value(explicit):
        return -1.0 if float(explicit) < 0.0 else 1.0
    raw = _get(row, "quantity", "notional", "nominal", "quantite", default=1.0)
    return -1.0 if float(raw) < 0.0 else 1.0


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

    return (
        _has_value(_get(row, "barrier_level", default=None))
        or _has_value(_get(row, "barrier", default=None))
        or _has_value(_get(row, "barrier_1", default=None))
        or any(keyword in text for keyword in ("barrier", "barrière", "down-and-out", "up-and-out", "down-and-in", "up-and-in", "knock-out", "knock-in", "ko", "ki"))
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
    if isinstance(value, pd.Series):
        return value.tolist()
    if isinstance(value, str) and "," in value:
        return [item.strip() for item in value.split(",") if item.strip()]
    return [value]


__all__ = [
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
