"""Portfolio valuation and aggregations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.factory.structured_note_factory import StructuredNoteFactory
from src.market.market_data import MarketData
from src.models.black_scholes import BlackScholesModel
from src.models.discounting_model import DiscountingModel
from src.products.option_strategies import OptionStrategy
from src.products.structured_notes import (
    CappedCapitalProtectedNote,
    CapitalProtectedNote,
    ReverseConvertible,
)
from src.products.vanilla_option import VanillaOption
from src.products.zero_coupon_bond import ZeroCouponBond


@dataclass(slots=True)
class PortfolioSnapshot:
    """Lightweight portfolio descriptor used for smoke tests and demos."""

    name: str
    position_count: int = 0


@dataclass(frozen=True, slots=True)
class PortfolioMarketContext:
    """Market assumptions used for portfolio valuation."""

    default_spot: float = 100.0
    rate: float = 0.03
    volatility: float = 0.20
    dividend_yield: float = 0.0
    spot_by_underlying: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class PortfolioValuationResult:
    """Hold line-level valuations and precomputed aggregates."""

    line_valuations: pd.DataFrame
    by_product: pd.DataFrame
    by_underlying: pd.DataFrame
    by_maturity: pd.DataFrame
    by_strike: pd.DataFrame
    by_portfolio: pd.DataFrame


class PortfolioValuationEngine:
    """Value inventory rows and aggregate prices/greeks."""

    def __init__(
        self,
        *,
        option_model: BlackScholesModel | None = None,
        discount_model: DiscountingModel | None = None,
        note_factory: StructuredNoteFactory | None = None,
    ) -> None:
        self.option_model = option_model or BlackScholesModel()
        self.discount_model = discount_model or DiscountingModel(rate=0.03)
        self.note_factory = note_factory or StructuredNoteFactory.with_defaults()

    def value_inventory(
        self,
        inventory: pd.DataFrame,
        *,
        market: PortfolioMarketContext | None = None,
    ) -> PortfolioValuationResult:
        """Price all inventory lines and return line-level and aggregated tables."""
        context = market or PortfolioMarketContext()
        rows: list[dict[str, Any]] = []

        for index, row in inventory.iterrows():
            rows.append(self._value_row(index=index, row=row, context=context))

        line_valuations = pd.DataFrame(rows)

        by_product = _aggregate_metric_table(line_valuations, ["product_type"])
        by_underlying = _aggregate_metric_table(line_valuations, ["underlying"])
        by_maturity = _aggregate_metric_table(line_valuations, ["maturity_years"])
        by_strike = _aggregate_metric_table(line_valuations, ["strike"])
        by_portfolio = _aggregate_metric_table(line_valuations, ["portfolio"])

        return PortfolioValuationResult(
            line_valuations=line_valuations,
            by_product=by_product,
            by_underlying=by_underlying,
            by_maturity=by_maturity,
            by_strike=by_strike,
            by_portfolio=by_portfolio,
        )

    def _value_row(
        self,
        *,
        index: int,
        row: pd.Series,
        context: PortfolioMarketContext,
    ) -> dict[str, Any]:
        base = {
            "line_index": int(index),
            "portfolio": str(row.get("portfolio", "default_portfolio")),
            "product_id": str(row.get("product_id", f"LINE-{index}")),
            "source_sheet": str(row.get("source_sheet", "unknown")),
            "underlying": _to_underlying(row.get("underlying")),
            "quantity": float(_safe_numeric(row.get("quantity"), default=1.0)),
        }

        try:
            product, product_type = self._build_product_from_row(row, context=context)
            market_data = _build_market_data(row, context=context)

            price, metrics = self._price_and_risk(product, market_data)

            maturity_years = _extract_maturity(product)
            strike = _extract_strike(product)

            return {
                **base,
                "product_type": product_type,
                "status": "supported",
                "price": price,
                "delta": float(metrics.get("delta", 0.0)),
                "gamma": float(metrics.get("gamma", 0.0)),
                "vega": float(metrics.get("vega", 0.0)),
                "theta": float(metrics.get("theta", 0.0)),
                "rho": float(metrics.get("rho", 0.0)),
                "maturity_years": maturity_years,
                "strike": strike,
                "error_message": "",
            }
        except (KeyError, TypeError, ValueError) as exc:
            return {
                **base,
                "product_type": str(row.get("product_type", "unknown")).strip().lower() or "unknown",
                "status": "unsupported",
                "price": np.nan,
                "delta": 0.0,
                "gamma": 0.0,
                "vega": 0.0,
                "theta": 0.0,
                "rho": 0.0,
                "maturity_years": np.nan,
                "strike": np.nan,
                "error_message": str(exc),
            }

    def _build_product_from_row(
        self,
        row: pd.Series,
        *,
        context: PortfolioMarketContext,
    ) -> tuple[object, str]:
        source_sheet = str(row.get("source_sheet", "")).strip().lower()
        product_type_raw = str(row.get("product_type", "")).strip().lower()
        product_id = str(row.get("product_id", f"LINE-{row.name}"))

        maturity = _safe_numeric(row.get("time_to_maturity_years"), default=1.0)
        if maturity < 0.0:
            maturity = 0.0

        underlying = _to_underlying(row.get("underlying"))
        quantity = _safe_numeric(row.get("quantity"), default=1.0)

        strike_1 = _safe_numeric(row.get("strike_1"), default=np.nan)
        strike_2 = _safe_numeric(row.get("strike_2"), default=np.nan)
        strike_3 = _safe_numeric(row.get("strike_3"), default=np.nan)

        if source_sheet == "structured_notes" or any(
            key in product_type_raw for key in ("capital", "reverse", "convertible")
        ):
            note = self.note_factory.build_from_inventory_row(
                row,
                spot_reference=_resolve_spot(underlying, context),
                valuation_date=row.get("valuation_date"),
            )
            return note, type(note).__name__.lower()

        if source_sheet == "options" or any(
            key in product_type_raw for key in ("call", "put", "spread", "butterfly")
        ):
            if "call spread" in product_type_raw:
                if not np.isfinite(strike_1) or not np.isfinite(strike_2):
                    raise ValueError("call spread requires strike_1 and strike_2")
                strategy = OptionStrategy.call_spread(
                    product_id=product_id,
                    maturity=maturity,
                    strike_low=float(min(strike_1, strike_2)),
                    strike_high=float(max(strike_1, strike_2)),
                    underlying=underlying,
                    notional=max(quantity, 1e-12),
                )
                return strategy, "call_spread"

            if "put spread" in product_type_raw:
                if not np.isfinite(strike_1) or not np.isfinite(strike_2):
                    raise ValueError("put spread requires strike_1 and strike_2")
                strategy = OptionStrategy.put_spread(
                    product_id=product_id,
                    maturity=maturity,
                    strike_low=float(min(strike_1, strike_2)),
                    strike_high=float(max(strike_1, strike_2)),
                    underlying=underlying,
                    notional=max(quantity, 1e-12),
                )
                return strategy, "put_spread"

            if "butterfly" in product_type_raw:
                if not np.isfinite(strike_1) or not np.isfinite(strike_2) or not np.isfinite(strike_3):
                    raise ValueError("butterfly requires strike_1, strike_2, strike_3")
                strategy = OptionStrategy.butterfly(
                    product_id=product_id,
                    maturity=maturity,
                    strike_low=float(min(strike_1, strike_2, strike_3)),
                    strike_mid=float(sorted([strike_1, strike_2, strike_3])[1]),
                    strike_high=float(max(strike_1, strike_2, strike_3)),
                    underlying=underlying,
                    notional=max(quantity, 1e-12),
                )
                return strategy, "butterfly"

            if "put" in product_type_raw:
                strike = strike_1 if np.isfinite(strike_1) else _safe_numeric(row.get("strike"), default=np.nan)
                if not np.isfinite(strike):
                    raise ValueError("put option requires strike_1 or strike")
                product = VanillaOption(
                    product_id=product_id,
                    option_type="put",
                    strike=float(strike),
                    maturity=maturity,
                    notional=max(quantity, 1e-12),
                    underlying=underlying,
                )
                return product, "vanilla_put"

            if "call" in product_type_raw:
                strike = strike_1 if np.isfinite(strike_1) else _safe_numeric(row.get("strike"), default=np.nan)
                if not np.isfinite(strike):
                    raise ValueError("call option requires strike_1 or strike")
                product = VanillaOption(
                    product_id=product_id,
                    option_type="call",
                    strike=float(strike),
                    maturity=maturity,
                    notional=max(quantity, 1e-12),
                    underlying=underlying,
                )
                return product, "vanilla_call"

        if source_sheet == "bonds" or "zero" in product_type_raw:
            bond = ZeroCouponBond(
                product_id=product_id,
                notional=max(quantity, 1e-12),
                maturity=maturity,
                currency=str(row.get("currency", "EUR")),
            )
            return bond, "zero_coupon_bond"

        raise TypeError(f"Unsupported product row: source_sheet={source_sheet!r}, product_type={product_type_raw!r}")

    def _price_and_risk(self, product: object, market_data: MarketData) -> tuple[float, dict[str, float]]:
        if isinstance(product, VanillaOption):
            return float(self.option_model.price(product, market_data)), self.option_model.risk(product, market_data)

        if isinstance(product, ZeroCouponBond):
            return float(self.discount_model.price(product, market_data)), self.discount_model.risk(product, market_data)

        if isinstance(product, OptionStrategy):
            price = product.price(self.option_model, market_data)
            metrics = _aggregate_strategy_greeks(product, self.option_model, market_data)
            return float(price), metrics

        if isinstance(product, (CapitalProtectedNote, CappedCapitalProtectedNote, ReverseConvertible)):
            price = product.price(self.option_model, self.discount_model, market_data)
            metrics = _aggregate_structured_note_greeks(product, self.option_model, self.discount_model, market_data)
            return float(price), metrics

        raise TypeError(f"Unsupported product type: {type(product)!r}")


def _aggregate_metric_table(line_valuations: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    supported = line_valuations[line_valuations["status"] == "supported"].copy()

    if supported.empty:
        return pd.DataFrame(columns=group_columns + ["price", "delta", "gamma", "vega", "theta", "rho", "line_count"])

    aggregated = (
        supported.groupby(group_columns, dropna=False)[["price", "delta", "gamma", "vega", "theta", "rho"]]
        .sum(min_count=1)
        .reset_index()
    )

    counts = supported.groupby(group_columns, dropna=False).size().rename("line_count").reset_index()
    aggregated = aggregated.merge(counts, on=group_columns, how="left")

    return aggregated


def _aggregate_strategy_greeks(
    strategy: OptionStrategy,
    option_model: BlackScholesModel,
    market_data: MarketData,
) -> dict[str, float]:
    totals = {"price": 0.0, "delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}

    for leg in strategy.legs:
        risk = option_model.risk(leg.product, market_data)
        for key in totals:
            totals[key] += strategy.notional * leg.quantity * float(risk.get(key, 0.0))

    return totals


def _aggregate_structured_note_greeks(product, option_model, discount_model, market_data) -> dict[str, float]:
    totals = {"price": 0.0, "delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}

    for leg in product.decomposition():
        if isinstance(leg.product, VanillaOption):
            leg_risk = option_model.risk(leg.product, market_data)
        elif isinstance(leg.product, ZeroCouponBond):
            leg_risk = discount_model.risk(leg.product, market_data)
            leg_risk = {
                "price": float(leg_risk.get("price", 0.0)),
                "delta": 0.0,
                "gamma": 0.0,
                "vega": 0.0,
                "theta": 0.0,
                "rho": float(leg_risk.get("dv01", 0.0)) * 1e4,
            }
        else:
            continue

        for key in totals:
            totals[key] += float(leg.quantity) * float(leg_risk.get(key, 0.0))

    return totals


def _build_market_data(row: pd.Series, *, context: PortfolioMarketContext) -> MarketData:
    underlying = _to_underlying(row.get("underlying"))
    spot = _resolve_spot(underlying, context)

    rate = _safe_numeric(row.get("rate"), default=context.rate)
    volatility = _safe_numeric(row.get("volatility"), default=context.volatility)
    dividend_yield = _safe_numeric(row.get("dividend_yield"), default=context.dividend_yield)

    return MarketData(
        spot=spot,
        rate=rate,
        volatility=volatility,
        dividend_yield=dividend_yield,
    )


def _resolve_spot(underlying: str, context: PortfolioMarketContext) -> float:
    if underlying and underlying in context.spot_by_underlying:
        return float(context.spot_by_underlying[underlying])
    return float(context.default_spot)


def _to_underlying(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value).strip().upper()


def _safe_numeric(value: Any, *, default: float) -> float:
    if value is None:
        return float(default)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(default)
    if np.isnan(numeric):
        return float(default)
    return numeric


def _extract_maturity(product: object) -> float:
    maturity = getattr(product, "maturity", np.nan)
    try:
        return float(maturity)
    except (TypeError, ValueError):
        return float("nan")


def _extract_strike(product: object) -> float:
    if isinstance(product, VanillaOption):
        return float(product.strike)

    if isinstance(product, OptionStrategy):
        strikes = [leg.product.strike for leg in product.legs]
        return float(np.mean(strikes)) if strikes else float("nan")

    if isinstance(product, (CapitalProtectedNote, CappedCapitalProtectedNote, ReverseConvertible)):
        return float(product.spot_reference)

    return float("nan")


__all__ = [
    "PortfolioMarketContext",
    "PortfolioSnapshot",
    "PortfolioValuationEngine",
    "PortfolioValuationResult",
]
