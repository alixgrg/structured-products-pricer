"""Portfolio pricing engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.factory.builders import build_product_from_row
from src.factory.pricing_router import PricingRouter
from src.market.market_data import MarketData
from src.rates.yield_curve import YieldCurve
from src.risk.numerical_greeks import NumericalGreeksEngine


@dataclass(slots=True)
class PortfolioPricingConfig:
    default_spot: float = 100.0
    default_rate: float = 0.03
    default_volatility: float = 0.20
    dividend_yield: float = 0.0
    default_currency: str = "EUR"
    default_equity_currency: str = "USD"
    spot_by_underlying: dict[str, float] = field(default_factory=dict)
    volatility_by_underlying: dict[str, float] = field(default_factory=dict)
    n_paths: int = 20_000
    n_steps: int = 252
    seed: int | None = 42


class PortfolioPricingEngine:
    """Build products, route them to models, and return a pricing dataframe."""

    def __init__(
        self,
        config: PortfolioPricingConfig | None = None,
        *,
        router: PricingRouter | None = None,
        yield_curve: YieldCurve | None = None,
        vol_surface: Any | None = None,
        numerical_greeks_engine: NumericalGreeksEngine | None = None,
        use_numerical_greeks: bool = False,
        numerical_greeks_for: tuple[str, ...] = ("BarrierOption", "AutocallProduct"),
        force_numerical_greeks: bool = False,
    ) -> None:
        self.config = config or PortfolioPricingConfig()
        self.yield_curve = yield_curve
        self.vol_surface = vol_surface
        self.numerical_greeks_engine = numerical_greeks_engine
        self.use_numerical_greeks = bool(use_numerical_greeks)
        self.numerical_greeks_for = tuple(numerical_greeks_for)
        self.force_numerical_greeks = bool(force_numerical_greeks)

        self.router = router or PricingRouter.with_defaults(
            rate=self.config.default_rate,
            volatility=self.config.default_volatility,
            dividend_yield=self.config.dividend_yield,
            yield_curve=yield_curve,
            vol_surface=vol_surface,
            n_paths=self.config.n_paths,
            n_steps=self.config.n_steps,
            seed=self.config.seed,
        )

    def price_portfolio(self, inventory: dict[str, pd.DataFrame] | pd.DataFrame) -> pd.DataFrame:
        if isinstance(inventory, dict):
            from src.portfolio.inventory_loader import build_pricing_inventory
            frame = build_pricing_inventory(inventory)
        else:
            frame = inventory.copy()

        return pd.DataFrame([self._price_row(int(i), row) for i, row in frame.iterrows()])

    def _price_row(self, line_index: int, row: pd.Series) -> dict[str, Any]:
        base = self._base_output(line_index, row)
        product = None
        model = None
        market_data = None

        try:
            if str(row.get("pricing_status_hint", "ok")) not in {"ok", "", "nan", "<NA>"}:
                base["pricing_status_hint"] = str(row.get("pricing_status_hint"))

            product = build_product_from_row(row, spot_reference=self._spot_reference(row))
            market_data = self._market_data(row)
            model = self.router.model_for(product)

            metrics = self.router.price_and_risk(product, market_data)
            metrics = self._maybe_apply_numerical_greeks(product, model, market_data, metrics)

            sign = self._position_sign(row)
            signed = self._signed_metrics(metrics, sign)

            maturity = float(getattr(product, "maturity", np.nan))
            strike = self._extract_strike(product)

            currency = self._product_currency(product, row)
            risk_currency = self._risk_currency(product, row, currency)
            risk_underlying = self._risk_underlying(product, row, risk_currency)

            display_underlying = self._clean_upper_label(row.get("underlying"))
            if self._is_rate_product(product) or not display_underlying:
                display_underlying = risk_underlying

            return {
                **base,
                "portfolio": self._portfolio(row),
                "currency": currency,
                "risk_currency": risk_currency,
                "underlying": display_underlying,
                "risk_underlying": risk_underlying,
                "product_id": getattr(product, "product_id", base["product_id"]),
                "product_type": str(row.get("product_type", type(product).__name__)),
                "product_class": type(product).__name__,
                "model_name": type(model).__name__,
                "status": "priced",
                **signed,
                "maturity_years": maturity,
                "strike": strike,
                "maturity_bucket": maturity_bucket(maturity),
                "strike_bucket": strike_bucket(strike, market_data.spot),
                "spot_used": market_data.spot,
                "rate_used": market_data.rate,
                "volatility_used": market_data.volatility,
                "position_sign": sign,
                "numerical_greeks_used": float(metrics.get("numerical_greeks_used", 0.0)),
                "numerical_greeks_error": str(metrics.get("numerical_greeks_error", "")),
                "error_message": "",
            }

        except Exception as exc:
            maturity = float(getattr(product, "maturity", np.nan)) if product is not None else np.nan
            strike = self._extract_strike(product) if product is not None else np.nan
            spot_used = getattr(market_data, "spot", np.nan) if market_data is not None else np.nan
            rate_used = getattr(market_data, "rate", np.nan) if market_data is not None else np.nan
            vol_used = getattr(market_data, "volatility", np.nan) if market_data is not None else np.nan

            currency = self._product_currency(product, row) if product is not None else base["currency"]
            risk_currency = self._risk_currency(product, row, currency)
            risk_underlying = self._risk_underlying(product, row, risk_currency)
            display_underlying = self._clean_upper_label(row.get("underlying")) or risk_underlying

            return {
                **base,
                "portfolio": self._portfolio(row),
                "currency": currency,
                "risk_currency": risk_currency,
                "underlying": display_underlying,
                "risk_underlying": risk_underlying,
                "product_id": getattr(product, "product_id", base["product_id"]) if product is not None else base["product_id"],
                "product_class": type(product).__name__ if product is not None else None,
                "model_name": type(model).__name__ if model is not None else None,
                "status": "error",
                "price": np.nan,
                "delta": 0.0,
                "gamma": 0.0,
                "vega": 0.0,
                "theta": 0.0,
                "rho": 0.0,
                "dv01": 0.0,
                "standard_error": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "maturity_years": maturity,
                "strike": strike,
                "maturity_bucket": maturity_bucket(maturity),
                "strike_bucket": strike_bucket(strike, spot_used),
                "spot_used": spot_used,
                "rate_used": rate_used,
                "volatility_used": vol_used,
                "position_sign": self._position_sign(row),
                "quantity": self._first_number(row, ["quantity"], default=np.nan),
                "notional": self._first_number(row, ["notional"], default=np.nan),
                "position_size": self._first_number(row, ["position_size"], default=np.nan),
                "booking_notional": self._first_number(row, ["booking_notional"], default=np.nan),
                "contract_multiplier": self._first_number(row, ["contract_multiplier"], default=1.0),
                "price_unit": str(row.get("price_unit", "amount")),
                "numerical_greeks_used": 0.0,
                "numerical_greeks_error": "",
                "error_message": str(exc),
            }

    def _maybe_apply_numerical_greeks(
        self,
        product: object,
        model: object,
        market_data: MarketData,
        metrics: dict[str, Any],
    ) -> dict[str, Any]:
        if not self.use_numerical_greeks:
            return metrics

        if type(product).__name__ not in self.numerical_greeks_for:
            return metrics

        engine = self.numerical_greeks_engine or NumericalGreeksEngine()

        try:
            enriched = engine.enrich_metrics(
                product,
                model,
                market_data,
                metrics,
                force=self.force_numerical_greeks,
            )
        except Exception as exc:
            # Critical: numerical Greeks should not make a priceable product fail.
            enriched = dict(metrics)
            enriched["numerical_greeks_used"] = 0.0
            enriched["numerical_greeks_error"] = str(exc)

        for key in ("standard_error", "ci_low", "ci_high", "n_paths", "confidence_level"):
            if key in metrics and key not in enriched:
                enriched[key] = metrics[key]

        return enriched

    def _base_output(self, line_index: int, row: pd.Series) -> dict[str, Any]:
        portfolio = self._portfolio(row)
        currency = self._row_currency(row)
        underlying = self._clean_upper_label(row.get("underlying"))
        source_sheet = str(row.get("source_sheet", "")).strip().lower()

        if underlying:
            risk_underlying = underlying
        elif source_sheet in {"swaps", "swap", "bonds", "bond", "rates"}:
            risk_underlying = self._rate_curve_label(currency)
        else:
            risk_underlying = "UNKNOWN_UNDERLYING"

        return {
            "line_index": line_index,
            "source_sheet": str(row.get("source_sheet", "")),
            "source_row": row.get("source_row", line_index),
            "portfolio": portfolio,
            "currency": currency,
            "risk_currency": currency,
            "underlying": underlying or risk_underlying,
            "risk_underlying": risk_underlying,
            "product_id": str(row.get("product_id", f"LINE-{line_index}")),
            "product_type": str(row.get("product_type", "")),
            "pricing_status_hint": str(row.get("pricing_status_hint", "ok")),
        }

    def _market_data(self, row: pd.Series) -> MarketData:
        underlying = self._clean_upper_label(row.get("underlying"))

        spot = self._first_number(
            row,
            ["spot", "underlying_price"],
            default=self.config.spot_by_underlying.get(underlying, self.config.default_spot),
        )
        rate = self._first_number(row, ["rate", "zero_rate"], default=self.config.default_rate)
        vol = self._first_number(
            row,
            ["volatility", "implied_vol"],
            default=self.config.volatility_by_underlying.get(underlying, self.config.default_volatility),
        )
        dividend = self._first_number(row, ["dividend_yield", "q"], default=self.config.dividend_yield)

        return MarketData(float(spot), float(rate), float(vol), float(dividend))

    def _spot_reference(self, row: pd.Series) -> float:
        underlying = self._clean_upper_label(row.get("underlying"))
        return self._first_number(
            row,
            ["spot_reference", "initial_spot", "spot", "underlying_price"],
            default=self.config.spot_by_underlying.get(underlying, self.config.default_spot),
        )
    
    def _portfolio(self, row: pd.Series) -> str:
        value = row.get("portfolio", "default")
        cleaned = self._clean_label(value)
        return cleaned or "default"

    def _row_currency(self, row: pd.Series) -> str:
        for column in ("currency", "rate_currency", "devise"):
            if column in row.index:
                cleaned = self._clean_upper_label(row.get(column))
                if cleaned:
                    return cleaned

        source_sheet = str(row.get("source_sheet", "")).strip().lower()
        if source_sheet in {"options", "option", "autocalls", "autocall"}:
            return str(self.config.default_equity_currency).upper()

        return str(self.config.default_currency).upper()

    def _product_currency(self, product: object | None, row: pd.Series) -> str:
        product_currency = self._clean_upper_label(getattr(product, "currency", None))
        if product_currency:
            return product_currency
        return self._row_currency(row)

    def _risk_currency(self, product: object | None, row: pd.Series, currency: str) -> str:
        risk_currency = self._clean_upper_label(row.get("risk_currency"))
        if risk_currency:
            return risk_currency
        return self._clean_upper_label(currency) or self._row_currency(row)

    def _risk_underlying(self, product: object | None, row: pd.Series, risk_currency: str) -> str:
        if self._is_rate_product(product):
            return self._rate_curve_label(risk_currency)

        underlying = self._clean_upper_label(getattr(product, "underlying", None))
        if not underlying:
            underlying = self._clean_upper_label(row.get("underlying"))

        if underlying:
            return underlying

        source_sheet = str(row.get("source_sheet", "")).strip().lower()
        if source_sheet in {"swaps", "swap", "bonds", "bond", "rates"}:
            return self._rate_curve_label(risk_currency)

        return "UNKNOWN_UNDERLYING"

    @staticmethod
    def _rate_curve_label(currency: str) -> str:
        cleaned = str(currency or "EUR").strip().upper()
        return f"{cleaned}_RATE_CURVE"

    @staticmethod
    def _is_rate_product(product: object | None) -> bool:
        if product is None:
            return False
        return type(product).__name__ in {
            "ZeroCouponBond",
            "CouponBond",
            "InterestRateSwap",
            "BasisSwap",
        }

    @staticmethod
    def _clean_label(value: Any) -> str:
        if value is None:
            return ""
        try:
            if pd.isna(value):
                return ""
        except Exception:
            pass

        text = str(value).strip()
        if text.lower() in {"", "nan", "none", "<na>", "nat"}:
            return ""
        return text

    @classmethod
    def _clean_upper_label(cls, value: Any) -> str:
        return cls._clean_label(value).upper()

    @staticmethod
    def _first_number(row: pd.Series, columns: list[str], *, default: float) -> float:
        for column in columns:
            if column in row.index:
                value = row.get(column)
                try:
                    if pd.notna(value):
                        return float(value)
                except (TypeError, ValueError):
                    pass
        return float(default)

    @staticmethod
    def _position_sign(row: pd.Series) -> float:
        if "position_sign" in row.index and pd.notna(row.get("position_sign")):
            return -1.0 if float(row.get("position_sign")) < 0.0 else 1.0
        for column in ("quantity", "notional"):
            if column in row.index and pd.notna(row.get(column)):
                return -1.0 if float(row.get(column)) < 0.0 else 1.0
        return 1.0

    @staticmethod
    def _signed_metrics(metrics: dict[str, Any], sign: float) -> dict[str, float]:
        signed_keys = ["price", "delta", "gamma", "vega", "theta", "rho", "dv01"]
        out = {key: sign * float(metrics.get(key, 0.0)) for key in signed_keys}

        out["standard_error"] = float(metrics.get("standard_error", np.nan)) if "standard_error" in metrics else np.nan
        if "ci_low" in metrics and "ci_high" in metrics:
            low = sign * float(metrics["ci_low"])
            high = sign * float(metrics["ci_high"])
            out["ci_low"] = min(low, high)
            out["ci_high"] = max(low, high)
        else:
            out["ci_low"] = np.nan
            out["ci_high"] = np.nan

        return out

    @staticmethod
    def _extract_strike(product: object | None) -> float:
        if product is None:
            return np.nan

        strike = getattr(product, "strike", np.nan)
        try:
            if pd.notna(strike):
                return float(strike)
        except Exception:
            pass

        if hasattr(product, "get_legs"):
            try:
                strikes = [float(option.strike) for option, _ in product.get_legs()]
                return float(np.mean(strikes)) if strikes else np.nan
            except Exception:
                return np.nan

        if hasattr(product, "spot_reference"):
            return float(getattr(product, "spot_reference"))

        return np.nan


def maturity_bucket(maturity: float) -> str:
    if not np.isfinite(maturity):
        return "NA"
    if maturity <= 0.5:
        return "0-6M"
    if maturity <= 1.0:
        return "6M-1Y"
    if maturity <= 2.0:
        return "1Y-2Y"
    if maturity <= 5.0:
        return "2Y-5Y"
    if maturity <= 10.0:
        return "5Y-10Y"
    return "10Y+"


def strike_bucket(strike: float, spot: float | None) -> str:
    if not np.isfinite(strike) or spot is None or not np.isfinite(spot) or spot <= 0.0:
        return "NA"
    m = strike / spot
    if m < 0.80:
        return "deep_low"
    if m < 0.95:
        return "low"
    if m <= 1.05:
        return "atm"
    if m <= 1.20:
        return "high"
    return "deep_high"


__all__ = ["PortfolioPricingConfig", "PortfolioPricingEngine", "maturity_bucket", "strike_bucket"]
