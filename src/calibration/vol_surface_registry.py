"""Volatility-surface registry by underlying and valuation date.

This module prevents the main modelling mistake caught in the audit:
building one volatility surface from several underlyings.

One surface = one (underlying, valuation_date).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd

from src.calibration.implied_vol import (
    ImpliedVolSurface,
    calibrate_implied_vol_panel,
    normalize_option_surface_quotes,
)
from src.calibration.svi import SSVIVolSurface, SVIVolSurface


SurfaceModelName = Literal["interpolated", "svi", "ssvi"]


@dataclass(frozen=True, slots=True)
class VolSurfaceKey:
    underlying: str
    valuation_date: pd.Timestamp

    def __post_init__(self) -> None:
        object.__setattr__(self, "underlying", str(self.underlying).strip().upper())
        object.__setattr__(self, "valuation_date", pd.Timestamp(self.valuation_date).normalize())

    @property
    def label(self) -> str:
        return f"{self.underlying}|{self.valuation_date.date().isoformat()}"


@dataclass(slots=True)
class VolSurfaceRecord:
    key: VolSurfaceKey
    quotes: pd.DataFrame
    surfaces: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)

    @property
    def quote_count(self) -> int:
        return int(len(self.quotes))

    @property
    def available_models(self) -> tuple[str, ...]:
        return tuple(self.surfaces.keys())

    def get(
        self,
        preferred_models: Iterable[str] = ("ssvi", "svi", "interpolated"),
    ) -> Any:
        for model_name in preferred_models:
            if model_name in self.surfaces:
                return self.surfaces[model_name]
        raise KeyError(
            f"No requested surface model available for {self.key.label}. "
            f"Available={list(self.surfaces)}."
        )


@dataclass(slots=True)
class VolSurfaceRegistry:
    """Container for volatility surfaces indexed by (underlying, valuation_date)."""

    records: dict[VolSurfaceKey, VolSurfaceRecord] = field(default_factory=dict)
    default_underlying: str | None = None
    preferred_models: tuple[str, ...] = ("ssvi", "svi", "interpolated")

    @classmethod
    def from_option_quotes(
        cls,
        option_quotes: pd.DataFrame,
        *,
        rate: float = 0.0,
        dividend_yield: float = 0.0,
        rate_column: str | None = None,
        dividend_yield_column: str | None = None,
        min_quotes_per_surface: int = 8,
        fit_interpolated: bool = True,
        fit_svi: bool = True,
        fit_ssvi: bool = True,
        default_underlying: str | None = None,
        preferred_underlyings: tuple[str, ...] = ("MSFT", "AAPL"),
        preferred_models: tuple[str, ...] = ("ssvi", "svi", "interpolated"),
    ) -> "VolSurfaceRegistry":
        """Build all surfaces, one per (underlying, valuation_date)."""
        canonical = _prepare_surface_quotes(
            option_quotes,
            rate=rate,
            dividend_yield=dividend_yield,
            rate_column=rate_column,
            dividend_yield_column=dividend_yield_column,
        )

        if canonical.empty:
            return cls(records={}, default_underlying=None, preferred_models=preferred_models)

        selected_default = (
            str(default_underlying).strip().upper()
            if default_underlying is not None
            else choose_default_underlying(
                canonical,
                preferred=preferred_underlyings,
                min_quotes=min_quotes_per_surface,
            )
        )

        records: dict[VolSurfaceKey, VolSurfaceRecord] = {}

        for (underlying, valuation_date), group in canonical.groupby(
            ["underlying", "valuation_date"],
            dropna=False,
            sort=True,
        ):
            group = group.copy().reset_index(drop=True)
            key = VolSurfaceKey(str(underlying), pd.Timestamp(valuation_date))

            record = VolSurfaceRecord(key=key, quotes=group)

            if len(group) < min_quotes_per_surface:
                record.errors["surface"] = (
                    f"Not enough quotes: {len(group)} < {min_quotes_per_surface}."
                )
                records[key] = record
                continue

            model_quotes = _to_model_quote_columns(group)

            if fit_interpolated:
                try:
                    record.surfaces["interpolated"] = ImpliedVolSurface.from_quotes(
                        model_quotes,
                        maturity_column="time_to_maturity_years",
                        log_moneyness_column="log_moneyness",
                        iv_column="implied_vol",
                        require_single_underlying=True,
                        require_single_valuation_date=True,
                    )
                    record.diagnostics["interpolated_quote_count"] = float(len(model_quotes))
                except Exception as exc:
                    record.errors["interpolated"] = str(exc)

            if fit_svi:
                try:
                    svi = SVIVolSurface.fit_from_quotes(model_quotes)
                    record.surfaces["svi"] = svi
                    record.diagnostics.update(_prefixed_diagnostics("svi", svi.diagnostics()))
                except Exception as exc:
                    record.errors["svi"] = str(exc)

            if fit_ssvi:
                try:
                    ssvi = SSVIVolSurface.fit_from_quotes(model_quotes)
                    record.surfaces["ssvi"] = ssvi
                    record.diagnostics.update(_prefixed_diagnostics("ssvi", ssvi.diagnostics()))
                except Exception as exc:
                    record.errors["ssvi"] = str(exc)

            records[key] = record

        return cls(
            records=records,
            default_underlying=selected_default,
            preferred_models=preferred_models,
        )

    def get(
        self,
        underlying: str | None = None,
        date: str | pd.Timestamp | None = None,
        *,
        model: str | None = None,
    ) -> Any:
        """Return the selected surface.

        If date is None, the latest available date for the underlying is used.
        If model is None, the registry tries preferred_models in order.
        """
        key = self.resolve_key(underlying=underlying, date=date)
        record = self.records[key]

        preferred = (model,) if model is not None else self.preferred_models
        return record.get(preferred)

    def get_record(
        self,
        underlying: str | None = None,
        date: str | pd.Timestamp | None = None,
    ) -> VolSurfaceRecord:
        return self.records[self.resolve_key(underlying=underlying, date=date)]

    def resolve_key(
        self,
        underlying: str | None = None,
        date: str | pd.Timestamp | None = None,
    ) -> VolSurfaceKey:
        if not self.records:
            raise KeyError("VolSurfaceRegistry is empty.")

        selected_underlying = (
            str(underlying).strip().upper()
            if underlying is not None
            else self.default_underlying
        )

        if not selected_underlying:
            selected_underlying = sorted({key.underlying for key in self.records})[0]

        dates = sorted(
            key.valuation_date
            for key in self.records
            if key.underlying == selected_underlying
        )

        if not dates:
            raise KeyError(f"No volatility surface for underlying={selected_underlying!r}.")

        if date is None:
            selected_date = dates[-1]
        else:
            requested = pd.Timestamp(date).normalize()
            if requested in dates:
                selected_date = requested
            else:
                # Nearest date fallback for dashboard convenience.
                selected_date = min(dates, key=lambda item: abs((item - requested).days))

        key = VolSurfaceKey(selected_underlying, selected_date)
        if key not in self.records:
            raise KeyError(f"No volatility surface for key={key.label}.")

        return key

    def available_underlyings(self) -> list[str]:
        return sorted({key.underlying for key in self.records})

    def available_dates(self, underlying: str | None = None) -> list[pd.Timestamp]:
        if underlying is None:
            return sorted({key.valuation_date for key in self.records})

        selected = str(underlying).strip().upper()
        return sorted(
            key.valuation_date
            for key in self.records
            if key.underlying == selected
        )

    def summary(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []

        for key, record in sorted(
            self.records.items(),
            key=lambda item: (item[0].underlying, item[0].valuation_date),
        ):
            row: dict[str, Any] = {
                "underlying": key.underlying,
                "valuation_date": key.valuation_date,
                "quote_count": record.quote_count,
                "available_models": ",".join(record.available_models),
                "has_interpolated": "interpolated" in record.surfaces,
                "has_svi": "svi" in record.surfaces,
                "has_ssvi": "ssvi" in record.surfaces,
                "errors": " | ".join(f"{k}: {v}" for k, v in record.errors.items()),
            }
            row.update(record.diagnostics)
            rows.append(row)

        return pd.DataFrame(rows)


def choose_default_underlying(
    option_quotes: pd.DataFrame,
    *,
    preferred: tuple[str, ...] = ("MSFT", "AAPL"),
    min_quotes: int = 12,
    min_maturities: int = 2,
) -> str | None:
    """Choose a stable default underlying for QA/dashboard.

    Preference order:
    1. preferred tickers with enough quotes and maturity diversity;
    2. largest panel by quote count and maturity diversity;
    3. first available underlying.
    """
    if option_quotes is None or option_quotes.empty or "underlying" not in option_quotes.columns:
        return None

    data = normalize_option_surface_quotes(option_quotes) if "iv" not in option_quotes.columns else option_quotes.copy()
    if data.empty:
        return None

    stats = (
        data.groupby("underlying")
        .agg(
            quote_count=("iv", "size"),
            maturity_count=("maturity", "nunique"),
            date_count=("valuation_date", "nunique"),
        )
        .reset_index()
    )

    stats["is_usable"] = (
        (stats["quote_count"] >= int(min_quotes))
        & (stats["maturity_count"] >= int(min_maturities))
    )

    usable = stats[stats["is_usable"]].copy()

    for ticker in preferred:
        ticker = ticker.strip().upper()
        if ticker in usable["underlying"].tolist():
            return ticker

    if not usable.empty:
        usable = usable.sort_values(
            ["quote_count", "maturity_count", "date_count"],
            ascending=False,
            ignore_index=True,
        )
        return str(usable.loc[0, "underlying"])

    stats = stats.sort_values(
        ["quote_count", "maturity_count", "date_count"],
        ascending=False,
        ignore_index=True,
    )
    return str(stats.loc[0, "underlying"]) if not stats.empty else None


def _prepare_surface_quotes(
    option_quotes: pd.DataFrame,
    *,
    rate: float,
    dividend_yield: float,
    rate_column: str | None,
    dividend_yield_column: str | None,
) -> pd.DataFrame:
    """Return canonical quotes with underlying/date preserved."""
    if option_quotes is None or option_quotes.empty:
        return pd.DataFrame()

    raw = option_quotes.copy()

    has_iv = any(column in raw.columns for column in ("iv", "implied_vol", "implied_volatility", "volatility"))

    if has_iv:
        canonical = normalize_option_surface_quotes(raw)
    else:
        # calibrate_implied_vol_panel expects time_to_maturity_years.
        if "time_to_maturity_years" not in raw.columns and "maturity" in raw.columns:
            raw["time_to_maturity_years"] = raw["maturity"]

        calibrated, _ = calibrate_implied_vol_panel(
            raw,
            rate=rate,
            dividend_yield=dividend_yield,
            rate_column=rate_column,
            dividend_yield_column=dividend_yield_column,
        )
        canonical = normalize_option_surface_quotes(calibrated)

    if canonical.empty:
        return canonical

    # Absolute safety: never let one group contain mixed underlyings/dates.
    canonical["underlying"] = canonical["underlying"].astype("string").str.strip().str.upper()
    canonical["valuation_date"] = pd.to_datetime(canonical["valuation_date"]).dt.normalize()

    return canonical.reset_index(drop=True)


def _to_model_quote_columns(group: pd.DataFrame) -> pd.DataFrame:
    out = group.copy()
    out["time_to_maturity_years"] = out["maturity"]
    out["implied_vol"] = out["iv"]
    return out


def _prefixed_diagnostics(prefix: str, diagnostics: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in diagnostics.items()}


__all__ = [
    "VolSurfaceKey",
    "VolSurfaceRecord",
    "VolSurfaceRegistry",
    "choose_default_underlying",
]