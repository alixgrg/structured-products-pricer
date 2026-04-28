"""Portfolio risk aggregation by portfolio, currency and risk pillars."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from src.portfolio.pricing_engine import maturity_bucket, strike_bucket


DEFAULT_RISK_METRICS = ("price", "delta", "gamma", "vega", "theta", "rho", "dv01")

DEFAULT_GROUP_COLUMNS = (
    "portfolio",
    "currency",
    "risk_currency",
    "product_class",
    "underlying",
    "risk_underlying",
    "maturity_bucket",
    "strike_bucket",
)

SAFE_TOTAL_COLUMNS = ("portfolio", "currency", "risk_currency")

RATE_PRODUCT_CLASSES = {
    "ZeroCouponBond",
    "CouponBond",
    "InterestRateSwap",
    "BasisSwap",
}

RATE_SOURCE_SHEETS = {"swaps", "swap", "bonds", "bond", "rates"}


@dataclass(frozen=True, slots=True)
class RiskAggregator:
    """Aggregate priced line-level risk into reporting pillars.

    Important:
    totals are safe only inside a homogeneous risk_currency bucket.
    Do not aggregate EUR and USD together unless an explicit FX conversion layer
    has been applied before this class.
    """

    metrics: tuple[str, ...] = DEFAULT_RISK_METRICS
    group_columns: tuple[str, ...] = DEFAULT_GROUP_COLUMNS
    accepted_statuses: tuple[str, ...] = ("priced", "supported")

    def aggregate_by_pillar(self, priced_portfolio: pd.DataFrame) -> pd.DataFrame:
        """Aggregate by portfolio x currency x risk_currency x product_class x underlying x risk_underlying x maturity x strike."""
        data = self._prepare(priced_portfolio)
        if data.empty:
            return pd.DataFrame(columns=[*self.group_columns, *self.metrics, "line_count", "gross_price"])

        return self._aggregate(data, self.group_columns, include_gross_price=True)

    def aggregate_by(self, priced_portfolio: pd.DataFrame, group_by: Iterable[str]) -> pd.DataFrame:
        """Generic aggregation helper for dashboards and notebooks."""
        group_columns = tuple(group_by)
        if not group_columns:
            raise ValueError("group_by must contain at least one column.")

        data = self._prepare(priced_portfolio, required_extra_columns=group_columns)
        if data.empty:
            return pd.DataFrame(columns=[*group_columns, *self.metrics, "line_count"])

        return self._aggregate(data, group_columns, include_gross_price=False)

    def aggregate_safe_totals(
        self,
        priced_portfolio: pd.DataFrame,
        *,
        group_by: Iterable[str] = SAFE_TOTAL_COLUMNS,
    ) -> pd.DataFrame:
        """Aggregate portfolio totals without mixing currencies."""
        group_columns = tuple(group_by)
        if "risk_currency" not in group_columns and "currency" not in group_columns:
            raise ValueError("safe totals must include risk_currency or currency.")

        data = self._prepare(priced_portfolio, required_extra_columns=group_columns)
        if data.empty:
            return pd.DataFrame(columns=[*group_columns, *self.metrics, "line_count", "gross_price"])

        return self._aggregate(data, group_columns, include_gross_price=True)

    def total(
        self,
        priced_portfolio: pd.DataFrame,
        *,
        allow_mixed_currency: bool = False,
    ) -> dict[str, float]:
        """Return a single total only when it is currency-safe.

        By default this raises if more than one risk_currency is present.
        """
        data = self._prepare(priced_portfolio)

        currencies = sorted(data["risk_currency"].dropna().astype(str).unique().tolist())
        if len(currencies) > 1 and not allow_mixed_currency:
            raise ValueError(
                "Cannot compute a single portfolio total across multiple currencies "
                f"without FX conversion. Found risk_currency={currencies}."
            )

        return {metric: float(data[metric].sum()) for metric in self.metrics if metric in data.columns}

    def top_exposures(
        self,
        priced_portfolio: pd.DataFrame,
        *,
        metric: str = "vega",
        n: int = 10,
    ) -> pd.DataFrame:
        """Return largest absolute line exposures for one metric."""
        data = self._prepare(priced_portfolio)
        if data.empty or metric not in data.columns:
            return pd.DataFrame()

        return (
            data.assign(abs_exposure=data[metric].abs())
            .sort_values("abs_exposure", ascending=False)
            .head(n)
            .drop(columns="abs_exposure")
            .reset_index(drop=True)
        )

    def pnl_attribution(
        self,
        current: pd.DataFrame,
        previous: pd.DataFrame,
        *,
        key: str = "product_id",
    ) -> pd.DataFrame:
        """Compare two priced portfolios and attribute P&L by product id/currency."""
        current_prepared = self._prepare(current)
        previous_prepared = self._prepare(previous)

        key_columns = [
            column
            for column in (
                "portfolio",
                "risk_currency",
                key,
                "product_class",
                "risk_underlying",
            )
            if column in current_prepared.columns and column in previous_prepared.columns
        ]

        if key not in key_columns:
            key_columns.append(key)

        required = set(key_columns).union({"price"})
        missing_current = required.difference(current_prepared.columns)
        missing_previous = required.difference(previous_prepared.columns)

        if missing_current:
            raise ValueError(f"current portfolio is missing columns: {sorted(missing_current)}")
        if missing_previous:
            raise ValueError(f"previous portfolio is missing columns: {sorted(missing_previous)}")

        lhs = current_prepared[key_columns + ["price"]].rename(columns={"price": "price_current"})
        rhs = previous_prepared[key_columns + ["price"]].rename(columns={"price": "price_previous"})

        merged = lhs.merge(rhs, on=key_columns, how="outer")
        merged["price_current"] = merged["price_current"].fillna(0.0)
        merged["price_previous"] = merged["price_previous"].fillna(0.0)
        merged["pnl"] = merged["price_current"] - merged["price_previous"]

        return merged.sort_values("pnl", key=lambda s: s.abs(), ascending=False, ignore_index=True)

    def _aggregate(
        self,
        data: pd.DataFrame,
        group_columns: tuple[str, ...],
        *,
        include_gross_price: bool,
    ) -> pd.DataFrame:
        available_metrics = [metric for metric in self.metrics if metric in data.columns]

        grouped = (
            data.groupby(list(group_columns), dropna=False)[available_metrics]
            .sum(min_count=1)
            .reset_index()
        )

        counts = (
            data.groupby(list(group_columns), dropna=False)
            .size()
            .rename("line_count")
            .reset_index()
        )

        grouped = grouped.merge(counts, on=list(group_columns), how="left")

        if include_gross_price:
            gross = (
                data.assign(_abs_price=data["price"].abs() if "price" in data.columns else 0.0)
                .groupby(list(group_columns), dropna=False)["_abs_price"]
                .sum(min_count=1)
                .rename("gross_price")
                .reset_index()
            )
            grouped = grouped.merge(gross, on=list(group_columns), how="left")

        return grouped.sort_values(list(group_columns), ignore_index=True)

    def _prepare(
        self,
        priced_portfolio: pd.DataFrame,
        *,
        required_extra_columns: Iterable[str] = (),
    ) -> pd.DataFrame:
        if priced_portfolio is None or priced_portfolio.empty:
            return pd.DataFrame()

        data = priced_portfolio.copy()

        if "status" in data.columns:
            data = data[data["status"].astype(str).str.lower().isin(self.accepted_statuses)].copy()

        data = self._ensure_metadata_columns(data)

        if "maturity_bucket" not in data.columns:
            if "maturity_years" not in data.columns:
                raise ValueError("priced_portfolio must contain maturity_bucket or maturity_years.")
            data["maturity_bucket"] = data["maturity_years"].map(maturity_bucket)

        if "strike_bucket" not in data.columns:
            if "strike" not in data.columns:
                data["strike"] = np.nan
            if "spot" not in data.columns:
                if "spot_used" in data.columns:
                    data["spot"] = data["spot_used"]
                else:
                    data["spot"] = np.nan
            data["strike_bucket"] = [
                strike_bucket(k, s)
                for k, s in zip(data["strike"], data["spot"], strict=False)
            ]

        required = set(self.group_columns).union(required_extra_columns)
        missing = required.difference(data.columns)
        if missing:
            raise ValueError(f"priced_portfolio is missing columns: {sorted(missing)}")

        for metric in self.metrics:
            if metric not in data.columns:
                data[metric] = 0.0
            data[metric] = pd.to_numeric(data[metric], errors="coerce").fillna(0.0)

        return data

    @staticmethod
    def _ensure_metadata_columns(data: pd.DataFrame) -> pd.DataFrame:
        out = data.copy()

        if "portfolio" not in out.columns:
            out["portfolio"] = "default"
        out["portfolio"] = _clean_text_series(out["portfolio"], default="default", upper=False)

        if "currency" not in out.columns:
            out["currency"] = "EUR"
        out["currency"] = _clean_text_series(out["currency"], default="EUR", upper=True)

        if "risk_currency" not in out.columns:
            out["risk_currency"] = out["currency"]
        out["risk_currency"] = _clean_text_series(out["risk_currency"], default="EUR", upper=True)

        if "product_class" not in out.columns:
            if "product_type" in out.columns:
                out["product_class"] = out["product_type"]
            else:
                out["product_class"] = "UnknownProduct"
        out["product_class"] = _clean_text_series(out["product_class"], default="UnknownProduct", upper=False)

        if "underlying" not in out.columns:
            out["underlying"] = ""
        out["underlying"] = _clean_text_series(out["underlying"], default="", upper=True)

        source_sheet = (
            out["source_sheet"].astype("string").str.strip().str.lower()
            if "source_sheet" in out.columns
            else pd.Series("", index=out.index, dtype="string")
        )

        is_rate_product = out["product_class"].astype(str).isin(RATE_PRODUCT_CLASSES)
        is_rate_source = source_sheet.isin(RATE_SOURCE_SHEETS)
        rate_curve_label = out["risk_currency"].astype(str).str.upper() + "_RATE_CURVE"

        if "risk_underlying" not in out.columns:
            out["risk_underlying"] = out["underlying"]

        out["risk_underlying"] = _clean_text_series(out["risk_underlying"], default="", upper=True)

        missing_risk_underlying = out["risk_underlying"].eq("")
        out.loc[missing_risk_underlying & (is_rate_product | is_rate_source), "risk_underlying"] = rate_curve_label
        out.loc[missing_risk_underlying & ~(is_rate_product | is_rate_source), "risk_underlying"] = "UNKNOWN_UNDERLYING"

        missing_underlying = out["underlying"].eq("")
        out.loc[missing_underlying, "underlying"] = out.loc[missing_underlying, "risk_underlying"]

        return out


def _clean_text_series(series: pd.Series, *, default: str, upper: bool) -> pd.Series:
    values = (
        series.astype("string")
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA, "NaT": pd.NA})
        .fillna(default)
    )
    if upper:
        values = values.str.upper()
    return values


__all__ = ["RiskAggregator"]