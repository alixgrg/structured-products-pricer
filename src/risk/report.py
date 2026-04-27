"""Risk containers and aggregation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(slots=True)
class RiskSnapshot:
    """Store a priced position and its risk metrics."""

    product_id: str
    price: float = 0.0
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class PortfolioRiskSummary:
    """Hold precomputed risk tables used in reports/notebooks."""

    by_product: pd.DataFrame
    by_underlying: pd.DataFrame
    by_maturity: pd.DataFrame
    by_portfolio: pd.DataFrame


def aggregate_greeks(
    line_valuations: pd.DataFrame,
    *,
    group_by: list[str],
) -> pd.DataFrame:
    """Aggregate portfolio greeks by selected keys on supported rows."""
    supported = line_valuations[line_valuations["status"] == "supported"].copy()
    if supported.empty:
        return pd.DataFrame(columns=group_by + ["price", "delta", "gamma", "vega", "theta", "rho", "line_count"])

    metrics = ["price", "delta", "gamma", "vega", "theta", "rho"]

    grouped = (
        supported.groupby(group_by, dropna=False)[metrics]
        .sum(min_count=1)
        .reset_index()
    )
    counts = supported.groupby(group_by, dropna=False).size().rename("line_count").reset_index()
    return grouped.merge(counts, on=group_by, how="left")


def build_portfolio_risk_summary(line_valuations: pd.DataFrame) -> PortfolioRiskSummary:
    """Create standard risk summary tables for portfolio reporting."""
    return PortfolioRiskSummary(
        by_product=aggregate_greeks(line_valuations, group_by=["product_type"]),
        by_underlying=aggregate_greeks(line_valuations, group_by=["underlying"]),
        by_maturity=aggregate_greeks(line_valuations, group_by=["maturity_years"]),
        by_portfolio=aggregate_greeks(line_valuations, group_by=["portfolio"]),
    )


def risk_pivot_table(
    line_valuations: pd.DataFrame,
    *,
    index: str,
    columns: str,
    value: str,
) -> pd.DataFrame:
    """Build a pivot table for notebook visualizations."""
    supported = line_valuations[line_valuations["status"] == "supported"].copy()
    if supported.empty:
        return pd.DataFrame()

    return pd.pivot_table(
        supported,
        index=index,
        columns=columns,
        values=value,
        aggfunc="sum",
        fill_value=0.0,
    )


__all__ = [
    "PortfolioRiskSummary",
    "RiskSnapshot",
    "aggregate_greeks",
    "build_portfolio_risk_summary",
    "risk_pivot_table",
]
