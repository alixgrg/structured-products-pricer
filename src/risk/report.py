"""Risk containers, aggregation helpers and portfolio report exports.

Phase 4 / step 15.

This file keeps the previous helper functions while extending ``RiskSnapshot``
and adding ``PortfolioRiskReport`` for CSV exports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


RISK_METRICS = ["price", "delta", "gamma", "vega", "theta", "rho", "dv01"]


@dataclass(slots=True)
class RiskSnapshot:
    """Store a priced position or a portfolio-level risk snapshot.

    The first three fields are intentionally backward-compatible with the
    previous minimal class: ``RiskSnapshot(product_id, price, metrics)`` still
    works.
    """

    product_id: str = "portfolio"
    price: float = 0.0
    metrics: dict[str, float] = field(default_factory=dict)
    valuation_date: pd.Timestamp | None = None
    line_valuations: pd.DataFrame = field(default_factory=pd.DataFrame)
    portfolio_greeks: pd.DataFrame = field(default_factory=pd.DataFrame)
    pnl_attribution: pd.DataFrame = field(default_factory=pd.DataFrame)
    top_exposures: pd.DataFrame = field(default_factory=pd.DataFrame)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_priced_portfolio(
        cls,
        priced_portfolio: pd.DataFrame,
        *,
        valuation_date: str | pd.Timestamp | None = None,
        name: str = "portfolio",
        top_metric: str = "vega",
        top_n: int = 10,
    ) -> "RiskSnapshot":
        from src.risk.aggregator import RiskAggregator

        aggregator = RiskAggregator()
        supported = _supported_rows(priced_portfolio)

        portfolio_greeks = aggregator.aggregate_by_pillar(supported)
        top_exposures = aggregator.top_exposures(supported, metric=top_metric, n=top_n)
        safe_totals = aggregator.aggregate_safe_totals(supported)

        # Avoid one global total if multiple risk currencies are present.
        if len(safe_totals) == 1:
            metrics = {
                metric: float(safe_totals[metric].iloc[0])
                for metric in RISK_METRICS
                if metric in safe_totals.columns
            }
            price = float(metrics.get("price", 0.0))
            mixed_currency_total_blocked = False
        else:
            metrics = {}
            price = 0.0
            mixed_currency_total_blocked = True

        return cls(
            product_id=name,
            price=price,
            metrics=metrics,
            valuation_date=pd.to_datetime(valuation_date) if valuation_date is not None else None,
            line_valuations=supported.reset_index(drop=True),
            portfolio_greeks=portfolio_greeks,
            top_exposures=top_exposures,
            metadata={
                "line_count": int(len(supported)),
                "top_metric": top_metric,
                "mixed_currency_total_blocked": mixed_currency_total_blocked,
                "safe_total_rows": int(len(safe_totals)),
            },
        )


@dataclass(slots=True)
class PortfolioRiskSummary:
    """Hold precomputed risk tables used in reports/notebooks.

    No table here aggregates multiple risk currencies into a single total.
    """

    by_portfolio_currency: pd.DataFrame
    by_product: pd.DataFrame
    by_product_class: pd.DataFrame
    by_underlying: pd.DataFrame
    by_risk_underlying: pd.DataFrame
    by_maturity: pd.DataFrame
    by_pillar: pd.DataFrame


@dataclass(slots=True)
class PortfolioRiskReport:
    """Collection of risk snapshots with dataframe and CSV export helpers."""

    snapshots: list[RiskSnapshot]

    @classmethod
    def from_priced_portfolio(
        cls,
        priced_portfolio: pd.DataFrame,
        *,
        valuation_date: str | pd.Timestamp | None = None,
        name: str = "portfolio",
    ) -> "PortfolioRiskReport":
        return cls([RiskSnapshot.from_priced_portfolio(priced_portfolio, valuation_date=valuation_date, name=name)])

    def to_dataframe(self) -> pd.DataFrame:
        """Return one summary row per snapshot."""
        rows: list[dict[str, Any]] = []
        for snapshot in self.snapshots:
            row: dict[str, Any] = {
                "snapshot": snapshot.product_id,
                "valuation_date": snapshot.valuation_date,
                "price": snapshot.price,
                "line_count": snapshot.metadata.get("line_count", len(snapshot.line_valuations)),
            }
            row.update(snapshot.metrics)
            rows.append(row)
        return pd.DataFrame(rows)

    def portfolio_greeks_to_dataframe(self) -> pd.DataFrame:
        """Return concatenated pillar-level Greeks for all snapshots."""
        frames: list[pd.DataFrame] = []
        for snapshot in self.snapshots:
            if snapshot.portfolio_greeks.empty:
                continue
            frame = snapshot.portfolio_greeks.copy()
            frame.insert(0, "snapshot", snapshot.product_id)
            frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def pnl_attribution_to_dataframe(self) -> pd.DataFrame:
        """Return concatenated P&L attribution tables for all snapshots."""
        frames: list[pd.DataFrame] = []
        for snapshot in self.snapshots:
            if snapshot.pnl_attribution.empty:
                continue
            frame = snapshot.pnl_attribution.copy()
            frame.insert(0, "snapshot", snapshot.product_id)
            frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def top_exposures_to_dataframe(self) -> pd.DataFrame:
        """Return concatenated top-exposure tables for all snapshots."""
        frames: list[pd.DataFrame] = []
        for snapshot in self.snapshots:
            if snapshot.top_exposures.empty:
                continue
            frame = snapshot.top_exposures.copy()
            frame.insert(0, "snapshot", snapshot.product_id)
            frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def export_csv(self, path: str | Path) -> dict[str, Path]:
        """Export summary and detailed report tables as CSV files.

        If ``path`` is a directory, files are created inside it. If ``path`` is a
        CSV file path, sibling files are created with suffixes.
        """
        target = Path(path)
        if target.suffix.lower() == ".csv":
            target.parent.mkdir(parents=True, exist_ok=True)
            base = target.with_suffix("")
            paths = {
                "summary": target,
                "portfolio_greeks": Path(f"{base}_portfolio_greeks.csv"),
                "top_exposures": Path(f"{base}_top_exposures.csv"),
                "pnl_attribution": Path(f"{base}_pnl_attribution.csv"),
            }
        else:
            target.mkdir(parents=True, exist_ok=True)
            paths = {
                "summary": target / "risk_summary.csv",
                "portfolio_greeks": target / "portfolio_greeks.csv",
                "top_exposures": target / "top_exposures.csv",
                "pnl_attribution": target / "pnl_attribution.csv",
            }

        self.to_dataframe().to_csv(paths["summary"], index=False)
        self.portfolio_greeks_to_dataframe().to_csv(paths["portfolio_greeks"], index=False)
        self.top_exposures_to_dataframe().to_csv(paths["top_exposures"], index=False)
        self.pnl_attribution_to_dataframe().to_csv(paths["pnl_attribution"], index=False)
        return paths


def aggregate_greeks(
    line_valuations: pd.DataFrame,
    *,
    group_by: list[str],
) -> pd.DataFrame:
    """Aggregate portfolio Greeks by selected keys on supported/priced rows.

    The caller must include risk_currency in group_by when totals are intended.
    """
    from src.risk.aggregator import RiskAggregator

    aggregator = RiskAggregator()
    return aggregator.aggregate_by(line_valuations, group_by=group_by)


def build_portfolio_risk_summary(line_valuations: pd.DataFrame) -> PortfolioRiskSummary:
    """Create currency-safe standard risk summary tables."""
    from src.risk.aggregator import RiskAggregator

    aggregator = RiskAggregator()

    return PortfolioRiskSummary(
        by_portfolio_currency=aggregator.aggregate_safe_totals(
            line_valuations,
            group_by=["portfolio", "risk_currency"],
        ),
        by_product=aggregator.aggregate_by(
            line_valuations,
            group_by=["portfolio", "risk_currency", "product_type"],
        ),
        by_product_class=aggregator.aggregate_by(
            line_valuations,
            group_by=["portfolio", "risk_currency", "product_class"],
        ),
        by_underlying=aggregator.aggregate_by(
            line_valuations,
            group_by=["portfolio", "risk_currency", "underlying"],
        ),
        by_risk_underlying=aggregator.aggregate_by(
            line_valuations,
            group_by=["portfolio", "risk_currency", "risk_underlying"],
        ),
        by_maturity=aggregator.aggregate_by(
            line_valuations,
            group_by=["portfolio", "risk_currency", "maturity_bucket"]
            if "maturity_bucket" in line_valuations.columns
            else ["portfolio", "risk_currency", "maturity_years"],
        ),
        by_pillar=aggregator.aggregate_by_pillar(line_valuations),
    )


def risk_pivot_table(
    line_valuations: pd.DataFrame,
    *,
    index: str,
    columns: str,
    value: str,
) -> pd.DataFrame:
    """Build a pivot table for notebook visualizations."""
    supported = _supported_rows(line_valuations)
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


def _supported_rows(line_valuations: pd.DataFrame) -> pd.DataFrame:
    if line_valuations is None or line_valuations.empty:
        return pd.DataFrame()

    from src.risk.aggregator import RiskAggregator

    return RiskAggregator()._prepare(line_valuations)


__all__ = [
    "PortfolioRiskReport",
    "PortfolioRiskSummary",
    "RiskSnapshot",
    "aggregate_greeks",
    "build_portfolio_risk_summary",
    "risk_pivot_table",
]
