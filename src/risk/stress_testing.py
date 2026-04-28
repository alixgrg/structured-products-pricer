"""Portfolio stress testing utilities.

The stress tester reuses PortfolioPricingEngine and applies market shocks:
- spot shocks through spot_by_underlying and default_spot;
- volatility shocks through volatility_by_underlying and, when available, a
  shifted volatility surface;
- rate shocks through default_rate and a parallel-shifted YieldCurve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.portfolio.pricing_engine import PortfolioPricingConfig, PortfolioPricingEngine
from src.rates.yield_curve import YieldCurve
from src.risk.numerical_greeks import NumericalGreeksEngine


@dataclass(frozen=True, slots=True)
class StressScenario:
    """Market shock definition.

    Shocks are expressed in decimal format:
    - spot_shock = 0.10 means +10%;
    - vol_shock = 0.05 means +5 volatility points;
    - rate_shock = 0.01 means +100 bps.
    """

    name: str
    spot_shock: float = 0.0
    vol_shock: float = 0.0
    rate_shock: float = 0.0
    dividend_yield_shock: float = 0.0
    spot_shocks_by_underlying: dict[str, float] = field(default_factory=dict)
    vol_shocks_by_underlying: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StressTestResult:
    """Container returned by StressTester.run()."""

    line_results: pd.DataFrame
    scenario_summary: pd.DataFrame
    pnl_by_position: pd.DataFrame


class ShiftedVolSurface:
    """Simple wrapper applying an additive volatility shift."""

    def __init__(self, base_surface: Any, shift: float, floor: float = 1e-6) -> None:
        self.base_surface = base_surface
        self.shift = float(shift)
        self.floor = float(floor)

    def volatility(self, maturity, log_moneyness):
        values = self.base_surface.volatility(maturity, log_moneyness)
        return np.maximum(np.asarray(values, dtype=float) + self.shift, self.floor)


@dataclass(slots=True)
class StressTester:
    """Run stress scenarios by repricing the full portfolio."""

    base_config: PortfolioPricingConfig
    yield_curve: YieldCurve | None = None
    vol_surface: Any | None = None
    numerical_greeks_engine: NumericalGreeksEngine | None = None
    use_numerical_greeks: bool = False
    numerical_greeks_for: tuple[str, ...] = ("BarrierOption", "AutocallProduct")

    def run(
        self,
        inventory: dict[str, pd.DataFrame] | pd.DataFrame,
        scenarios: Sequence[StressScenario],
    ) -> StressTestResult:
        if not scenarios:
            raise ValueError("scenarios must contain at least one StressScenario.")

        line_frames: list[pd.DataFrame] = []

        base_result: pd.DataFrame | None = None
        for scenario in scenarios:
            priced = self.price_scenario(inventory, scenario)
            priced["scenario"] = scenario.name
            line_frames.append(priced)
            if scenario.name.lower() in {"base", "central", "baseline"}:
                base_result = priced

        line_results = pd.concat(line_frames, ignore_index=True, sort=False)

        if base_result is None:
            base_result = line_frames[0]

        scenario_summary = self._scenario_summary(line_results, base_scenario=base_result["scenario"].iloc[0])
        pnl_by_position = self._pnl_by_position(line_results, base_scenario=base_result["scenario"].iloc[0])

        return StressTestResult(
            line_results=line_results,
            scenario_summary=scenario_summary,
            pnl_by_position=pnl_by_position,
        )

    def price_scenario(
        self,
        inventory: dict[str, pd.DataFrame] | pd.DataFrame,
        scenario: StressScenario,
    ) -> pd.DataFrame:
        config = self._scenario_config(scenario)
        curve = self._shifted_curve(scenario.rate_shock)
        surface = self._shifted_surface(scenario.vol_shock)

        engine = PortfolioPricingEngine(
            config=config,
            yield_curve=curve,
            vol_surface=surface,
            numerical_greeks_engine=self.numerical_greeks_engine,
            use_numerical_greeks=self.use_numerical_greeks,
            numerical_greeks_for=self.numerical_greeks_for,
        )
        return engine.price_portfolio(inventory)

    def _scenario_config(self, scenario: StressScenario) -> PortfolioPricingConfig:
        spot_by_underlying = {}
        for underlying, base_spot in self.base_config.spot_by_underlying.items():
            shock = scenario.spot_shocks_by_underlying.get(underlying, scenario.spot_shock)
            spot_by_underlying[underlying] = float(base_spot) * (1.0 + float(shock))

        vol_by_underlying = {}
        for underlying, base_vol in self.base_config.volatility_by_underlying.items():
            shock = scenario.vol_shocks_by_underlying.get(underlying, scenario.vol_shock)
            vol_by_underlying[underlying] = max(float(base_vol) + float(shock), 1e-6)

        return PortfolioPricingConfig(
            default_spot=float(self.base_config.default_spot) * (1.0 + float(scenario.spot_shock)),
            default_rate=float(self.base_config.default_rate) + float(scenario.rate_shock),
            default_volatility=max(float(self.base_config.default_volatility) + float(scenario.vol_shock), 1e-6),
            dividend_yield=float(self.base_config.dividend_yield) + float(scenario.dividend_yield_shock),
            spot_by_underlying=spot_by_underlying,
            volatility_by_underlying=vol_by_underlying,
            n_paths=self.base_config.n_paths,
            n_steps=self.base_config.n_steps,
            seed=self.base_config.seed,
        )

    def _shifted_curve(self, rate_shift: float) -> YieldCurve | None:
        if self.yield_curve is None:
            return None
        if abs(rate_shift) < 1e-15:
            return self.yield_curve

        return YieldCurve(
            maturities=self.yield_curve.maturities.copy(),
            zero_rates=self.yield_curve.zero_rates + float(rate_shift),
            interpolation=self.yield_curve.interpolation,
            name=f"{self.yield_curve.name}_shift_{float(rate_shift):+.4f}",
            interpolation_on=self.yield_curve.interpolation_on,
        )

    def _shifted_surface(self, vol_shift: float):
        if self.vol_surface is None:
            return None
        if abs(vol_shift) < 1e-15:
            return self.vol_surface
        if not hasattr(self.vol_surface, "volatility"):
            raise TypeError("vol_surface must expose volatility().")
        return ShiftedVolSurface(self.vol_surface, vol_shift)

    @staticmethod
    def _scenario_summary(line_results: pd.DataFrame, *, base_scenario: str) -> pd.DataFrame:
        metrics = [c for c in ("price", "delta", "gamma", "vega", "theta", "rho", "dv01") if c in line_results.columns]

        group_cols = [
            column
            for column in ("scenario", "portfolio", "risk_currency")
            if column in line_results.columns
        ]

        if "scenario" not in group_cols:
            group_cols.insert(0, "scenario")

        summary = (
            line_results.groupby(group_cols, dropna=False)[metrics]
            .sum(min_count=1)
            .reset_index()
        )

        counts = (
            line_results.groupby(group_cols, dropna=False)
            .size()
            .rename("line_count")
            .reset_index()
        )
        summary = summary.merge(counts, on=group_cols, how="left")

        base_keys = [column for column in group_cols if column != "scenario"]
        base = summary[summary["scenario"] == base_scenario].copy()

        if base_keys:
            base = base[base_keys + ["price"]].rename(columns={"price": "base_price"})
            summary = summary.merge(base, on=base_keys, how="left")
        else:
            base_price = float(summary.loc[summary["scenario"] == base_scenario, "price"].iloc[0])
            summary["base_price"] = base_price

        summary["base_price"] = summary["base_price"].fillna(0.0)
        summary["pnl_vs_base"] = summary["price"] - summary["base_price"]
        summary["base_scenario"] = base_scenario

        return summary

    @staticmethod
    def _pnl_by_position(line_results: pd.DataFrame, *, base_scenario: str) -> pd.DataFrame:
        required = {"scenario", "product_id", "price"}
        missing = required.difference(line_results.columns)
        if missing:
            raise ValueError(f"line_results missing required columns: {sorted(missing)}")

        index_cols = [
            c
            for c in (
                "portfolio",
                "risk_currency",
                "product_id",
                "product_type",
                "product_class",
                "underlying",
                "risk_underlying",
                "maturity_bucket",
                "strike_bucket",
            )
            if c in line_results.columns
        ]

        base = (
            line_results[line_results["scenario"] == base_scenario][index_cols + ["price"]]
            .rename(columns={"price": "base_price"})
        )

        rows = []
        for scenario, group in line_results.groupby("scenario", dropna=False):
            current = group[index_cols + ["price"]].rename(columns={"price": "scenario_price"})
            merged = current.merge(base, on=index_cols, how="left")
            merged["scenario"] = scenario
            merged["base_scenario"] = base_scenario
            merged["base_price"] = merged["base_price"].fillna(0.0)
            merged["pnl"] = merged["scenario_price"] - merged["base_price"]
            rows.append(merged)

        result = pd.concat(rows, ignore_index=True, sort=False)
        return result.sort_values(
            ["scenario", "pnl"],
            key=lambda s: s.abs() if s.name == "pnl" else s,
            ascending=False,
            ignore_index=True,
        )

__all__ = [
    "ShiftedVolSurface",
    "StressScenario",
    "StressTester",
    "StressTestResult",
]
