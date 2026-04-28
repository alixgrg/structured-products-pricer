"""Simplified autocallable product definition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.products._helpers import extract_path, normalize_positive_float
from src.products.base_product import Product


@dataclass(slots=True)
class AutocallProduct(Product):
    """Simplified mono-underlying autocall.

    The product is intentionally path-dependent and should be routed to a Monte Carlo
    model in Phase 2/3. ``payoff`` is implemented to validate scenarios and tests.
    """

    product_id: str
    underlying: str
    observation_dates: list[float | str | pd.Timestamp]
    trigger_levels: list[float]
    coupon_rate: float
    barrier_protection: float
    notional: float = 100.0
    initial_spot: float = 100.0
    currency: str = "EUR"

    def __post_init__(self) -> None:
        self.notional = normalize_positive_float(self.notional, "notional")
        self.initial_spot = normalize_positive_float(self.initial_spot, "initial_spot")
        self.underlying = str(self.underlying).strip().upper()
        self.currency = str(self.currency).strip().upper()
        self.coupon_rate = float(self.coupon_rate)
        self.barrier_protection = _normalize_level(self.barrier_protection)
        if not self.observation_dates:
            raise ValueError("observation_dates cannot be empty.")
        if len(self.trigger_levels) == 1:
            self.trigger_levels = list(self.trigger_levels) * len(self.observation_dates)
        if len(self.trigger_levels) != len(self.observation_dates):
            raise ValueError("trigger_levels must contain either one level or one per observation date.")
        self.trigger_levels = [_normalize_level(level) for level in self.trigger_levels]
        self.maturity = float(len(self.observation_dates))

    @property
    def requires_monte_carlo(self) -> bool:
        return True

    def payoff(self, market_data: Any) -> float:
        path = extract_path(market_data)
        if len(path) < len(self.observation_dates):
            raise ValueError("path must contain at least one spot per observation date.")

        observed = path[: len(self.observation_dates)]
        for index, (spot, trigger) in enumerate(zip(observed, self.trigger_levels, strict=True), start=1):
            if spot >= self.initial_spot * trigger:
                return float(self.notional * (1.0 + self.coupon_rate * index))

        final_spot = float(observed[-1])
        if final_spot >= self.initial_spot * self.barrier_protection:
            return float(self.notional * (1.0 + self.coupon_rate * len(self.observation_dates)))

        # Capital is at risk below protection barrier.
        return float(self.notional * final_spot / self.initial_spot)

    def get_risk_factors(self) -> list[str]:
        return ["spot", "rate", "volatility", "autocall_trigger", "protection_barrier"]


def _normalize_level(value: float) -> float:
    level = float(value)
    if level > 10.0:
        return level / 100.0
    return level


__all__ = ["AutocallProduct"]
