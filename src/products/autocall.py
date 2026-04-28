"""Simplified autocallable product definition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.products._helpers import extract_path, normalize_positive_float
from src.products.base_product import Product


def _normalize_level(value: float) -> float:
    level = float(value)
    if level > 2.0:
        level = level / 100.0
    if level <= 0.0:
        raise ValueError("level must be strictly positive.")
    return level


def _is_number_like(value: Any) -> bool:
    try:
        float(value)
        return True
    except Exception:
        return False


@dataclass(slots=True)
class AutocallProduct(Product):
    product_id: str
    underlying: str
    observation_dates: list[float | str | pd.Timestamp]
    trigger_levels: list[float]
    coupon_rate: float
    barrier_protection: float
    notional: float = 100.0
    initial_spot: float = 100.0
    currency: str = "EUR"
    maturity: float | None = None

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

        self.trigger_levels = [_normalize_level(x) for x in self.trigger_levels]

        if not all(_is_number_like(x) for x in self.observation_dates):
            raise ValueError("observation_dates must be converted to year fractions before creating AutocallProduct.")

        self.observation_dates = [max(float(x), 1e-12) for x in self.observation_dates]

        if self.maturity is None:
            self.maturity = max(self.observation_dates)

        self.maturity = max(float(self.maturity), max(self.observation_dates), 1e-12)

    @property
    def requires_monte_carlo(self) -> bool:
        return True

    def payoff(self, market_data: Any) -> float:
        path = extract_path(market_data)
        if len(path) < len(self.observation_dates):
            raise ValueError("path must contain at least one spot per observation date.")

        observed = path[: len(self.observation_dates)]

        for spot, trigger, obs_time in zip(observed, self.trigger_levels, self.observation_dates, strict=True):
            if float(spot) >= self.initial_spot * float(trigger):
                return float(self.notional * (1.0 + self.coupon_rate * float(obs_time)))

        final_spot = float(observed[-1])
        final_time = float(self.observation_dates[-1])

        if final_spot >= self.initial_spot * self.barrier_protection:
            return float(self.notional * (1.0 + self.coupon_rate * final_time))

        return float(self.notional * final_spot / self.initial_spot)

    def get_risk_factors(self) -> list[str]:
        return ["spot", "rate", "volatility"]


__all__ = ["AutocallProduct"]
