from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MarketData:
    spot: float | None = None
    rate: float | None = None
    volatility: float | None = None
    dividend_yield: float = 0.0