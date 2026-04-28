"""Barrier option product definition.

The analytic Merton/Reiner-Rubinstein pricing belongs in ``src.models.barrier_model``
(Phase 2). This product stores all information required by that model and provides
payoff logic for terminal/path validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from src.products._helpers import extract_path, normalize_non_negative_float, normalize_positive_float
from src.products.base_product import Product

BarrierKind = Literal["KO", "KI"]
BarrierDirection = Literal["up", "down"]


@dataclass(slots=True)
class BarrierOption(Product):
    """Single-barrier European option, knock-out or knock-in.

    Parameters
    ----------
    barrier_type:
        ``"KO"`` or ``"KI"``. Legacy values such as ``"up-and-out"`` are also
        accepted for backward compatibility.
    barrier_direction:
        ``"up"`` or ``"down"``. If omitted, it is inferred from ``initial_spot``
        when available, otherwise from ``barrier >= strike``.
    """

    product_id: str
    option_type: str
    strike: float
    maturity: float
    barrier: float
    barrier_type: str
    notional: float = 1.0
    underlying: str = ""
    barrier_direction: str | None = None
    initial_spot: float | None = None
    currency: str = "EUR"
    dividend_yield: float | None = None

    def __post_init__(self) -> None:
        option_type = str(self.option_type).strip().lower()
        if option_type in {"c", "call"}:
            option_type = "call"
        elif option_type in {"p", "put"}:
            option_type = "put"
        else:
            raise ValueError("option_type must be 'call' or 'put'.")

        barrier_type_raw = str(self.barrier_type).strip().lower().replace("_", "-")
        direction = self.barrier_direction
        if barrier_type_raw in {"ko", "knock-out", "knockout"}:
            barrier_type = "KO"
        elif barrier_type_raw in {"ki", "knock-in", "knockin"}:
            barrier_type = "KI"
        elif barrier_type_raw in {"up-and-out", "down-and-out", "up-out", "down-out"}:
            barrier_type = "KO"
            direction = "up" if barrier_type_raw.startswith("up") else "down"
        elif barrier_type_raw in {"up-and-in", "down-and-in", "up-in", "down-in"}:
            barrier_type = "KI"
            direction = "up" if barrier_type_raw.startswith("up") else "down"
        else:
            raise ValueError("barrier_type must be 'KO', 'KI', or legacy up/down-and-in/out.")

        self.option_type = option_type
        self.barrier_type = barrier_type
        self.strike = normalize_positive_float(self.strike, "strike")
        self.maturity = normalize_non_negative_float(self.maturity, "maturity")
        self.barrier = normalize_positive_float(self.barrier, "barrier")
        self.notional = normalize_positive_float(self.notional, "notional")
        self.underlying = str(self.underlying).strip().upper()
        self.currency = str(self.currency).strip().upper()
        if self.initial_spot is not None:
            self.initial_spot = normalize_positive_float(self.initial_spot, "initial_spot")
        if self.dividend_yield is not None:
            self.dividend_yield = float(self.dividend_yield)

        if direction is None:
            reference = self.initial_spot if self.initial_spot is not None else self.strike
            direction = "up" if self.barrier >= reference else "down"
        direction = str(direction).strip().lower()
        if direction not in {"up", "down"}:
            raise ValueError("barrier_direction must be 'up' or 'down'.")
        self.barrier_direction = direction

    @property
    def is_knock_out(self) -> bool:
        return self.barrier_type == "KO"

    @property
    def is_knock_in(self) -> bool:
        return self.barrier_type == "KI"

    def barrier_touched(self, path_or_market_data) -> bool:
        path = extract_path(path_or_market_data)
        if self.barrier_direction == "up":
            return bool(np.max(path) >= self.barrier)
        return bool(np.min(path) <= self.barrier)

    def vanilla_payoff(self, terminal_spot: float) -> float:
        if self.option_type == "call":
            return float(self.notional * max(terminal_spot - self.strike, 0.0))
        return float(self.notional * max(self.strike - terminal_spot, 0.0))

    def payoff(self, market_data) -> float:
        path = extract_path(market_data)
        touched = self.barrier_touched(path)
        terminal_spot = float(path[-1])

        if self.is_knock_out and touched:
            return 0.0
        if self.is_knock_in and not touched:
            return 0.0
        return self.vanilla_payoff(terminal_spot)

    def get_risk_factors(self) -> list[str]:
        return ["spot", "rate", "volatility", "barrier"]


__all__ = ["BarrierOption"]
