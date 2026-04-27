"""Simplified barrier option product.

This implementation uses terminal barrier checks only.
It is useful for payoff analysis in notebooks and tests.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.products._helpers import extract_spot
from src.products.base_product import Product


@dataclass(frozen=True, slots=True)
class BarrierOption(Product):
    """Terminal barrier option with in/out feature."""

    product_id: str
    option_type: str
    strike: float
    maturity: float
    barrier: float
    barrier_type: str
    notional: float = 1.0

    def __post_init__(self) -> None:
        option_type = self.option_type.lower().strip()
        if option_type in {"c", "call"}:
            option_type = "call"
        elif option_type in {"p", "put"}:
            option_type = "put"
        else:
            raise ValueError("option_type must be 'call' or 'put'.")

        barrier_type = self.barrier_type.lower().strip()
        supported = {"up-and-out", "down-and-out", "up-and-in", "down-and-in"}
        if barrier_type not in supported:
            raise ValueError(f"barrier_type must be one of {sorted(supported)}")

        if self.strike <= 0.0:
            raise ValueError("strike must be strictly positive.")
        if self.barrier <= 0.0:
            raise ValueError("barrier must be strictly positive.")
        if self.maturity < 0.0:
            raise ValueError("maturity must be non-negative.")
        if self.notional <= 0.0:
            raise ValueError("notional must be strictly positive.")

        object.__setattr__(self, "option_type", option_type)
        object.__setattr__(self, "barrier_type", barrier_type)

    def payoff(self, market_data) -> float:
        spot = extract_spot(market_data)

        touched = _is_terminal_barrier_touched(spot, self.barrier_type, self.barrier)
        knocked_out = self.barrier_type.endswith("out") and touched
        knocked_in = self.barrier_type.endswith("in") and touched

        if self.barrier_type.endswith("out") and knocked_out:
            return 0.0

        if self.barrier_type.endswith("in") and not knocked_in:
            return 0.0

        if self.option_type == "call":
            return float(self.notional * max(spot - self.strike, 0.0))

        return float(self.notional * max(self.strike - spot, 0.0))

    def get_risk_factors(self) -> list[str]:
        return ["spot", "rate", "volatility", "barrier"]


def _is_terminal_barrier_touched(spot: float, barrier_type: str, barrier: float) -> bool:
    if barrier_type.startswith("up"):
        return bool(spot >= barrier)
    return bool(spot <= barrier)


__all__ = ["BarrierOption"]
