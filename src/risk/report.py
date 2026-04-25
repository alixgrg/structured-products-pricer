"""Containers for pricing outputs and risks."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class RiskSnapshot:
    """Store a priced position and its risk metrics."""

    product_id: str
    price: float = 0.0
    metrics: dict[str, float] = field(default_factory=dict)


__all__ = ["RiskSnapshot"]
