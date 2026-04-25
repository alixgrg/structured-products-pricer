"""Small portfolio-level containers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class PortfolioSnapshot:
    """Lightweight portfolio descriptor used for smoke tests and demos."""

    name: str
    position_count: int = 0


__all__ = ["PortfolioSnapshot"]
