"""Shared product helpers."""

from __future__ import annotations


def extract_spot(market_data) -> float:
    """Extract a spot from a MarketData object, dict, or raw number."""
    if isinstance(market_data, int | float):
        return float(market_data)

    if isinstance(market_data, dict):
        if "spot" not in market_data:
            raise ValueError("market_data dict must contain a 'spot' key.")
        return float(market_data["spot"])

    spot = getattr(market_data, "spot", None)
    if spot is None:
        raise ValueError("market_data must provide a spot.")

    return float(spot)

