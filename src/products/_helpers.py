"""Shared product helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def extract_spot(market_data: Any) -> float:
    """Extract one terminal spot from a MarketData object, dict, or raw number."""
    if isinstance(market_data, int | float):
        return float(market_data)

    if isinstance(market_data, dict):
        if "spot" in market_data:
            return float(market_data["spot"])
        if "path" in market_data:
            path = market_data["path"]
            if len(path) == 0:
                raise ValueError("market_data['path'] cannot be empty.")
            return float(path[-1])
        raise ValueError("market_data dict must contain a 'spot' or 'path' key.")

    spot = getattr(market_data, "spot", None)
    if spot is None:
        raise ValueError("market_data must provide a spot.")

    return float(spot)


def extract_path(market_data: Any) -> np.ndarray:
    """Extract a spot path when available; otherwise return an array with terminal spot."""
    if isinstance(market_data, dict) and "path" in market_data:
        path = np.asarray(market_data["path"], dtype=float)
    elif hasattr(market_data, "path"):
        path = np.asarray(getattr(market_data, "path"), dtype=float)
    elif isinstance(market_data, np.ndarray):
        path = np.asarray(market_data, dtype=float)
    elif isinstance(market_data, Sequence) and not isinstance(market_data, str | bytes | dict):
        path = np.asarray(market_data, dtype=float)
    else:
        path = np.asarray([extract_spot(market_data)], dtype=float)

    if path.ndim != 1 or len(path) == 0:
        raise ValueError("path must be a non-empty one-dimensional sequence.")
    if not np.all(np.isfinite(path)):
        raise ValueError("path contains non-finite values.")
    return path


def normalize_positive_float(value: float, field_name: str) -> float:
    result = float(value)
    if result <= 0.0:
        raise ValueError(f"{field_name} must be strictly positive.")
    return result


def normalize_non_negative_float(value: float, field_name: str) -> float:
    result = float(value)
    if result < 0.0:
        raise ValueError(f"{field_name} must be non-negative.")
    return result


__all__ = [
    "extract_path",
    "extract_spot",
    "normalize_non_negative_float",
    "normalize_positive_float",
]
