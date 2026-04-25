"""Minimal dashboard configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DashboardConfig:
    """Initial Streamlit configuration used by future steps."""

    app_title: str = "Structured Products Factory"
    refresh_interval_seconds: int = 60
    default_pages: tuple[str, ...] = ("overview", "market", "portfolio")


__all__ = ["DashboardConfig"]
