"""Business-day adjustment helpers.

This is a compact calendar layer. It handles weekends by default and accepts an
optional holiday set. For a classroom project this is enough to make date logic
explicit without pulling in a full market calendar dependency.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date, datetime
from typing import Literal

import pandas as pd

BusinessDayConvention = Literal["following", "modified_following", "preceding", "none"]


class BusinessCalendar:
    """Weekend/holiday calendar with standard business-day adjustments."""

    def __init__(self, holidays: Iterable[date | datetime | str | pd.Timestamp] | None = None) -> None:
        self.holidays = {
            pd.Timestamp(holiday).normalize().date()
            for holiday in (holidays or [])
        }

    def is_business_day(self, value: date | datetime | str | pd.Timestamp) -> bool:
        ts = pd.Timestamp(value).normalize()
        if ts.weekday() >= 5:
            return False
        return ts.date() not in self.holidays

    def adjust(
        self,
        value: date | datetime | str | pd.Timestamp,
        convention: BusinessDayConvention = "modified_following",
    ) -> pd.Timestamp:
        """Adjust a date according to a business-day convention."""
        convention = convention.lower()  # type: ignore[assignment]
        ts = pd.Timestamp(value).normalize()

        if convention == "none":
            return ts
        if self.is_business_day(ts):
            return ts

        if convention == "following":
            return self._following(ts)
        if convention == "preceding":
            return self._preceding(ts)
        if convention == "modified_following":
            following = self._following(ts)
            if following.month != ts.month:
                return self._preceding(ts)
            return following

        raise ValueError("Unsupported business-day convention.")

    def _following(self, value: pd.Timestamp) -> pd.Timestamp:
        ts = value
        while not self.is_business_day(ts):
            ts += pd.DateOffset(days=1)
        return ts

    def _preceding(self, value: pd.Timestamp) -> pd.Timestamp:
        ts = value
        while not self.is_business_day(ts):
            ts -= pd.DateOffset(days=1)
        return ts


def generate_schedule(
    start: date | datetime | str | pd.Timestamp,
    end: date | datetime | str | pd.Timestamp,
    *,
    frequency: str = "1Y",
    calendar: BusinessCalendar | None = None,
    business_day_convention: BusinessDayConvention = "modified_following",
) -> list[pd.Timestamp]:
    """Generate adjusted coupon/payment dates from start excluded to end included."""
    from src.conventions.day_count import Tenor

    cal = calendar or BusinessCalendar()
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    if end_ts <= start_ts:
        raise ValueError("end must be strictly after start.")

    tenor = Tenor.parse(frequency)
    dates: list[pd.Timestamp] = []
    cursor = start_ts

    while True:
        cursor = tenor.add_to(cursor)
        if cursor >= end_ts:
            dates.append(cal.adjust(end_ts, business_day_convention))
            break
        dates.append(cal.adjust(cursor, business_day_convention))

    # Remove duplicates that can occur after modified-following adjustment.
    unique: list[pd.Timestamp] = []
    for item in dates:
        if not unique or item != unique[-1]:
            unique.append(item)
    return unique


__all__ = [
    "BusinessCalendar",
    "BusinessDayConvention",
    "generate_schedule",
]
