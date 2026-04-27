"""Day-count and tenor helpers used by rates and volatility modules.

The project mostly works with year fractions expressed as floats. This module
adds a date-aware layer so that market instruments can be bootstrapped with
standard conventions rather than hard-coded maturity years.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal

import pandas as pd
from dateutil.relativedelta import relativedelta

DayCountConvention = Literal["ACT/360", "ACT/365F", "ACT/ACT", "30/360", "30E/360"]
TenorUnit = Literal["D", "W", "M", "Y"]


@dataclass(frozen=True, slots=True)
class Tenor:
    """Simple tenor representation, for example 3M, 6M, 1Y."""

    amount: int
    unit: TenorUnit

    @classmethod
    def parse(cls, value: str | int | float) -> "Tenor":
        """Parse a market tenor string.

        Examples
        --------
        >>> Tenor.parse("3M")
        Tenor(amount=3, unit='M')
        >>> Tenor.parse("1Y")
        Tenor(amount=1, unit='Y')
        """
        if isinstance(value, (int, float)):
            # Numeric values are interpreted as years.
            amount = int(round(float(value) * 12.0))
            return cls(amount=amount, unit="M")

        text = str(value).strip().upper().replace(" ", "")
        if not text:
            raise ValueError("Tenor cannot be empty.")

        unit = text[-1]
        if unit not in {"D", "W", "M", "Y"}:
            raise ValueError("Tenor unit must be one of D, W, M, Y.")

        try:
            amount = int(text[:-1])
        except ValueError as exc:
            raise ValueError(f"Invalid tenor: {value!r}") from exc

        if amount <= 0:
            raise ValueError("Tenor amount must be strictly positive.")

        return cls(amount=amount, unit=unit)  # type: ignore[arg-type]

    def add_to(self, start: date | datetime | pd.Timestamp) -> pd.Timestamp:
        """Return start + tenor as a pandas Timestamp."""
        ts = pd.Timestamp(start).normalize()
        if self.unit == "D":
            return ts + pd.DateOffset(days=self.amount)
        if self.unit == "W":
            return ts + pd.DateOffset(weeks=self.amount)
        if self.unit == "M":
            return ts + pd.DateOffset(months=self.amount)
        if self.unit == "Y":
            return ts + pd.DateOffset(years=self.amount)
        raise ValueError(f"Unsupported tenor unit: {self.unit}")

    def years_approx(self) -> float:
        """Return a rough year fraction used only for sorting/display."""
        if self.unit == "D":
            return self.amount / 365.0
        if self.unit == "W":
            return 7.0 * self.amount / 365.0
        if self.unit == "M":
            return self.amount / 12.0
        if self.unit == "Y":
            return float(self.amount)
        raise ValueError(f"Unsupported tenor unit: {self.unit}")


def add_tenor(start: date | datetime | pd.Timestamp, tenor: str | Tenor) -> pd.Timestamp:
    """Add a tenor to a date."""
    parsed = tenor if isinstance(tenor, Tenor) else Tenor.parse(tenor)
    return parsed.add_to(start)


def year_fraction(
    start: date | datetime | pd.Timestamp,
    end: date | datetime | pd.Timestamp,
    convention: DayCountConvention = "ACT/365F",
) -> float:
    """Compute year fraction between two dates.

    Supported conventions are intentionally limited but explicit:
    ACT/360, ACT/365F, ACT/ACT, 30/360 US, 30E/360 European.
    """
    d1 = pd.Timestamp(start).date()
    d2 = pd.Timestamp(end).date()

    if d2 < d1:
        raise ValueError("end date must be greater than or equal to start date.")

    conv = convention.upper()
    days = (d2 - d1).days

    if conv == "ACT/360":
        return days / 360.0
    if conv == "ACT/365F":
        return days / 365.0
    if conv == "ACT/ACT":
        return _year_fraction_act_act(d1, d2)
    if conv == "30/360":
        return _year_fraction_30_360_us(d1, d2)
    if conv == "30E/360":
        return _year_fraction_30e_360(d1, d2)

    raise ValueError("Unsupported day count convention.")


def _year_fraction_act_act(start: date, end: date) -> float:
    if start == end:
        return 0.0

    total = 0.0
    cursor = start
    while cursor < end:
        next_year = date(cursor.year + 1, 1, 1)
        period_end = min(end, next_year)
        denominator = 366.0 if _is_leap_year(cursor.year) else 365.0
        total += (period_end - cursor).days / denominator
        cursor = period_end
    return total


def _year_fraction_30_360_us(start: date, end: date) -> float:
    d1 = min(start.day, 30)
    d2 = end.day
    if d1 == 30 and d2 == 31:
        d2 = 30
    return ((end.year - start.year) * 360 + (end.month - start.month) * 30 + (d2 - d1)) / 360.0


def _year_fraction_30e_360(start: date, end: date) -> float:
    d1 = min(start.day, 30)
    d2 = min(end.day, 30)
    return ((end.year - start.year) * 360 + (end.month - start.month) * 30 + (d2 - d1)) / 360.0


def _is_leap_year(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)

