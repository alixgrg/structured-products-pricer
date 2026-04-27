"""Market rate instruments used for curve bootstrapping."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.conventions.business_day import BusinessCalendar, BusinessDayConvention, generate_schedule
from src.conventions.day_count import DayCountConvention, add_tenor, year_fraction


@dataclass(frozen=True, slots=True)
class DepositQuote:
    """Money-market deposit quote.

    The quote is interpreted as a simply-compounded annualized rate over
    [spot_date, maturity_date].
    """

    tenor: str
    rate: float
    day_count: DayCountConvention = "ACT/360"
    business_day_convention: BusinessDayConvention = "modified_following"

    def maturity_date(self, valuation_date: pd.Timestamp, calendar: BusinessCalendar) -> pd.Timestamp:
        raw = add_tenor(valuation_date, self.tenor)
        return calendar.adjust(raw, self.business_day_convention)


@dataclass(frozen=True, slots=True)
class FRAQuote:
    """Forward-rate agreement quote for a future money-market period."""

    start_tenor: str
    end_tenor: str
    rate: float
    day_count: DayCountConvention = "ACT/360"
    business_day_convention: BusinessDayConvention = "modified_following"

    def start_date(self, valuation_date: pd.Timestamp, calendar: BusinessCalendar) -> pd.Timestamp:
        return calendar.adjust(add_tenor(valuation_date, self.start_tenor), self.business_day_convention)

    def end_date(self, valuation_date: pd.Timestamp, calendar: BusinessCalendar) -> pd.Timestamp:
        return calendar.adjust(add_tenor(valuation_date, self.end_tenor), self.business_day_convention)


@dataclass(frozen=True, slots=True)
class SwapQuote:
    """Fixed-for-floating par swap quote.

    We bootstrap from the standard par-swap identity, assuming the floating leg is
    worth par at inception: fixed_rate * annuity + P(0,T) = 1.
    """

    maturity_tenor: str
    fixed_rate: float
    fixed_frequency: str = "1Y"
    fixed_day_count: DayCountConvention = "30/360"
    business_day_convention: BusinessDayConvention = "modified_following"

    def maturity_date(self, valuation_date: pd.Timestamp, calendar: BusinessCalendar) -> pd.Timestamp:
        return calendar.adjust(add_tenor(valuation_date, self.maturity_tenor), self.business_day_convention)

    def fixed_leg_schedule(self, valuation_date: pd.Timestamp, calendar: BusinessCalendar) -> list[pd.Timestamp]:
        return generate_schedule(
            valuation_date,
            self.maturity_date(valuation_date, calendar),
            frequency=self.fixed_frequency,
            calendar=calendar,
            business_day_convention=self.business_day_convention,
        )

    def accruals(self, valuation_date: pd.Timestamp, calendar: BusinessCalendar) -> list[float]:
        dates = [pd.Timestamp(valuation_date).normalize()] + self.fixed_leg_schedule(valuation_date, calendar)
        return [
            year_fraction(dates[i - 1], dates[i], self.fixed_day_count)
            for i in range(1, len(dates))
        ]


@dataclass(frozen=True, slots=True)
class BootstrapMarket:
    """Container for all curve instruments used in one bootstrap."""

    valuation_date: pd.Timestamp
    deposits: tuple[DepositQuote, ...] = ()
    fras: tuple[FRAQuote, ...] = ()
    swaps: tuple[SwapQuote, ...] = ()
    calendar: BusinessCalendar = BusinessCalendar()
    name: str = "bootstrapped_curve"


__all__ = [
    "BootstrapMarket",
    "DepositQuote",
    "FRAQuote",
    "SwapQuote",
]
