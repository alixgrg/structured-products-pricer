"""Interest-rate swap product."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.products._helpers import normalize_non_negative_float, normalize_positive_float
from src.products.base_product import Product
from src.products.coupon_bond import _coupon_times, _payments_per_year


@dataclass(slots=True)
class InterestRateSwap(Product):
    """Plain vanilla fixed-vs-floating interest-rate swap.

    Convention used here: positive payoff/value means fixed leg minus floating leg.
    Direction can be handled later in the portfolio layer through signed quantity.
    """

    product_id: str
    notional: float
    maturity: float
    fixed_rate: float
    float_index: str
    frequency: str | int = "6M"
    currency: str = "EUR"

    def __post_init__(self) -> None:
        self.notional = normalize_positive_float(self.notional, "notional")
        self.maturity = normalize_non_negative_float(self.maturity, "maturity")
        self.fixed_rate = float(self.fixed_rate)
        self.float_index = str(self.float_index).strip().upper()
        self.currency = str(self.currency).strip().upper()
        _payments_per_year(self.frequency)

    @property
    def payments_per_year(self) -> int:
        return _payments_per_year(self.frequency)

    def payment_times(self) -> list[float]:
        if self.maturity == 0.0:
            return [0.0]
        return _coupon_times(self.maturity, self.payments_per_year)

    def payment_dates(self, valuation_date: str | pd.Timestamp) -> list[pd.Timestamp]:
        start = pd.Timestamp(valuation_date).normalize()
        return [start + pd.DateOffset(days=int(round(t * 365.25))) for t in self.payment_times()]

    def fixed_leg_cash_flows(self, valuation_date: str | pd.Timestamp | None = None) -> list[tuple[Any, float]]:
        dates_or_times: list[Any]
        dates_or_times = self.payment_times() if valuation_date is None else self.payment_dates(valuation_date)
        amount = self.notional * self.fixed_rate / self.payments_per_year
        return [(item, float(amount)) for item in dates_or_times]

    def float_leg_cash_flows(
        self,
        forward_rates: float | list[float] | tuple[float, ...] | dict[Any, float] | None = None,
        valuation_date: str | pd.Timestamp | None = None,
    ) -> list[tuple[Any, float]]:
        dates_or_times: list[Any]
        dates_or_times = self.payment_times() if valuation_date is None else self.payment_dates(valuation_date)
        period = 1.0 / self.payments_per_year

        if forward_rates is None:
            rates = [0.0] * len(dates_or_times)
        elif isinstance(forward_rates, int | float):
            rates = [float(forward_rates)] * len(dates_or_times)
        elif isinstance(forward_rates, dict):
            rates = [float(forward_rates.get(item, 0.0)) for item in dates_or_times]
        else:
            rates = [float(item) for item in forward_rates]
            if len(rates) != len(dates_or_times):
                raise ValueError("forward_rates sequence must have one value per payment date.")

        return [(item, float(self.notional * rate * period)) for item, rate in zip(dates_or_times, rates, strict=True)]

    def payoff(self, market_data=None) -> float:
        """Terminal deterministic payoff proxy: sum fixed coupons minus projected floating coupons."""
        forward_rate = None
        if isinstance(market_data, dict):
            forward_rate = market_data.get("forward_rate", market_data.get("rate"))
        else:
            forward_rate = getattr(market_data, "rate", None)

        fixed_total = sum(amount for _, amount in self.fixed_leg_cash_flows())
        floating_total = sum(amount for _, amount in self.float_leg_cash_flows(forward_rate))
        return float(fixed_total - floating_total)

    def get_risk_factors(self) -> list[str]:
        return ["rate"]
