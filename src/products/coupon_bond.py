"""Fixed coupon bond product."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.products._helpers import normalize_non_negative_float, normalize_positive_float
from src.products.base_product import Product


def _payments_per_year(frequency: str | int | float) -> int:
    """Infer number of coupons per year from an integer or a tenor-like string."""
    if isinstance(frequency, int | float):
        freq = int(frequency)
        if freq <= 0:
            raise ValueError("frequency must be strictly positive.")
        return freq

    text = str(frequency).strip().upper().replace(" ", "")
    mapping = {"1Y": 1, "12M": 1, "6M": 2, "3M": 4, "1Q": 4, "Q": 4, "1M": 12, "M": 12}
    if text in mapping:
        return mapping[text]
    if text.endswith("M"):
        months = int(text[:-1])
        if months <= 0 or 12 % months != 0:
            raise ValueError("monthly frequency must divide 12, e.g. 1M, 3M, 6M, 12M.")
        return 12 // months
    if text.endswith("Y"):
        years = int(text[:-1])
        if years <= 0:
            raise ValueError("yearly frequency tenor must be positive.")
        # One coupon every N years. Keep at least one payment per year-equivalent schedule.
        return 1
    raise ValueError("frequency must be an int or a tenor such as '1Y', '6M', '3M'.")


def _coupon_times(maturity: float, payments_per_year: int) -> list[float]:
    n_payments = max(int(round(maturity * payments_per_year)), 1)
    step = 1.0 / payments_per_year
    times = [min((i + 1) * step, maturity) for i in range(n_payments)]
    if times[-1] != maturity:
        times[-1] = maturity
    return times


@dataclass(slots=True)
class CouponBond(Product):
    """Fixed-rate coupon bond.

    The product stores maturity as a year fraction for consistency with the rest of the
    project. ``get_cash_flows()`` therefore returns payment times by default. If a
    valuation date is provided, it returns calendar dates instead.
    """

    product_id: str
    notional: float
    maturity: float
    coupon_rate: float
    frequency: str | int = 1
    currency: str = "EUR"

    def __post_init__(self) -> None:
        self.notional = normalize_positive_float(self.notional, "notional")
        self.maturity = normalize_non_negative_float(self.maturity, "maturity")
        self.coupon_rate = float(self.coupon_rate)
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

    def get_cash_flows(self, valuation_date: str | pd.Timestamp | None = None) -> list[tuple[Any, float]]:
        dates_or_times: list[Any]
        dates_or_times = self.payment_times() if valuation_date is None else self.payment_dates(valuation_date)
        coupon = self.notional * self.coupon_rate / self.payments_per_year

        cash_flows = [(item, float(coupon)) for item in dates_or_times]
        if cash_flows:
            last_date, last_amount = cash_flows[-1]
            cash_flows[-1] = (last_date, float(last_amount + self.notional))
        return cash_flows

    def payoff(self, market_data=None) -> float:
        """Principal repayment at maturity, as requested in the project roadmap."""
        return float(self.notional)

    def get_risk_factors(self) -> list[str]:
        return ["rate"]
