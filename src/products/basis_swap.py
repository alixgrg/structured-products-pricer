"""Basis-swap product.

A basis swap exchanges two floating legs, usually with different tenors
(for example receive Euribor 6M and pay Euribor 3M).

Convention used in this project:
    positive value = PV(receive floating leg) - PV(pay floating leg)
                     + PV(spread leg paid on the receive schedule)

The sign of the portfolio position is handled outside the product, in the
portfolio pricing engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.products._helpers import normalize_non_negative_float, normalize_positive_float
from src.products.base_product import Product
from src.products.coupon_bond import _coupon_times, _payments_per_year


def _frequency_from_index(index: str | None, default: str = "6M") -> str:
    """Infer payment frequency from an index label such as EURIBOR6M or 3M."""
    if index is None:
        return default

    text = str(index).strip().upper().replace(" ", "")
    for token in ("12M", "6M", "3M", "1M", "1Y"):
        if token in text:
            return token
    return default


@dataclass(slots=True)
class BasisSwap(Product):
    """Plain vanilla floating-vs-floating basis swap.

    Parameters
    ----------
    receive_index:
        Floating index received, for example ``"6M"`` or ``"EURIBOR6M"``.
    pay_index:
        Floating index paid, for example ``"3M"`` or ``"EURIBOR3M"``.
    spread:
        Annualized spread added to the received leg. A positive spread increases
        the value of the product under the convention receive - pay.
    """

    product_id: str
    notional: float
    maturity: float
    receive_index: str
    pay_index: str
    receive_frequency: str | int | None = None
    pay_frequency: str | int | None = None
    spread: float = 0.0
    currency: str = "EUR"

    def __post_init__(self) -> None:
        self.notional = normalize_positive_float(self.notional, "notional")
        self.maturity = normalize_non_negative_float(self.maturity, "maturity")
        self.receive_index = str(self.receive_index).strip().upper()
        self.pay_index = str(self.pay_index).strip().upper()
        self.currency = str(self.currency).strip().upper()
        self.spread = float(self.spread)

        if self.receive_frequency is None:
            self.receive_frequency = _frequency_from_index(self.receive_index, default="6M")
        if self.pay_frequency is None:
            self.pay_frequency = _frequency_from_index(self.pay_index, default="3M")

        _payments_per_year(self.receive_frequency)
        _payments_per_year(self.pay_frequency)

    @property
    def receive_payments_per_year(self) -> int:
        return _payments_per_year(self.receive_frequency)

    @property
    def pay_payments_per_year(self) -> int:
        return _payments_per_year(self.pay_frequency)

    def receive_payment_times(self) -> list[float]:
        if self.maturity == 0.0:
            return [0.0]
        return _coupon_times(self.maturity, self.receive_payments_per_year)

    def pay_payment_times(self) -> list[float]:
        if self.maturity == 0.0:
            return [0.0]
        return _coupon_times(self.maturity, self.pay_payments_per_year)

    def payment_dates(
        self,
        valuation_date: str | pd.Timestamp,
        *,
        leg: str = "receive",
    ) -> list[pd.Timestamp]:
        start = pd.Timestamp(valuation_date).normalize()
        times = self.receive_payment_times() if leg == "receive" else self.pay_payment_times()
        return [start + pd.DateOffset(days=int(round(t * 365.25))) for t in times]

    def receive_leg_cash_flows(
        self,
        forward_rates: float | list[float] | tuple[float, ...] | dict[Any, float] | None = None,
    ) -> list[tuple[float, float]]:
        return self._floating_leg_cash_flows(
            self.receive_payment_times(),
            forward_rates=forward_rates,
            spread=self.spread,
        )

    def pay_leg_cash_flows(
        self,
        forward_rates: float | list[float] | tuple[float, ...] | dict[Any, float] | None = None,
    ) -> list[tuple[float, float]]:
        return self._floating_leg_cash_flows(
            self.pay_payment_times(),
            forward_rates=forward_rates,
            spread=0.0,
        )

    def _floating_leg_cash_flows(
        self,
        payment_times: list[float],
        *,
        forward_rates: float | list[float] | tuple[float, ...] | dict[Any, float] | None,
        spread: float,
    ) -> list[tuple[float, float]]:
        previous = 0.0
        cash_flows: list[tuple[float, float]] = []

        if forward_rates is None:
            rates = [0.0] * len(payment_times)
        elif isinstance(forward_rates, int | float):
            rates = [float(forward_rates)] * len(payment_times)
        elif isinstance(forward_rates, dict):
            rates = [float(forward_rates.get(t, 0.0)) for t in payment_times]
        else:
            rates = [float(x) for x in forward_rates]
            if len(rates) != len(payment_times):
                raise ValueError("forward_rates length must match payment_times length.")

        for t, rate in zip(payment_times, rates, strict=True):
            accrual = max(float(t) - previous, 0.0)
            amount = self.notional * (float(rate) + spread) * accrual
            cash_flows.append((float(t), float(amount)))
            previous = float(t)

        return cash_flows

    def payoff(self, market_data=None) -> float:
        return 0.0

    def get_risk_factors(self) -> list[str]:
        return ["rate"]
