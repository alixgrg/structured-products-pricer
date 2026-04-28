"""Zero-coupon bond product."""

from __future__ import annotations

from dataclasses import dataclass

from src.products._helpers import normalize_non_negative_float, normalize_positive_float
from src.products.base_product import Product


@dataclass(slots=True)
class ZeroCouponBond(Product):
    product_id: str
    notional: float
    maturity: float
    currency: str = "EUR"

    def __post_init__(self) -> None:
        self.notional = normalize_positive_float(self.notional, "notional")
        self.maturity = normalize_non_negative_float(self.maturity, "maturity")
        self.currency = str(self.currency).strip().upper()

    def payoff(self, market_data=None) -> float:
        return float(self.notional)

    def get_risk_factors(self) -> list[str]:
        return ["rate"]
