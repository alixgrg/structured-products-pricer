"""Zero-coupon bond product."""

from __future__ import annotations

from dataclasses import dataclass

from src.products.base_product import Product


@dataclass(frozen=True, slots=True)
class ZeroCouponBond(Product):
    """Zero-coupon bond paying its notional at maturity.

    Parameters
    ----------
    product_id:
        Product identifier.
    notional:
        Redemption amount paid at maturity.
    maturity:
        Maturity in years.
    currency:
        Bond currency.
    """

    product_id: str
    notional: float
    maturity: float
    currency: str = "EUR"

    def __post_init__(self) -> None:
        if self.notional <= 0.0:
            raise ValueError("notional must be strictly positive.")
        if self.maturity < 0.0:
            raise ValueError("maturity must be non-negative.")
        object.__setattr__(self, "currency", self.currency.upper())

    def payoff(self, market_data=None) -> float:
        """Return the redemption payoff at maturity."""
        return float(self.notional)

    def get_risk_factors(self) -> list[str]:
        """Return the risk factors used by the product."""
        return ["rate"]