"""Structured note products built from bonds and vanilla options."""

from __future__ import annotations

from dataclasses import dataclass

from src.products.base_product import Product
from src.products.vanilla_option import VanillaOption
from src.products.zero_coupon_bond import ZeroCouponBond


@dataclass(frozen=True, slots=True)
class StructuredNoteLeg:
    """One elementary building block of a structured note."""

    product: Product
    quantity: float = 1.0
    label: str = ""


@dataclass(frozen=True, slots=True)
class CapitalProtectedNote(Product):
    """Capital-protected note: bond + long call participation."""

    product_id: str
    notional: float
    maturity: float
    spot_reference: float
    participation_rate: float = 1.0
    currency: str = "EUR"

    def __post_init__(self) -> None:
        if self.notional <= 0.0:
            raise ValueError("notional must be strictly positive.")
        if self.maturity < 0.0:
            raise ValueError("maturity must be non-negative.")
        if self.spot_reference <= 0.0:
            raise ValueError("spot_reference must be strictly positive.")
        if self.participation_rate < 0.0:
            raise ValueError("participation_rate must be non-negative.")

        object.__setattr__(self, "currency", self.currency.upper())

    def decomposition(self) -> tuple[StructuredNoteLeg, ...]:
        bond = ZeroCouponBond(
            product_id=f"{self.product_id}-BOND",
            notional=self.notional,
            maturity=self.maturity,
            currency=self.currency,
        )
        call = VanillaOption(
            product_id=f"{self.product_id}-CALL",
            option_type="call",
            strike=self.spot_reference,
            maturity=self.maturity,
            notional=1.0,
        )
        call_qty = self.participation_rate * self.notional / self.spot_reference

        return (
            StructuredNoteLeg(bond, quantity=1.0, label="capital_protection_bond"),
            StructuredNoteLeg(call, quantity=call_qty, label="upside_call"),
        )

    def payoff(self, market_data) -> float:
        return float(sum(leg.quantity * leg.product.payoff(market_data) for leg in self.decomposition()))

    def price(self, option_model, discount_model, market_data) -> float:
        return _price_decomposition(self.decomposition(), option_model, discount_model, market_data)

    def get_risk_factors(self) -> list[str]:
        return ["spot", "rate", "volatility"]


@dataclass(frozen=True, slots=True)
class CappedCapitalProtectedNote(Product):
    """Capital-protected note with capped upside."""

    product_id: str
    notional: float
    maturity: float
    spot_reference: float
    cap_level: float
    participation_rate: float = 1.0
    currency: str = "EUR"

    def __post_init__(self) -> None:
        if self.notional <= 0.0:
            raise ValueError("notional must be strictly positive.")
        if self.maturity < 0.0:
            raise ValueError("maturity must be non-negative.")
        if self.spot_reference <= 0.0:
            raise ValueError("spot_reference must be strictly positive.")
        if self.participation_rate < 0.0:
            raise ValueError("participation_rate must be non-negative.")
        if self.cap_level <= 1.0:
            raise ValueError("cap_level must be strictly greater than 1.0.")

        object.__setattr__(self, "currency", self.currency.upper())

    def decomposition(self) -> tuple[StructuredNoteLeg, ...]:
        bond = ZeroCouponBond(
            product_id=f"{self.product_id}-BOND",
            notional=self.notional,
            maturity=self.maturity,
            currency=self.currency,
        )
        strike_low = self.spot_reference
        strike_high = self.spot_reference * self.cap_level

        long_call = VanillaOption(
            product_id=f"{self.product_id}-CALL-L",
            option_type="call",
            strike=strike_low,
            maturity=self.maturity,
            notional=1.0,
        )
        short_call = VanillaOption(
            product_id=f"{self.product_id}-CALL-S",
            option_type="call",
            strike=strike_high,
            maturity=self.maturity,
            notional=1.0,
        )

        call_qty = self.participation_rate * self.notional / self.spot_reference

        return (
            StructuredNoteLeg(bond, quantity=1.0, label="capital_protection_bond"),
            StructuredNoteLeg(long_call, quantity=call_qty, label="long_call_k1"),
            StructuredNoteLeg(short_call, quantity=-call_qty, label="short_call_k2"),
        )

    def payoff(self, market_data) -> float:
        return float(sum(leg.quantity * leg.product.payoff(market_data) for leg in self.decomposition()))

    def price(self, option_model, discount_model, market_data) -> float:
        return _price_decomposition(self.decomposition(), option_model, discount_model, market_data)

    def get_risk_factors(self) -> list[str]:
        return ["spot", "rate", "volatility"]


@dataclass(frozen=True, slots=True)
class ReverseConvertible(Product):
    """Reverse convertible: enhanced coupon with downside put exposure."""

    product_id: str
    notional: float
    maturity: float
    spot_reference: float
    coupon_rate: float
    currency: str = "EUR"

    def __post_init__(self) -> None:
        if self.notional <= 0.0:
            raise ValueError("notional must be strictly positive.")
        if self.maturity < 0.0:
            raise ValueError("maturity must be non-negative.")
        if self.spot_reference <= 0.0:
            raise ValueError("spot_reference must be strictly positive.")
        if self.coupon_rate < 0.0:
            raise ValueError("coupon_rate must be non-negative.")

        object.__setattr__(self, "currency", self.currency.upper())

    def decomposition(self) -> tuple[StructuredNoteLeg, ...]:
        bond = ZeroCouponBond(
            product_id=f"{self.product_id}-BOND",
            notional=self.notional * (1.0 + self.coupon_rate),
            maturity=self.maturity,
            currency=self.currency,
        )
        put = VanillaOption(
            product_id=f"{self.product_id}-PUT-SHORT",
            option_type="put",
            strike=self.spot_reference,
            maturity=self.maturity,
            notional=1.0,
        )
        put_qty = self.notional / self.spot_reference

        return (
            StructuredNoteLeg(bond, quantity=1.0, label="enhanced_coupon_bond"),
            StructuredNoteLeg(put, quantity=-put_qty, label="short_put"),
        )

    def payoff(self, market_data) -> float:
        return float(sum(leg.quantity * leg.product.payoff(market_data) for leg in self.decomposition()))

    def price(self, option_model, discount_model, market_data) -> float:
        return _price_decomposition(self.decomposition(), option_model, discount_model, market_data)

    def get_risk_factors(self) -> list[str]:
        return ["spot", "rate", "volatility"]


def _price_decomposition(legs, option_model, discount_model, market_data) -> float:
    total = 0.0
    for leg in legs:
        if isinstance(leg.product, VanillaOption):
            leg_price = option_model.price(leg.product, market_data)
        elif isinstance(leg.product, ZeroCouponBond):
            leg_price = discount_model.price(leg.product, market_data)
        else:
            raise TypeError(f"Unsupported leg type: {type(leg.product)!r}")

        total += leg.quantity * leg_price

    return float(total)


__all__ = [
    "CapitalProtectedNote",
    "CappedCapitalProtectedNote",
    "ReverseConvertible",
    "StructuredNoteLeg",
]
