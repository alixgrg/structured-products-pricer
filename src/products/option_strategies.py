"""Option strategies represented by static portfolios of vanilla options."""

from __future__ import annotations

from dataclasses import dataclass

from src.products._helpers import normalize_non_negative_float, normalize_positive_float
from src.products.base_product import Product
from src.products.vanilla_option import VanillaOption


@dataclass(slots=True)
class OptionLeg:
    """A signed vanilla-option leg used in static replication."""

    product: VanillaOption
    quantity: float

    def __post_init__(self) -> None:
        self.quantity = float(self.quantity)
        if self.quantity == 0.0:
            raise ValueError("leg quantity cannot be zero.")


@dataclass(slots=True)
class OptionStrategy(Product):
    """Static replication portfolio of signed vanilla-option legs."""

    product_id: str
    maturity: float
    legs: tuple[OptionLeg, ...]
    notional: float = 1.0
    underlying: str = ""
    strategy_type: str = "option_strategy"

    def __post_init__(self) -> None:
        self.maturity = normalize_non_negative_float(self.maturity, "maturity")
        self.notional = normalize_positive_float(self.notional, "notional")
        self.underlying = str(self.underlying).strip().upper()
        if not self.legs:
            raise ValueError("OptionStrategy requires at least one leg.")
        self.legs = tuple(self.legs)

    def get_legs(self) -> list[tuple[VanillaOption, float]]:
        return [(leg.product, float(leg.quantity)) for leg in self.legs]

    def payoff(self, market_data) -> float:
        return float(self.notional * sum(leg.quantity * leg.product.payoff(market_data) for leg in self.legs))

    def price(self, option_model, market_data=None) -> float:
        return float(self.notional * sum(leg.quantity * option_model.price(leg.product, market_data) for leg in self.legs))

    def get_risk_factors(self) -> list[str]:
        return ["spot", "rate", "volatility"]

    @classmethod
    def call_spread(
        cls,
        *,
        product_id: str,
        maturity: float,
        strike_low: float,
        strike_high: float,
        underlying: str = "",
        notional: float = 1.0,
        currency: str = "EUR",
    ) -> "CallSpread":
        """Convenience constructor for a call spread."""
        return CallSpread(product_id, maturity, strike_low, strike_high, underlying, notional, currency)

    @classmethod
    def put_spread(
        cls,
        *,
        product_id: str,
        maturity: float,
        strike_low: float,
        strike_high: float,
        underlying: str = "",
        notional: float = 1.0,
        currency: str = "EUR",
    ) -> "PutSpread":
        """Convenience constructor for a put spread."""
        return PutSpread(product_id, maturity, strike_low, strike_high, underlying, notional, currency)

    @classmethod
    def butterfly(
        cls,
        *,
        product_id: str,
        maturity: float,
        strike_low: float,
        strike_mid: float,
        strike_high: float,
        underlying: str = "",
        notional: float = 1.0,
        currency: str = "EUR",
    ) -> "Butterfly":
        """Convenience constructor for a butterfly strategy."""
        return Butterfly(product_id, maturity, strike_low, strike_mid, strike_high, underlying, notional, currency)

    @classmethod
    def straddle(
        cls,
        *,
        product_id: str,
        maturity: float,
        strike: float,
        underlying: str = "",
        notional: float = 1.0,
        currency: str = "EUR",
    ) -> "Straddle":
        """Convenience constructor for a straddle."""
        return Straddle(product_id, maturity, strike, underlying, notional, currency)


class CallSpread(OptionStrategy):
    """Long call K1 and short call K2, K1 < K2."""

    def __init__(
        self,
        product_id: str,
        maturity: float,
        strike_low: float,
        strike_high: float,
        underlying: str = "",
        notional: float = 1.0,
        currency: str = "EUR",
    ) -> None:
        if strike_low >= strike_high:
            raise ValueError("CallSpread requires strike_low < strike_high.")
        legs = (
            OptionLeg(VanillaOption(f"{product_id}-C-{strike_low:g}", "call", strike_low, maturity, currency=currency, underlying=underlying), 1.0),
            OptionLeg(VanillaOption(f"{product_id}-C-{strike_high:g}", "call", strike_high, maturity, currency=currency, underlying=underlying), -1.0),
        )
        super().__init__(product_id, maturity, legs, notional, underlying, "call_spread")


class PutSpread(OptionStrategy):
    """Long put K_high and short put K_low: bearish put spread."""

    def __init__(
        self,
        product_id: str,
        maturity: float,
        strike_low: float,
        strike_high: float,
        underlying: str = "",
        notional: float = 1.0,
        currency: str = "EUR",
    ) -> None:
        if strike_low >= strike_high:
            raise ValueError("PutSpread requires strike_low < strike_high.")
        legs = (
            OptionLeg(VanillaOption(f"{product_id}-P-{strike_high:g}", "put", strike_high, maturity, currency=currency, underlying=underlying), 1.0),
            OptionLeg(VanillaOption(f"{product_id}-P-{strike_low:g}", "put", strike_low, maturity, currency=currency, underlying=underlying), -1.0),
        )
        super().__init__(product_id, maturity, legs, notional, underlying, "put_spread")


class Butterfly(OptionStrategy):
    """Long call K1, short two calls K2, long call K3."""

    def __init__(
        self,
        product_id: str,
        maturity: float,
        strike_low: float,
        strike_mid: float,
        strike_high: float,
        underlying: str = "",
        notional: float = 1.0,
        currency: str = "EUR",
    ) -> None:
        if not (strike_low < strike_mid < strike_high):
            raise ValueError("Butterfly requires strike_low < strike_mid < strike_high.")
        legs = (
            OptionLeg(VanillaOption(f"{product_id}-C-{strike_low:g}", "call", strike_low, maturity, currency=currency, underlying=underlying), 1.0),
            OptionLeg(VanillaOption(f"{product_id}-C-{strike_mid:g}", "call", strike_mid, maturity, currency=currency, underlying=underlying), -2.0),
            OptionLeg(VanillaOption(f"{product_id}-C-{strike_high:g}", "call", strike_high, maturity, currency=currency, underlying=underlying), 1.0),
        )
        super().__init__(product_id, maturity, legs, notional, underlying, "butterfly")


class Straddle(OptionStrategy):
    """Long call and long put with same strike and maturity."""

    def __init__(
        self,
        product_id: str,
        maturity: float,
        strike: float,
        underlying: str = "",
        notional: float = 1.0,
        currency: str = "EUR",
    ) -> None:
        legs = (
            OptionLeg(VanillaOption(f"{product_id}-C-{strike:g}", "call", strike, maturity, currency=currency, underlying=underlying), 1.0),
            OptionLeg(VanillaOption(f"{product_id}-P-{strike:g}", "put", strike, maturity, currency=currency, underlying=underlying), 1.0),
        )
        super().__init__(product_id, maturity, legs, notional, underlying, "straddle")


__all__ = ["OptionLeg", "OptionStrategy", "CallSpread", "PutSpread", "Butterfly", "Straddle"]
