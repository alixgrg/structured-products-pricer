"""Composed vanilla option strategies.

Strategies are modeled as linear combinations of vanilla option legs.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.products.base_product import Product
from src.products.vanilla_option import VanillaOption


@dataclass(frozen=True, slots=True)
class OptionStrategyLeg:
    """One elementary option leg with a signed quantity."""

    product: VanillaOption
    quantity: float = 1.0
    label: str = ""

    def __post_init__(self) -> None:
        if self.quantity == 0.0:
            raise ValueError("Leg quantity must be non-zero.")


@dataclass(frozen=True, slots=True)
class OptionStrategy(Product):
    """Linear option strategy built from vanilla option legs."""

    product_id: str
    maturity: float
    legs: tuple[OptionStrategyLeg, ...]
    notional: float = 1.0

    def __post_init__(self) -> None:
        if self.maturity < 0.0:
            raise ValueError("maturity must be non-negative.")
        if self.notional <= 0.0:
            raise ValueError("notional must be strictly positive.")
        if not self.legs:
            raise ValueError("An option strategy must contain at least one leg.")

        for leg in self.legs:
            if abs(leg.product.maturity - self.maturity) > 1e-12:
                raise ValueError("All legs must share the same maturity as the strategy.")

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
    ) -> "OptionStrategy":
        if strike_high <= strike_low:
            raise ValueError("strike_high must be strictly greater than strike_low.")

        long_call = VanillaOption(
            product_id=f"{product_id}-C-LONG",
            option_type="call",
            strike=strike_low,
            maturity=maturity,
            notional=1.0,
            underlying=underlying,
        )
        short_call = VanillaOption(
            product_id=f"{product_id}-C-SHORT",
            option_type="call",
            strike=strike_high,
            maturity=maturity,
            notional=1.0,
            underlying=underlying,
        )
        return cls(
            product_id=product_id,
            maturity=maturity,
            notional=notional,
            legs=(
                OptionStrategyLeg(long_call, quantity=1.0, label="long_call_k1"),
                OptionStrategyLeg(short_call, quantity=-1.0, label="short_call_k2"),
            ),
        )

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
    ) -> "OptionStrategy":
        if strike_high <= strike_low:
            raise ValueError("strike_high must be strictly greater than strike_low.")

        long_put = VanillaOption(
            product_id=f"{product_id}-P-LONG",
            option_type="put",
            strike=strike_high,
            maturity=maturity,
            notional=1.0,
            underlying=underlying,
        )
        short_put = VanillaOption(
            product_id=f"{product_id}-P-SHORT",
            option_type="put",
            strike=strike_low,
            maturity=maturity,
            notional=1.0,
            underlying=underlying,
        )
        return cls(
            product_id=product_id,
            maturity=maturity,
            notional=notional,
            legs=(
                OptionStrategyLeg(long_put, quantity=1.0, label="long_put_k2"),
                OptionStrategyLeg(short_put, quantity=-1.0, label="short_put_k1"),
            ),
        )

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
    ) -> "OptionStrategy":
        if not (strike_low < strike_mid < strike_high):
            raise ValueError("Butterfly strikes must satisfy strike_low < strike_mid < strike_high.")

        call_low = VanillaOption(
            product_id=f"{product_id}-C-K1",
            option_type="call",
            strike=strike_low,
            maturity=maturity,
            notional=1.0,
            underlying=underlying,
        )
        call_mid = VanillaOption(
            product_id=f"{product_id}-C-K2",
            option_type="call",
            strike=strike_mid,
            maturity=maturity,
            notional=1.0,
            underlying=underlying,
        )
        call_high = VanillaOption(
            product_id=f"{product_id}-C-K3",
            option_type="call",
            strike=strike_high,
            maturity=maturity,
            notional=1.0,
            underlying=underlying,
        )
        return cls(
            product_id=product_id,
            maturity=maturity,
            notional=notional,
            legs=(
                OptionStrategyLeg(call_low, quantity=1.0, label="long_call_k1"),
                OptionStrategyLeg(call_mid, quantity=-2.0, label="short_two_calls_k2"),
                OptionStrategyLeg(call_high, quantity=1.0, label="long_call_k3"),
            ),
        )

    def payoff(self, market_data) -> float:
        return float(
            self.notional
            * sum(leg.quantity * leg.product.payoff(market_data) for leg in self.legs)
        )

    def price(self, option_model, market_data) -> float:
        """Price the strategy as the signed sum of leg prices."""
        return float(
            self.notional
            * sum(
                leg.quantity * option_model.price(leg.product, market_data)
                for leg in self.legs
            )
        )

    def decomposition(self) -> list[dict[str, float | str]]:
        """Return a readable decomposition of legs."""
        rows: list[dict[str, float | str]] = []
        for leg in self.legs:
            rows.append(
                {
                    "label": leg.label or leg.product.product_id,
                    "product_id": leg.product.product_id,
                    "option_type": leg.product.option_type,
                    "strike": leg.product.strike,
                    "quantity": leg.quantity,
                }
            )
        return rows

    def get_risk_factors(self) -> list[str]:
        return ["spot", "rate", "volatility", "dividend_yield"]


__all__ = ["OptionStrategy", "OptionStrategyLeg"]
