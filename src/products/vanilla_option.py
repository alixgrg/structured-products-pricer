"""European vanilla option product."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.products._helpers import extract_spot
from src.products.base_product import Product


OptionType = Literal["call", "put"]


@dataclass(frozen=True, slots=True)
class VanillaOption(Product):
    """European vanilla call or put option.

    Parameters
    ----------
    product_id:
        Product identifier.
    option_type:
        "call" or "put".
    strike:
        Option strike.
    maturity:
        Maturity in years.
    notional:
        Payoff multiplier. Default is 1.0.
    underlying:
        Underlying identifier.
    dividend_yield:
        Optional continuous dividend yield. If None, the model uses market data.
    """

    product_id: str
    option_type: str
    strike: float
    maturity: float
    notional: float = 1.0
    underlying: str = ""
    dividend_yield: float | None = None

    def __post_init__(self) -> None:
        option_type = self.option_type.lower().strip()

        if option_type in {"c", "call"}:
            option_type = "call"
        elif option_type in {"p", "put"}:
            option_type = "put"
        else:
            raise ValueError("option_type must be 'call' or 'put'.")

        if self.strike <= 0.0:
            raise ValueError("strike must be strictly positive.")
        if self.maturity < 0.0:
            raise ValueError("maturity must be non-negative.")
        if self.notional <= 0.0:
            raise ValueError("notional must be strictly positive.")

        object.__setattr__(self, "option_type", option_type)
        object.__setattr__(self, "underlying", self.underlying.upper())

    def payoff(self, market_data) -> float:
        """Return the option payoff at maturity for a given spot."""
        spot = extract_spot(market_data)

        if self.option_type == "call":
            intrinsic = max(spot - self.strike, 0.0)
        else:
            intrinsic = max(self.strike - spot, 0.0)

        return float(self.notional * intrinsic)

    def get_risk_factors(self) -> list[str]:
        """Return the risk factors used by the option."""
        return ["spot", "rate", "volatility", "dividend_yield"]


