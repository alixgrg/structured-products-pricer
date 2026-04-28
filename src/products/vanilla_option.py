"""European vanilla option product."""

from __future__ import annotations

from dataclasses import dataclass

from src.products._helpers import extract_spot, normalize_non_negative_float, normalize_positive_float
from src.products.base_product import Product


@dataclass(slots=True)
class VanillaOption(Product):
    """European vanilla call/put used as building block for static replication."""

    product_id: str
    option_type: str
    strike: float
    maturity: float
    notional: float = 1.0
    underlying: str = ""
    currency: str = "EUR"
    dividend_yield: float | None = None

    def __post_init__(self) -> None:
        option_type = str(self.option_type).strip().lower()
        if option_type in {"c", "call"}:
            option_type = "call"
        elif option_type in {"p", "put"}:
            option_type = "put"
        else:
            raise ValueError("option_type must be 'call' or 'put'.")

        self.option_type = option_type
        self.strike = normalize_positive_float(self.strike, "strike")
        self.maturity = normalize_non_negative_float(self.maturity, "maturity")
        self.notional = normalize_positive_float(self.notional, "notional")
        self.underlying = str(self.underlying).strip().upper()
        self.currency = str(self.currency).strip().upper()
        if self.dividend_yield is not None:
            self.dividend_yield = float(self.dividend_yield)

    def payoff(self, market_data) -> float:
        spot = extract_spot(market_data)
        if self.option_type == "call":
            return float(self.notional * max(spot - self.strike, 0.0))
        return float(self.notional * max(self.strike - spot, 0.0))

    def get_risk_factors(self) -> list[str]:
        return ["spot", "rate", "volatility"]
