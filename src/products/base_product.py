from abc import ABC, abstractmethod


class Product(ABC):
    """Abstract base class for all financial products."""

    def __init__(self, product_id: str, notional: float, maturity: float):
        self.product_id = product_id
        self.notional = notional
        self.maturity = maturity

    @abstractmethod
    def payoff(self, market_data):
        """Return the product payoff."""
        raise NotImplementedError

    @abstractmethod
    def get_risk_factors(self) -> list[str]:
        """Return the list of risk factors used by the product."""
        raise NotImplementedError