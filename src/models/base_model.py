from abc import ABC, abstractmethod


class PricingModel(ABC):
    """Abstract base class for all pricing models."""

    @abstractmethod
    def price(self, product, market_data) -> float:
        """Return the price of a product."""
        raise NotImplementedError

    def risk(self, product, market_data) -> dict:
        """Return risk metrics for a product."""
        return {}