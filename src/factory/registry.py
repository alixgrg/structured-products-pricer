"""Simple registry for future product builders."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass(slots=True)
class ProductFactoryRegistry:
    """Register product builders by product type."""

    builders: dict[str, Callable[..., object]] = field(default_factory=dict)

    def register(self, product_type: str, builder: Callable[..., object]) -> None:
        self.builders[product_type.lower()] = builder

    def build(self, product_type: str, **kwargs) -> object:
        key = product_type.lower()
        if key not in self.builders:
            raise KeyError(f"Unknown product type: {product_type}")
        return self.builders[key](**kwargs)


__all__ = ["ProductFactoryRegistry"]
