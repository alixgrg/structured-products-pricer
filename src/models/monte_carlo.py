"""Monte Carlo GBM pricing model.

The simulator uses the risk-neutral Black-Scholes dynamics with continuous
dividend yield:

    S[t+dt] = S[t] * exp((r - q - 0.5*sigma^2)dt + sigma*sqrt(dt)*Z)

It returns a price and a 95% confidence interval through ``price_with_error``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, sqrt
from typing import Any

import numpy as np

from src.market.market_data import MarketData
from src.models.base_model import PricingModel
from src.models.pricing_inputs import require_market_spot, resolve_dividend_yield, resolve_pricing_rate, resolve_pricing_volatility
from src.products.autocall import AutocallProduct
from src.products.barrier_option import BarrierOption
from src.products.vanilla_option import VanillaOption
from src.rates.yield_curve import YieldCurve


@dataclass(frozen=True, slots=True)
class MonteCarloResult:
    price: float
    standard_error: float
    confidence_interval_low: float
    confidence_interval_high: float
    n_paths: int
    confidence_level: float = 0.95

    def to_dict(self) -> dict[str, float]:
        return {
            "price": self.price,
            "standard_error": self.standard_error,
            "confidence_interval_low": self.confidence_interval_low,
            "confidence_interval_high": self.confidence_interval_high,
            "n_paths": float(self.n_paths),
            "confidence_level": self.confidence_level,
        }


@dataclass(frozen=True, slots=True)
class MonteCarloGBMModel(PricingModel):
    """Risk-neutral Monte Carlo pricer for vanilla and path-dependent products."""

    n_paths: int = 50_000
    n_steps: int = 252
    seed: int | None = 42
    antithetic: bool = True
    rate: float | None = None
    volatility: float | None = None
    dividend_yield: float = 0.0
    yield_curve: YieldCurve | None = None
    confidence_level: float = 0.95

    def __post_init__(self) -> None:
        if self.n_paths <= 0:
            raise ValueError("n_paths must be strictly positive.")
        if self.n_steps <= 0:
            raise ValueError("n_steps must be strictly positive.")
        if not (0.0 < self.confidence_level < 1.0):
            raise ValueError("confidence_level must be between 0 and 1.")

    def price(self, product, market_data: MarketData | None = None) -> float:
        return self.price_with_error(product, market_data).price

    def price_with_error(self, product, market_data: MarketData | None = None) -> MonteCarloResult:
        maturity = float(getattr(product, "maturity", 0.0))
        if maturity < 0.0:
            raise ValueError("product.maturity must be non-negative.")

        spot = require_market_spot(market_data)
        rate = self._resolve_rate(maturity, market_data)
        volatility = self._resolve_volatility(market_data)
        dividend_yield = self._resolve_dividend_yield(product, market_data)

        if maturity == 0.0:
            payoff = float(product.payoff({"spot": spot}))
            return MonteCarloResult(payoff, 0.0, payoff, payoff, 1, self.confidence_level)

        paths = self.simulate_paths(
            spot=spot,
            maturity=maturity,
            rate=rate,
            volatility=volatility,
            dividend_yield=dividend_yield,
        )
        payoffs = self._payoffs(product, paths)
        discounted = np.exp(-rate * maturity) * payoffs

        price = float(np.mean(discounted))
        standard_error = float(np.std(discounted, ddof=1) / np.sqrt(len(discounted))) if len(discounted) > 1 else 0.0
        z = _normal_z_value(self.confidence_level)
        return MonteCarloResult(
            price=price,
            standard_error=standard_error,
            confidence_interval_low=float(price - z * standard_error),
            confidence_interval_high=float(price + z * standard_error),
            n_paths=int(len(discounted)),
            confidence_level=self.confidence_level,
        )

    def risk(self, product, market_data: MarketData | None = None) -> dict[str, float]:
        result = self.price_with_error(product, market_data)
        return {
            "price": result.price,
            "standard_error": result.standard_error,
            "ci_low": result.confidence_interval_low,
            "ci_high": result.confidence_interval_high,
        }

    def simulate_paths(
        self,
        *,
        spot: float,
        maturity: float,
        rate: float,
        volatility: float,
        dividend_yield: float = 0.0,
    ) -> np.ndarray:
        if spot <= 0.0:
            raise ValueError("spot must be strictly positive.")
        if maturity <= 0.0:
            raise ValueError("maturity must be strictly positive.")
        if volatility <= 0.0:
            raise ValueError("volatility must be strictly positive.")

        n_paths = int(self.n_paths)
        rng = np.random.default_rng(self.seed)
        if self.antithetic:
            half = int(np.ceil(n_paths / 2.0))
            z_half = rng.standard_normal(size=(half, self.n_steps))
            z = np.concatenate([z_half, -z_half], axis=0)[:n_paths]
        else:
            z = rng.standard_normal(size=(n_paths, self.n_steps))

        dt = maturity / self.n_steps
        increments = (rate - dividend_yield - 0.5 * volatility * volatility) * dt + volatility * sqrt(dt) * z
        log_paths = np.cumsum(increments, axis=1)
        paths = np.empty((n_paths, self.n_steps + 1), dtype=float)
        paths[:, 0] = spot
        paths[:, 1:] = spot * np.exp(log_paths)
        return paths

    def _payoffs(self, product: Any, paths: np.ndarray) -> np.ndarray:
        if isinstance(product, VanillaOption):
            terminal = paths[:, -1]
            if product.option_type == "call":
                return product.notional * np.maximum(terminal - product.strike, 0.0)
            return product.notional * np.maximum(product.strike - terminal, 0.0)

        if isinstance(product, BarrierOption):
            touched = np.max(paths, axis=1) >= product.barrier if product.barrier_direction == "up" else np.min(paths, axis=1) <= product.barrier
            terminal = paths[:, -1]
            if product.option_type == "call":
                vanilla = product.notional * np.maximum(terminal - product.strike, 0.0)
            else:
                vanilla = product.notional * np.maximum(product.strike - terminal, 0.0)
            active = ~touched if product.is_knock_out else touched
            return np.where(active, vanilla, 0.0)

        if isinstance(product, AutocallProduct):
            observation_path = self._autocall_observation_paths(product, paths)
            return np.asarray([product.payoff({"path": row}) for row in observation_path], dtype=float)

        # Generic fallback for future path-dependent products.
        return np.asarray([float(product.payoff({"path": row})) for row in paths], dtype=float)

    def _autocall_observation_paths(self, product: AutocallProduct, paths: np.ndarray) -> np.ndarray:
        maturity = float(product.maturity)
        if maturity <= 0.0:
            raise ValueError("Autocall maturity must be strictly positive.")
        obs = list(product.observation_dates)
        if not obs:
            raise ValueError("Autocall observation_dates cannot be empty.")
        indices: list[int] = []
        for i, item in enumerate(obs, start=1):
            try:
                obs_time = float(item)
            except (TypeError, ValueError):
                obs_time = maturity * i / len(obs)
            obs_time = min(max(obs_time, 1e-12), maturity)
            index = int(round(obs_time / maturity * self.n_steps))
            index = min(max(index, 1), self.n_steps)
            indices.append(index)
        return paths[:, indices]


    def _resolve_rate(self, maturity: float, market_data: MarketData | None) -> float:
        return resolve_pricing_rate(
            maturity=maturity,
            yield_curve=self.yield_curve,
            model_rate=self.rate,
            market_data=market_data,
        )

    def _resolve_volatility(self, market_data: MarketData | None) -> float:
        return resolve_pricing_volatility(
            model_volatility=self.volatility,
            market_data=market_data,
        )

    def _resolve_dividend_yield(self, product, market_data: MarketData | None) -> float:
        return resolve_dividend_yield(
            product_dividend_yield=getattr(product, "dividend_yield", None),
            model_dividend_yield=self.dividend_yield,
            market_data=market_data,
        )


def _normal_z_value(confidence_level: float) -> float:
    # Exact for the default and common in finance; avoids requiring scipy just for ppf.
    if abs(confidence_level - 0.95) < 1e-12:
        return 1.959963984540054
    if abs(confidence_level - 0.99) < 1e-12:
        return 2.5758293035489004
    if abs(confidence_level - 0.90) < 1e-12:
        return 1.6448536269514722
    return 1.959963984540054


__all__ = ["MonteCarloGBMModel", "MonteCarloResult"]
