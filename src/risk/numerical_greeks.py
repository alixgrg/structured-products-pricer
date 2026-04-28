"""Universal numerical Greeks by bump-and-reprice."""

from __future__ import annotations

from dataclasses import dataclass, replace, is_dataclass
from typing import Any, Iterable

import numpy as np

from src.market.market_data import MarketData


@dataclass(frozen=True, slots=True)
class NumericalGreeksConfig:
    spot_relative_bump: float = 1e-4
    min_spot_bump: float = 1e-4
    volatility_bump: float = 1e-4
    rate_bump: float = 1e-4
    theta_bump_years: float = 1.0 / 365.25
    min_volatility: float = 1e-6
    min_rate: float = -0.99
    compute_theta: bool = True
    only_fill_missing_or_zero: bool = True
    zero_tolerance: float = 1e-14


@dataclass(frozen=True, slots=True)
class ShiftedVolSurfaceForGreeks:
    base_surface: Any
    shift: float
    floor: float = 1e-6

    def volatility(self, maturity, log_moneyness):
        values = self.base_surface.volatility(maturity, log_moneyness)
        return np.maximum(np.asarray(values, dtype=float) + float(self.shift), self.floor)


@dataclass(frozen=True, slots=True)
class NumericalGreeksEngine:
    config: NumericalGreeksConfig = NumericalGreeksConfig()

    def greeks(
        self,
        product: object,
        model: object,
        market_data: MarketData | None = None,
        *,
        base_price: float | None = None,
    ) -> dict[str, float]:
        md = self._normalize_market_data(market_data)
        price0 = float(base_price) if base_price is not None else self._price(model, product, md)

        spot = self._require_number(md.spot, "spot")
        volatility = self._resolve_effective_volatility(model, md)
        rate = self._resolve_effective_rate(model, md)

        spot_bump = max(abs(spot) * self.config.spot_relative_bump, self.config.min_spot_bump)
        vol_bump = self.config.volatility_bump
        rate_bump = self.config.rate_bump

        md_spot_up = replace(md, spot=spot + spot_bump)
        md_spot_down = replace(md, spot=max(spot - spot_bump, self.config.min_spot_bump))

        price_spot_up = self._price(model, product, md_spot_up)
        price_spot_down = self._price(model, product, md_spot_down)
        h_spot = 0.5 * (md_spot_up.spot - md_spot_down.spot)

        delta = (price_spot_up - price_spot_down) / (md_spot_up.spot - md_spot_down.spot)
        gamma = (price_spot_up - 2.0 * price0 + price_spot_down) / (h_spot**2)

        model_vol_up, md_vol_up = self._with_volatility(model, md, volatility + vol_bump)
        model_vol_down, md_vol_down = self._with_volatility(model, md, max(volatility - vol_bump, self.config.min_volatility))
        price_vol_up = self._price(model_vol_up, product, md_vol_up)
        price_vol_down = self._price(model_vol_down, product, md_vol_down)
        vol_denominator = self._resolve_effective_volatility(model_vol_up, md_vol_up) - self._resolve_effective_volatility(model_vol_down, md_vol_down)
        vega = (price_vol_up - price_vol_down) / vol_denominator if abs(vol_denominator) > 0 else 0.0

        model_rate_up, md_rate_up = self._with_rate(model, md, rate + rate_bump)
        model_rate_down, md_rate_down = self._with_rate(model, md, max(rate - rate_bump, self.config.min_rate))
        price_rate_up = self._price(model_rate_up, product, md_rate_up)
        price_rate_down = self._price(model_rate_down, product, md_rate_down)
        rate_denominator = self._resolve_effective_rate(model_rate_up, md_rate_up) - self._resolve_effective_rate(model_rate_down, md_rate_down)
        rho = (price_rate_up - price_rate_down) / rate_denominator if abs(rate_denominator) > 0 else 0.0

        theta = 0.0
        if self.config.compute_theta:
            try:
                product_shorter, dt = self._time_bumped_product(product)
                if dt > 0.0:
                    theta = (self._price(model, product_shorter, md) - price0) / dt
            except Exception:
                theta = 0.0

        return {
            "price": float(price0),
            "delta": float(delta),
            "gamma": float(gamma),
            "vega": float(vega),
            "theta": float(theta),
            "rho": float(rho),
        }

    def enrich_metrics(
        self,
        product: object,
        model: object,
        market_data: MarketData | None,
        metrics: dict[str, Any] | None = None,
        *,
        keys: Iterable[str] = ("delta", "gamma", "vega", "theta", "rho"),
        force: bool = False,
    ) -> dict[str, Any]:
        existing: dict[str, Any] = dict(metrics or {})
        numeric_existing: dict[str, float] = {}
        for key, value in existing.items():
            try:
                numeric_existing[key] = float(value)
            except Exception:
                pass

        needs = force or any(self._needs_fill(numeric_existing, key) for key in keys)
        if not needs:
            return existing

        numerical = self.greeks(
            product,
            model,
            market_data,
            base_price=numeric_existing.get("price"),
        )

        out = dict(existing)
        out.setdefault("price", numerical["price"])
        for key in keys:
            if force or self._needs_fill(numeric_existing, key):
                out[key] = numerical[key]
        out["numerical_greeks_used"] = 1.0
        return out

    def _needs_fill(self, metrics: dict[str, float], key: str) -> bool:
        if key not in metrics:
            return True
        value = metrics.get(key)
        if value is None or not np.isfinite(value):
            return True
        if not self.config.only_fill_missing_or_zero:
            return False
        return abs(float(value)) <= self.config.zero_tolerance

    @staticmethod
    def _price(model: object, product: object, market_data: MarketData) -> float:
        if not hasattr(model, "price"):
            raise TypeError(f"model {type(model)!r} does not expose price().")
        return float(model.price(product, market_data))

    @staticmethod
    def _normalize_market_data(market_data: MarketData | None) -> MarketData:
        if market_data is None:
            raise ValueError("NumericalGreeksEngine requires MarketData.")
        return market_data

    @staticmethod
    def _require_number(value: float | None, name: str) -> float:
        if value is None:
            raise ValueError(f"{name} is required.")
        value = float(value)
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite.")
        return value

    def _resolve_effective_volatility(self, model: object, md: MarketData) -> float:
        """Resolve effective volatility with the same priority as pricing models.

        Priority:
        1. market_data.volatility
        2. model volatility surface, when usable
        3. model.volatility
        """
        if md.volatility is not None:
            return self._require_number(md.volatility, "market_data.volatility")

        model_vol = getattr(model, "volatility", None)
        if model_vol is not None:
            return self._require_number(model_vol, "model.volatility")

        return 0.0

    def _resolve_effective_rate(self, model: object, md: MarketData) -> float:
        """Resolve effective rate with the same priority as pricing models.

        Priority:
        1. market_data.rate
        2. model.yield_curve/model.rate fallback
        """
        if md.rate is not None:
            return self._require_number(md.rate, "market_data.rate")

        model_rate = getattr(model, "rate", None)
        if model_rate is not None:
            return self._require_number(model_rate, "model.rate")

        return 0.0

    def _with_volatility(self, model: object, md: MarketData, new_volatility: float) -> tuple[object, MarketData]:
        new_volatility = max(float(new_volatility), self.config.min_volatility)

        if md.volatility is not None:
            return model, replace(md, volatility=new_volatility)

        if hasattr(model, "vol_surface") and getattr(model, "vol_surface") is not None:
            current_vol = self._resolve_effective_volatility(model, md)
            shift = new_volatility - current_vol
            return self._replace_model(
                model,
                vol_surface=ShiftedVolSurfaceForGreeks(getattr(model, "vol_surface"), shift),
            ), md

        if hasattr(model, "volatility_surface") and getattr(model, "volatility_surface") is not None:
            current_vol = self._resolve_effective_volatility(model, md)
            shift = new_volatility - current_vol
            return self._replace_model(
                model,
                volatility_surface=ShiftedVolSurfaceForGreeks(getattr(model, "volatility_surface"), shift),
            ), md

        if hasattr(model, "volatility") and getattr(model, "volatility") is not None:
            return self._replace_model(model, volatility=new_volatility), md

        return model, replace(md, volatility=new_volatility)

    def _with_rate(self, model: object, md: MarketData, new_rate: float) -> tuple[object, MarketData]:
        """Return bumped model/market pair for rate finite differences.

        Since MarketData has priority in production pricing, bump MarketData first
        whenever market_data.rate is available.
        """
        new_rate = float(new_rate)

        if md.rate is not None:
            return model, replace(md, rate=new_rate)

        curve = getattr(model, "yield_curve", None)
        if curve is not None:
            current_rate = self._resolve_effective_rate(model, md)
            shifted_curve = self._shift_curve(curve, new_rate - current_rate)
            return self._replace_model(model, yield_curve=shifted_curve), md

        if hasattr(model, "rate") and getattr(model, "rate") is not None:
            return self._replace_model(model, rate=new_rate), md

        return model, replace(md, rate=new_rate)

    @staticmethod
    def _replace_model(model: object, **kwargs: Any) -> object:
        if is_dataclass(model):
            return replace(model, **kwargs)

        clone = model.__class__.__new__(model.__class__)
        if hasattr(model, "__dict__"):
            clone.__dict__.update(model.__dict__)
        for key, value in kwargs.items():
            setattr(clone, key, value)
        return clone

    @staticmethod
    def _shift_curve(curve: object, shift: float) -> object:
        from src.rates.yield_curve import YieldCurve

        return YieldCurve(
            maturities=curve.maturities.copy(),
            zero_rates=curve.zero_rates + float(shift),
            interpolation=curve.interpolation,
            name=f"{curve.name}_shift_{float(shift):+.6f}",
            interpolation_on=curve.interpolation_on,
        )

    def _time_bumped_product(self, product: object) -> tuple[object, float]:
        if not hasattr(product, "maturity"):
            raise TypeError("product does not expose maturity.")

        maturity = float(getattr(product, "maturity"))
        if maturity <= 0.0:
            raise ValueError("product maturity must be positive.")

        dt = min(self.config.theta_bump_years, maturity * 0.5)
        new_maturity = max(maturity - dt, 1e-12)

        replacements: dict[str, Any] = {"maturity": new_maturity}

        if hasattr(product, "observation_dates"):
            obs = []
            for item in getattr(product, "observation_dates"):
                try:
                    obs.append(max(float(item) - dt, 1e-12))
                except Exception:
                    obs.append(item)
            replacements["observation_dates"] = obs

        if is_dataclass(product):
            return replace(product, **replacements), dt

        clone = product.__class__.__new__(product.__class__)
        if hasattr(product, "__dict__"):
            clone.__dict__.update(product.__dict__)
        for key, value in replacements.items():
            setattr(clone, key, value)
        return clone, dt


__all__ = ["NumericalGreeksConfig", "NumericalGreeksEngine"]
