"""SVI and SSVI volatility-surface calibration.

Inputs are expected to come from ``calibrate_implied_vol_panel`` and therefore
contain at least:
- time_to_maturity_years
- log_moneyness = log(K / S)
- implied_vol

The module calibrates total variance, not volatility directly:
    w(T, k) = sigma_BS(T, k)^2 * T
This is the market standard representation for smile fitting and static
arbitrage checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log, sqrt
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import ndtr

from src.calibration.base import CalibrationResult


# ---------------------------------------------------------------------------
# SVI raw parametrisation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SVIParameters:
    """Raw SVI parameters for one maturity slice.

    w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))
    """

    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def as_array(self) -> np.ndarray:
        return np.array([self.a, self.b, self.rho, self.m, self.sigma], dtype=float)


@dataclass(frozen=True, slots=True)
class SVISlice:
    """One calibrated SVI smile slice."""

    maturity: float
    params: SVIParameters
    objective_value: float
    quote_count: int

    def total_variance(self, log_moneyness: float | np.ndarray) -> float | np.ndarray:
        return svi_total_variance(log_moneyness, self.params)

    def volatility(self, log_moneyness: float | np.ndarray) -> float | np.ndarray:
        w = np.maximum(np.asarray(self.total_variance(log_moneyness), dtype=float), 0.0)
        vol = np.sqrt(w / max(self.maturity, 1e-14))
        if np.isscalar(log_moneyness):
            return float(vol)
        return vol


@dataclass(frozen=True, slots=True)
class SVIVolSurface:
    """Piecewise SVI surface with linear interpolation in total variance over maturity."""

    slices: tuple[SVISlice, ...]
    calibration: CalibrationResult

    @classmethod
    def fit_from_quotes(
        cls,
        quotes: pd.DataFrame,
        *,
        maturity_column: str = "time_to_maturity_years",
        log_moneyness_column: str = "log_moneyness",
        iv_column: str = "implied_vol",
        min_points_per_slice: int = 5,
        maturity_round: int = 8,
    ) -> "SVIVolSurface":
        required = {maturity_column, log_moneyness_column, iv_column}
        missing = required.difference(quotes.columns)
        if missing:
            raise ValueError(f"Missing columns for SVI calibration: {sorted(missing)}")

        data = quotes[list(required)].dropna().copy()
        data = data[(data[maturity_column] > 0.0) & (data[iv_column] > 0.0)].copy()
        if data.empty:
            raise ValueError("No usable quotes for SVI calibration.")

        data["maturity_bucket"] = data[maturity_column].round(maturity_round)
        fitted: list[SVISlice] = []

        for maturity_bucket, group in data.groupby("maturity_bucket", sort=True):
            if len(group) < min_points_per_slice:
                continue
            maturity = float(group[maturity_column].median())
            k = group[log_moneyness_column].to_numpy(dtype=float)
            w = np.square(group[iv_column].to_numpy(dtype=float)) * maturity
            fitted.append(fit_svi_slice(k, w, maturity=maturity))

        if not fitted:
            raise ValueError("No SVI slice could be calibrated. Increase data quality or reduce min_points_per_slice.")

        fitted = sorted(fitted, key=lambda item: item.maturity)
        objective = float(np.mean([s.objective_value for s in fitted]))
        params = {
            "slice_count": float(len(fitted)),
            "quote_count": float(sum(s.quote_count for s in fitted)),
            "mean_slice_objective": objective,
        }
        calibration = CalibrationResult("svi_raw_surface", parameters=params, objective_value=objective)
        return cls(slices=tuple(fitted), calibration=calibration)

    @property
    def maturities(self) -> np.ndarray:
        return np.array([s.maturity for s in self.slices], dtype=float)

    def total_variance(self, maturity: float | np.ndarray, log_moneyness: float | np.ndarray) -> float | np.ndarray:
        """Evaluate total variance with linear interpolation between SVI slices."""
        t_arr, k_arr = np.broadcast_arrays(np.asarray(maturity, dtype=float), np.asarray(log_moneyness, dtype=float))
        values = np.empty_like(t_arr, dtype=float)
        mats = self.maturities

        for index in np.ndindex(t_arr.shape):
            t = float(t_arr[index])
            k = float(k_arr[index])
            if t <= mats[0]:
                values[index] = float(self.slices[0].total_variance(k)) * t / max(mats[0], 1e-14)
            elif t >= mats[-1]:
                values[index] = float(self.slices[-1].total_variance(k)) * t / max(mats[-1], 1e-14)
            else:
                upper = int(np.searchsorted(mats, t, side="right"))
                lower = upper - 1
                t0 = mats[lower]
                t1 = mats[upper]
                w0 = float(self.slices[lower].total_variance(k))
                w1 = float(self.slices[upper].total_variance(k))
                weight = (t - t0) / (t1 - t0)
                values[index] = (1.0 - weight) * w0 + weight * w1

        values = np.maximum(values, 0.0)
        if np.isscalar(maturity) and np.isscalar(log_moneyness):
            return float(values)
        return values

    def volatility(self, maturity: float | np.ndarray, log_moneyness: float | np.ndarray) -> float | np.ndarray:
        t = np.asarray(maturity, dtype=float)
        w = np.asarray(self.total_variance(maturity, log_moneyness), dtype=float)
        vol = np.sqrt(np.maximum(w, 0.0) / np.maximum(t, 1e-14))
        if np.isscalar(maturity) and np.isscalar(log_moneyness):
            return float(vol)
        return vol

    def diagnostics(
        self,
        *,
        log_moneyness_grid: np.ndarray | None = None,
        tolerance: float = 1e-8,
    ) -> dict[str, bool | float]:
        if log_moneyness_grid is None:
            log_moneyness_grid = np.linspace(-0.7, 0.7, 101)

        butterfly = [
            check_butterfly_arbitrage_slice(s, log_moneyness_grid=log_moneyness_grid, tolerance=tolerance)
            for s in self.slices
        ]
        calendar_ok = check_calendar_arbitrage_surface(self, log_moneyness_grid=log_moneyness_grid, tolerance=tolerance)

        return {
            "butterfly_arbitrage_free": bool(all(item["ok"] for item in butterfly)),
            "calendar_arbitrage_free": bool(calendar_ok),
            "min_butterfly_convexity": float(min(item["min_convexity"] for item in butterfly)),
            "max_total_variance": float(max(np.max(s.total_variance(log_moneyness_grid)) for s in self.slices)),
            "min_total_variance": float(min(np.min(s.total_variance(log_moneyness_grid)) for s in self.slices)),
        }


def svi_total_variance(log_moneyness: float | np.ndarray, params: SVIParameters) -> float | np.ndarray:
    k = np.asarray(log_moneyness, dtype=float)
    x = k - params.m
    values = params.a + params.b * (params.rho * x + np.sqrt(x * x + params.sigma * params.sigma))
    if np.isscalar(log_moneyness):
        return float(values)
    return values


def fit_svi_slice(
    log_moneyness: np.ndarray,
    total_variance: np.ndarray,
    *,
    maturity: float,
    weights: np.ndarray | None = None,
) -> SVISlice:
    """Calibrate one raw-SVI slice by weighted least squares."""
    k = np.asarray(log_moneyness, dtype=float)
    w = np.asarray(total_variance, dtype=float)
    if k.ndim != 1 or w.ndim != 1 or len(k) != len(w):
        raise ValueError("log_moneyness and total_variance must be one-dimensional arrays of same length.")
    if len(k) < 5:
        raise ValueError("Raw SVI requires at least five points per maturity slice.")
    if np.any(w <= 0.0):
        raise ValueError("Total variances must be strictly positive.")

    order = np.argsort(k)
    k = k[order]
    w = w[order]
    ww = np.ones_like(w) if weights is None else np.asarray(weights, dtype=float)[order]
    ww = ww / np.mean(ww)

    initial = _initial_svi_guess(k, w)
    bounds = _svi_bounds(k, w)

    def objective(x: np.ndarray) -> float:
        params = SVIParameters(float(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]))
        model = np.asarray(svi_total_variance(k, params), dtype=float)
        residual = model - w
        penalty = _svi_penalty(params, k)
        return float(np.mean(ww * residual * residual) + penalty)

    starts = [initial]
    # A few deterministic starts reduce local-minimum sensitivity without making
    # calibration too slow for a student project.
    for rho in (-0.7, -0.3, 0.0, 0.3):
        guess = initial.copy()
        guess[2] = rho
        starts.append(guess)

    best = None
    for x0 in starts:
        result = minimize(objective, x0=x0, bounds=bounds, method="L-BFGS-B", options={"maxiter": 2000})
        if best is None or result.fun < best.fun:
            best = result

    if best is None or not best.success:
        message = "unknown" if best is None else best.message
        raise RuntimeError(f"SVI calibration failed for maturity={maturity}: {message}")

    params = SVIParameters(float(best.x[0]), float(best.x[1]), float(best.x[2]), float(best.x[3]), float(best.x[4]))
    return SVISlice(maturity=float(maturity), params=params, objective_value=float(best.fun), quote_count=int(len(k)))


def _initial_svi_guess(k: np.ndarray, w: np.ndarray) -> np.ndarray:
    min_w = float(np.min(w))
    max_w = float(np.max(w))
    k_span = float(max(np.max(k) - np.min(k), 1e-3))
    m0 = float(k[np.argmin(w)])
    sigma0 = max(0.10, 0.25 * k_span)
    b0 = max((max_w - min_w) / (k_span + sigma0), 1e-4)
    a0 = max(min_w - b0 * sigma0, 1e-8)
    return np.array([a0, b0, -0.30, m0, sigma0], dtype=float)


def _svi_bounds(k: np.ndarray, w: np.ndarray) -> list[tuple[float, float]]:
    max_w = float(max(np.max(w), 1e-4))
    k_min = float(np.min(k))
    k_max = float(np.max(k))
    k_span = max(k_max - k_min, 0.25)
    return [
        (1e-10, 4.0 * max_w + 1.0),     # a
        (1e-8, 20.0),                    # b
        (-0.999, 0.999),                 # rho
        (k_min - k_span, k_max + k_span),# m
        (1e-5, 5.0),                     # sigma
    ]


def _svi_penalty(params: SVIParameters, k: np.ndarray) -> float:
    penalty = 0.0
    if params.b <= 0.0:
        penalty += 1e6 * (abs(params.b) + 1.0)
    if params.sigma <= 0.0:
        penalty += 1e6 * (abs(params.sigma) + 1.0)
    if abs(params.rho) >= 1.0:
        penalty += 1e6 * (abs(params.rho) - 0.999) ** 2

    min_variance = params.a + params.b * params.sigma * sqrt(max(1.0 - params.rho * params.rho, 0.0))
    if min_variance <= 0.0:
        penalty += 1e6 * min_variance * min_variance + 1e3

    values = np.asarray(svi_total_variance(k, params), dtype=float)
    if np.any(values <= 0.0):
        penalty += 1e6 * float(np.sum(np.square(np.minimum(values, 0.0)))) + 1e3
    return float(penalty)


# ---------------------------------------------------------------------------
# SSVI surface parametrisation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SSVIParameters:
    """SSVI parameters.

    theta(T) = T * [v_inf + (v0 - v_inf) * (1 - exp(-kappa T)) / (kappa T)]
    phi(theta) = eta * theta^(-lambda)
    """

    v0: float
    v_inf: float
    kappa: float
    rho: float
    eta: float
    lambda_: float

    def as_array(self) -> np.ndarray:
        return np.array([self.v0, self.v_inf, self.kappa, self.rho, self.eta, self.lambda_], dtype=float)


@dataclass(frozen=True, slots=True)
class SSVIVolSurface:
    """Parametric SSVI surface."""

    params: SSVIParameters
    calibration: CalibrationResult

    @classmethod
    def fit_from_quotes(
        cls,
        quotes: pd.DataFrame,
        *,
        maturity_column: str = "time_to_maturity_years",
        log_moneyness_column: str = "log_moneyness",
        iv_column: str = "implied_vol",
        max_multistarts: int = 8,
    ) -> "SSVIVolSurface":
        required = {maturity_column, log_moneyness_column, iv_column}
        missing = required.difference(quotes.columns)
        if missing:
            raise ValueError(f"Missing columns for SSVI calibration: {sorted(missing)}")

        data = quotes[list(required)].dropna().copy()
        data = data[(data[maturity_column] > 0.0) & (data[iv_column] > 0.0)].copy()
        if len(data) < 12:
            raise ValueError("SSVI calibration requires a reasonably broad option panel, at least 12 quotes.")

        t = data[maturity_column].to_numpy(dtype=float)
        k = data[log_moneyness_column].to_numpy(dtype=float)
        w = np.square(data[iv_column].to_numpy(dtype=float)) * t
        weights = 1.0 / np.maximum(1.0, np.abs(k))
        weights = weights / np.mean(weights)

        bounds = [
            (1e-5, 2.0),    # v0
            (1e-5, 2.0),    # v_inf
            (1e-4, 20.0),   # kappa
            (-0.999, 0.999),# rho
            (1e-5, 20.0),   # eta
            (0.0, 1.0),     # lambda
        ]
        starts = _ssvi_initial_guesses(t, k, w, max_multistarts=max_multistarts)

        def objective(x: np.ndarray) -> float:
            params = SSVIParameters(float(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5]))
            model = np.asarray(ssvi_total_variance(t, k, params), dtype=float)
            residual = model - w
            penalty = _ssvi_penalty(params, maturities=np.unique(t))
            return float(np.mean(weights * residual * residual) + penalty)

        best = None
        for x0 in starts:
            result = minimize(objective, x0=x0, bounds=bounds, method="L-BFGS-B", options={"maxiter": 3000})
            if best is None or result.fun < best.fun:
                best = result

        if best is None or not best.success:
            message = "unknown" if best is None else best.message
            raise RuntimeError(f"SSVI calibration failed: {message}")

        params = SSVIParameters(float(best.x[0]), float(best.x[1]), float(best.x[2]), float(best.x[3]), float(best.x[4]), float(best.x[5]))
        calibration = CalibrationResult(
            model_name="ssvi_surface",
            parameters={
                "v0": params.v0,
                "v_inf": params.v_inf,
                "kappa": params.kappa,
                "rho": params.rho,
                "eta": params.eta,
                "lambda": params.lambda_,
                "quote_count": float(len(data)),
            },
            objective_value=float(best.fun),
        )
        return cls(params=params, calibration=calibration)

    def theta(self, maturity: float | np.ndarray) -> float | np.ndarray:
        return ssvi_theta(maturity, self.params)

    def total_variance(self, maturity: float | np.ndarray, log_moneyness: float | np.ndarray) -> float | np.ndarray:
        return ssvi_total_variance(maturity, log_moneyness, self.params)

    def volatility(self, maturity: float | np.ndarray, log_moneyness: float | np.ndarray) -> float | np.ndarray:
        t = np.asarray(maturity, dtype=float)
        w = np.asarray(self.total_variance(maturity, log_moneyness), dtype=float)
        vol = np.sqrt(np.maximum(w, 0.0) / np.maximum(t, 1e-14))
        if np.isscalar(maturity) and np.isscalar(log_moneyness):
            return float(vol)
        return vol

    def diagnostics(
        self,
        *,
        maturity_grid: np.ndarray | None = None,
        log_moneyness_grid: np.ndarray | None = None,
        tolerance: float = 1e-8,
    ) -> dict[str, bool | float]:
        if maturity_grid is None:
            maturity_grid = np.array([1/12, 0.25, 0.5, 1.0, 2.0, 5.0], dtype=float)
        if log_moneyness_grid is None:
            log_moneyness_grid = np.linspace(-0.7, 0.7, 101)

        pseudo_slices = tuple(
            SVISlice(
                maturity=float(t),
                params=fit_svi_slice(
                    log_moneyness_grid,
                    np.asarray(self.total_variance(float(t), log_moneyness_grid), dtype=float),
                    maturity=float(t),
                ).params,
                objective_value=0.0,
                quote_count=len(log_moneyness_grid),
            )
            for t in maturity_grid
        )
        surface = SVIVolSurface(pseudo_slices, CalibrationResult("ssvi_diagnostics"))
        return surface.diagnostics(log_moneyness_grid=log_moneyness_grid, tolerance=tolerance)


def ssvi_theta(maturity: float | np.ndarray, params: SSVIParameters) -> float | np.ndarray:
    t = np.asarray(maturity, dtype=float)
    kt = params.kappa * t
    factor = np.ones_like(t, dtype=float)
    mask = np.abs(kt) > 1e-12
    factor[mask] = (1.0 - np.exp(-kt[mask])) / kt[mask]
    avg_variance = params.v_inf + (params.v0 - params.v_inf) * factor
    theta = np.maximum(t * avg_variance, 1e-14)
    if np.isscalar(maturity):
        return float(theta)
    return theta


def ssvi_total_variance(
    maturity: float | np.ndarray,
    log_moneyness: float | np.ndarray,
    params: SSVIParameters,
) -> float | np.ndarray:
    t, k = np.broadcast_arrays(np.asarray(maturity, dtype=float), np.asarray(log_moneyness, dtype=float))
    theta = np.asarray(ssvi_theta(t, params), dtype=float)
    phi = params.eta * np.power(np.maximum(theta, 1e-14), -params.lambda_)
    x = phi * k
    inside = np.square(x + params.rho) + 1.0 - params.rho * params.rho
    values = 0.5 * theta * (1.0 + params.rho * x + np.sqrt(np.maximum(inside, 0.0)))
    values = np.maximum(values, 1e-14)
    if np.isscalar(maturity) and np.isscalar(log_moneyness):
        return float(values)
    return values


def _ssvi_initial_guesses(t: np.ndarray, k: np.ndarray, w: np.ndarray, *, max_multistarts: int) -> list[np.ndarray]:
    atm_mask = np.abs(k) <= max(0.05, np.quantile(np.abs(k), 0.25))
    if np.any(atm_mask):
        atm_var = float(np.median(w[atm_mask] / t[atm_mask]))
    else:
        atm_var = float(np.median(w / t))
    atm_var = min(max(atm_var, 1e-4), 1.0)

    starts = []
    for rho in (-0.7, -0.4, -0.1, 0.2):
        for lam in (0.2, 0.5):
            starts.append(np.array([atm_var, atm_var, 1.0, rho, 0.5, lam], dtype=float))
    return starts[:max_multistarts]


def _ssvi_penalty(params: SSVIParameters, *, maturities: np.ndarray) -> float:
    penalty = 0.0
    if params.v0 <= 0.0 or params.v_inf <= 0.0 or params.kappa <= 0.0 or params.eta <= 0.0:
        penalty += 1e6
    if not (-1.0 < params.rho < 1.0):
        penalty += 1e6
    if not (0.0 <= params.lambda_ <= 1.0):
        penalty += 1e6

    theta = np.asarray(ssvi_theta(maturities, params), dtype=float)
    if np.any(np.diff(theta) < -1e-10):
        penalty += 1e5 * float(np.sum(np.square(np.minimum(np.diff(theta), 0.0))))

    # Practical Gatheral-Jacquier style constraints sampled on theta grid.
    phi = params.eta * np.power(np.maximum(theta, 1e-14), -params.lambda_)
    c1 = theta * phi * (1.0 + abs(params.rho))
    c2 = theta * phi * phi * (1.0 + abs(params.rho))
    penalty += 1e4 * float(np.sum(np.square(np.maximum(c1 - 4.0, 0.0))))
    penalty += 1e4 * float(np.sum(np.square(np.maximum(c2 - 4.0, 0.0))))
    return float(penalty)


# ---------------------------------------------------------------------------
# Static-arbitrage checks via vanilla-call prices
# ---------------------------------------------------------------------------


def check_butterfly_arbitrage_slice(
    svi_slice: SVISlice,
    *,
    log_moneyness_grid: np.ndarray,
    tolerance: float = 1e-8,
) -> dict[str, bool | float]:
    """Check call monotonicity and convexity in strike for one smile slice."""
    k = np.asarray(log_moneyness_grid, dtype=float)
    order = np.argsort(k)
    k = k[order]
    strikes = np.exp(k)
    total_variance = np.asarray(svi_slice.total_variance(k), dtype=float)
    calls = _forward_call_prices_from_total_variance(strikes, total_variance)

    decreasing = bool(np.all(np.diff(calls) <= tolerance))
    slopes = np.diff(calls) / np.diff(strikes)
    convexity = np.diff(slopes) / ((strikes[2:] - strikes[:-2]) / 2.0)
    convex = bool(np.all(convexity >= -tolerance))
    bounds_ok = bool(np.all(calls >= -tolerance) and np.all(calls <= 1.0 + tolerance))

    return {
        "ok": bool(decreasing and convex and bounds_ok),
        "decreasing": decreasing,
        "convex": convex,
        "bounds_ok": bounds_ok,
        "min_convexity": float(np.min(convexity)) if len(convexity) else float("nan"),
    }


def check_calendar_arbitrage_surface(
    surface: SVIVolSurface,
    *,
    log_moneyness_grid: np.ndarray,
    tolerance: float = 1e-8,
) -> bool:
    """Calendar no-arbitrage proxy: total variance non-decreasing in maturity."""
    maturities = surface.maturities
    k = np.asarray(log_moneyness_grid, dtype=float)
    previous = None
    for maturity in maturities:
        current = np.asarray(surface.total_variance(float(maturity), k), dtype=float)
        if previous is not None and np.any(current + tolerance < previous):
            return False
        previous = current
    return True


def _forward_call_prices_from_total_variance(strikes: np.ndarray, total_variance: np.ndarray) -> np.ndarray:
    """Black-76 call with forward F=1 and discount factor=1."""
    k = np.log(strikes)
    sqrt_w = np.sqrt(np.maximum(total_variance, 1e-14))
    d1 = (-k + 0.5 * total_variance) / sqrt_w
    d2 = d1 - sqrt_w
    return ndtr(d1) - strikes * ndtr(d2)


__all__ = [
    "SSVIParameters",
    "SSVIVolSurface",
    "SVIParameters",
    "SVISlice",
    "SVIVolSurface",
    "check_butterfly_arbitrage_slice",
    "check_calendar_arbitrage_surface",
    "fit_svi_slice",
    "ssvi_theta",
    "ssvi_total_variance",
    "svi_total_variance",
]
