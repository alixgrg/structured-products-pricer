"""Yield curve objects and parametric rate-curve helpers.

This file is a drop-in replacement for the previous ``src/rates/yield_curve.py``.
It keeps the same public methods used by ``BlackScholesModel`` and
``DiscountingModel`` while adding a clean constructor from discount factors,
log-discount interpolation and zero-coupon consistency checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

InterpolationMethod = Literal["linear", "cubic"]
CompoundingMethod = Literal["continuous", "annual"]
CurveInputType = Literal["zero_rates", "discount_factors"]


@dataclass(frozen=True, slots=True)
class YieldCurve:
    """Interpolated zero-rate curve.

    Parameters
    ----------
    maturities:
        Maturities in years.
    zero_rates:
        Continuously-compounded zero rates in decimal format.
    interpolation:
        ``linear`` is robust. ``cubic`` is available when scipy is installed.
    name:
        Curve name.
    interpolation_on:
        ``zero_rates`` interpolates zero rates. ``discount_factors`` interpolates
        log discount factors and is preferable for bootstrapped market curves.
    """

    maturities: np.ndarray
    zero_rates: np.ndarray
    interpolation: InterpolationMethod = "linear"
    name: str = "yield_curve"
    interpolation_on: CurveInputType = "zero_rates"

    def __post_init__(self) -> None:
        maturities = np.asarray(self.maturities, dtype=float)
        zero_rates = np.asarray(self.zero_rates, dtype=float)

        if maturities.ndim != 1 or zero_rates.ndim != 1:
            raise ValueError("maturities and zero_rates must be one-dimensional arrays.")
        if len(maturities) != len(zero_rates):
            raise ValueError("maturities and zero_rates must have the same length.")
        if len(maturities) < 2:
            raise ValueError("At least two curve points are required.")
        if np.any(~np.isfinite(maturities)) or np.any(~np.isfinite(zero_rates)):
            raise ValueError("maturities and zero_rates must contain finite values.")
        if np.any(maturities < 0.0):
            raise ValueError("maturities must be non-negative.")
        if self.interpolation not in ("linear", "cubic"):
            raise ValueError("interpolation must be either 'linear' or 'cubic'.")
        if self.interpolation_on not in ("zero_rates", "discount_factors"):
            raise ValueError("interpolation_on must be 'zero_rates' or 'discount_factors'.")

        order = np.argsort(maturities)
        maturities = maturities[order]
        zero_rates = zero_rates[order]

        if len(np.unique(maturities)) != len(maturities):
            raise ValueError("maturities must be unique.")

        object.__setattr__(self, "maturities", maturities)
        object.__setattr__(self, "zero_rates", zero_rates)

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        *,
        maturity_column: str = "curve_tenor_years",
        rate_column: str = "rate_decimal",
        interpolation: InterpolationMethod = "linear",
        name: str = "yield_curve",
    ) -> "YieldCurve":
        required = {maturity_column, rate_column}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        clean = frame[[maturity_column, rate_column]].dropna().copy().sort_values(maturity_column)
        return cls(
            maturities=clean[maturity_column].to_numpy(dtype=float),
            zero_rates=clean[rate_column].to_numpy(dtype=float),
            interpolation=interpolation,
            name=name,
            interpolation_on="zero_rates",
        )

    @classmethod
    def from_rate_curves(
        cls,
        frame: pd.DataFrame,
        *,
        country: str | None = None,
        observation_date: str | pd.Timestamp | None = None,
        interpolation: InterpolationMethod = "linear",
    ) -> "YieldCurve":
        required = {"country", "observation_date", "curve_tenor_years", "rate_decimal"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        data = frame.copy()
        data["observation_date"] = pd.to_datetime(data["observation_date"]).dt.normalize()

        if country is not None:
            selected_country = country
            data = data[data["country"].astype(str) == selected_country]
            if data.empty:
                raise ValueError(f"No rate curve data found for country={selected_country!r}.")
        else:
            selected_country = None

        selected_date = (
            data["observation_date"].max()
            if observation_date is None
            else pd.to_datetime(observation_date).normalize()
        )
        data = data[data["observation_date"] == selected_date]

        if selected_country is None:
            if data.empty:
                raise ValueError(f"No rate curve data found for date {selected_date}.")
            selected_country = str(data["country"].iloc[0])
            data = data[data["country"].astype(str) == selected_country]

        if data.empty:
            raise ValueError(
                f"No rate curve data found for country={selected_country!r} "
                f"and observation_date={selected_date.date()}."
            )

        return cls.from_frame(
            data,
            interpolation=interpolation,
            name=f"{selected_country} {selected_date.date()}",
        )

    @classmethod
    def from_discount_factors(
        cls,
        maturities: np.ndarray | list[float],
        discount_factors: np.ndarray | list[float],
        *,
        interpolation: InterpolationMethod = "linear",
        name: str = "discount_curve",
        interpolation_on: CurveInputType = "discount_factors",
    ) -> "YieldCurve":
        """Build a curve from discount factors.

        The internal representation remains zero rates for compatibility, but
        the curve can interpolate directly on log discount factors, which is
        more stable for bootstrapped market curves.
        """
        t = np.asarray(maturities, dtype=float)
        df = np.asarray(discount_factors, dtype=float)

        if t.ndim != 1 or df.ndim != 1:
            raise ValueError("maturities and discount_factors must be one-dimensional arrays.")
        if len(t) != len(df):
            raise ValueError("maturities and discount_factors must have the same length.")
        if np.any(t <= 0.0):
            raise ValueError("discount-factor curve maturities must be strictly positive.")
        if np.any(df <= 0.0) or np.any(df > 2.0):
            raise ValueError("discount factors must be positive and reasonably bounded.")

        zero_rates = -np.log(df) / t
        return cls(
            maturities=t,
            zero_rates=zero_rates,
            interpolation=interpolation,
            name=name,
            interpolation_on=interpolation_on,
        )

    def zero_rate(self, maturity: float | np.ndarray) -> float | np.ndarray:
        """Return the interpolated continuously-compounded zero rate."""
        t = np.asarray(maturity, dtype=float)
        if np.any(t < 0.0):
            raise ValueError("maturity must be non-negative.")

        if self.interpolation_on == "discount_factors":
            df = np.asarray(self.discount_factor(t), dtype=float)
            values = np.where(t > 1e-14, -np.log(df) / np.maximum(t, 1e-14), self.zero_rates[0])
        elif self.interpolation == "linear":
            values = np.interp(t, self.maturities, self.zero_rates, left=self.zero_rates[0], right=self.zero_rates[-1])
        else:
            values = self._cubic_interpolate(t)

        if np.isscalar(maturity):
            return float(values)
        return values

    def discount_factor(
        self,
        maturity: float | np.ndarray,
        *,
        compounding: CompoundingMethod = "continuous",
    ) -> float | np.ndarray:
        """Return discount factor P(0,T)."""
        t = np.asarray(maturity, dtype=float)
        if np.any(t < 0.0):
            raise ValueError("maturity must be non-negative.")

        if self.interpolation_on == "discount_factors" and compounding == "continuous":
            values = self._interpolated_discount_factors(t)
        else:
            rates = np.asarray(self.zero_rate(t), dtype=float)
            if compounding == "continuous":
                values = np.exp(-rates * t)
            elif compounding == "annual":
                values = 1.0 / np.power(1.0 + rates, t)
            else:
                raise ValueError("compounding must be either 'continuous' or 'annual'.")

        if np.isscalar(maturity):
            return float(values)
        return values

    def forward_rate(
        self,
        start: float,
        end: float,
        *,
        compounding: CompoundingMethod = "continuous",
    ) -> float:
        """Return the forward rate between start and end."""
        if start < 0.0:
            raise ValueError("start must be non-negative.")
        if end <= start:
            raise ValueError("end must be strictly greater than start.")

        df_start = self.discount_factor(start, compounding=compounding)
        df_end = self.discount_factor(end, compounding=compounding)
        tau = end - start

        if compounding == "continuous":
            return float(-np.log(df_end / df_start) / tau)
        if compounding == "annual":
            return float((df_start / df_end) ** (1.0 / tau) - 1.0)
        raise ValueError("compounding must be either 'continuous' or 'annual'.")

    def zero_coupon_price(
        self,
        maturity: float,
        *,
        notional: float = 100.0,
        compounding: CompoundingMethod = "continuous",
    ) -> float:
        return float(notional * self.discount_factor(maturity, compounding=compounding))

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "maturity": self.maturities,
                "zero_rate": self.zero_rates,
                "discount_factor": self.discount_factor(self.maturities),
            }
        )

    def check_no_static_arbitrage(self, *, tolerance: float = 1e-10) -> dict[str, bool | float]:
        """Basic ZC curve sanity checks.

        A clean discount curve should have positive discount factors and, for
        positive-rate markets, mostly non-increasing discount factors. Negative
        rates can make long-dated discount factors above 1, so this is a sanity
        check rather than a hard mathematical theorem.
        """
        df = np.asarray(self.discount_factor(self.maturities), dtype=float)
        return {
            "positive_discount_factors": bool(np.all(df > 0.0)),
            "non_increasing_discount_factors": bool(np.all(np.diff(df) <= tolerance)),
            "min_discount_factor": float(np.min(df)),
            "max_discount_factor": float(np.max(df)),
        }

    def _interpolated_discount_factors(self, maturity: np.ndarray) -> np.ndarray:
        t = np.asarray(maturity, dtype=float)
        log_df_nodes = -self.zero_rates * self.maturities

        if self.interpolation == "linear":
            log_df = np.interp(t, self.maturities, log_df_nodes, left=log_df_nodes[0], right=log_df_nodes[-1])
        else:
            log_df = self._cubic_interpolate_generic(t, self.maturities, log_df_nodes)

        values = np.exp(log_df)
        values = np.where(t <= 1e-14, 1.0, values)
        return values

    def _cubic_interpolate(self, maturity: np.ndarray) -> np.ndarray:
        return self._cubic_interpolate_generic(maturity, self.maturities, self.zero_rates)

    @staticmethod
    def _cubic_interpolate_generic(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
        try:
            from scipy.interpolate import CubicSpline
        except ImportError as exc:
            raise ImportError("Cubic interpolation requires scipy. Install it or use interpolation='linear'.") from exc

        spline = CubicSpline(xp, fp, bc_type="natural", extrapolate=True)
        values = np.asarray(spline(x), dtype=float)
        below = x < xp[0]
        above = x > xp[-1]
        if np.any(below):
            values[below] = fp[0]
        if np.any(above):
            values[above] = fp[-1]
        return values


@dataclass(frozen=True, slots=True)
class NelsonSiegelParameters:
    beta0: float
    beta1: float
    beta2: float
    lambda_: float


def nelson_siegel_zero_rate(maturity: float | np.ndarray, params: NelsonSiegelParameters) -> float | np.ndarray:
    t = np.asarray(maturity, dtype=float)
    if np.any(t < 0.0):
        raise ValueError("maturity must be non-negative.")

    lambda_t = params.lambda_ * t
    factor1 = np.ones_like(t, dtype=float)
    non_zero = np.abs(lambda_t) > 1e-12
    factor1[non_zero] = (1.0 - np.exp(-lambda_t[non_zero])) / lambda_t[non_zero]
    factor2 = factor1 - np.exp(-lambda_t)
    values = params.beta0 + params.beta1 * factor1 + params.beta2 * factor2

    if np.isscalar(maturity):
        return float(values)
    return values


def fit_nelson_siegel(
    maturities: np.ndarray,
    zero_rates: np.ndarray,
    *,
    initial_guess: NelsonSiegelParameters | None = None,
) -> NelsonSiegelParameters:
    try:
        from scipy.optimize import minimize
    except ImportError as exc:
        raise ImportError("Nelson-Siegel fitting requires scipy.") from exc

    t = np.asarray(maturities, dtype=float)
    r = np.asarray(zero_rates, dtype=float)
    if initial_guess is None:
        initial_guess = NelsonSiegelParameters(
            beta0=float(r[-1]),
            beta1=float(r[0] - r[-1]),
            beta2=0.0,
            lambda_=1.0,
        )

    def objective(x: np.ndarray) -> float:
        params = NelsonSiegelParameters(float(x[0]), float(x[1]), float(x[2]), float(x[3]))
        errors = np.asarray(nelson_siegel_zero_rate(t, params), dtype=float) - r
        return float(np.mean(errors**2))

    result = minimize(
        objective,
        x0=np.array([initial_guess.beta0, initial_guess.beta1, initial_guess.beta2, initial_guess.lambda_]),
        bounds=[(-0.20, 0.20), (-0.50, 0.50), (-0.50, 0.50), (1e-4, 10.0)],
        method="L-BFGS-B",
    )
    if not result.success:
        raise RuntimeError(f"Nelson-Siegel calibration failed: {result.message}")

    return NelsonSiegelParameters(float(result.x[0]), float(result.x[1]), float(result.x[2]), float(result.x[3]))


@dataclass(frozen=True, slots=True)
class NelsonSiegelCurve:
    params: NelsonSiegelParameters
    name: str = "nelson_siegel_curve"

    @classmethod
    def fit(cls, maturities: np.ndarray, zero_rates: np.ndarray, *, name: str = "nelson_siegel_curve") -> "NelsonSiegelCurve":
        return cls(params=fit_nelson_siegel(maturities, zero_rates), name=name)

    def zero_rate(self, maturity: float | np.ndarray) -> float | np.ndarray:
        return nelson_siegel_zero_rate(maturity, self.params)

    def discount_factor(self, maturity: float | np.ndarray, *, compounding: CompoundingMethod = "continuous") -> float | np.ndarray:
        t = np.asarray(maturity, dtype=float)
        rates = np.asarray(self.zero_rate(t), dtype=float)
        if compounding == "continuous":
            values = np.exp(-rates * t)
        elif compounding == "annual":
            values = 1.0 / np.power(1.0 + rates, t)
        else:
            raise ValueError("compounding must be either 'continuous' or 'annual'.")
        if np.isscalar(maturity):
            return float(values)
        return values

    def forward_rate(self, start: float, end: float, *, compounding: CompoundingMethod = "continuous") -> float:
        if end <= start:
            raise ValueError("end must be strictly greater than start.")
        df_start = self.discount_factor(start, compounding=compounding)
        df_end = self.discount_factor(end, compounding=compounding)
        tau = end - start
        if compounding == "continuous":
            return float(-np.log(df_end / df_start) / tau)
        if compounding == "annual":
            return float((df_start / df_end) ** (1.0 / tau) - 1.0)
        raise ValueError("compounding must be either 'continuous' or 'annual'.")

    def zero_coupon_price(self, maturity: float, *, notional: float = 100.0, compounding: CompoundingMethod = "continuous") -> float:
        return float(notional * self.discount_factor(maturity, compounding=compounding))


__all__ = [
    "CompoundingMethod",
    "InterpolationMethod",
    "NelsonSiegelCurve",
    "NelsonSiegelParameters",
    "YieldCurve",
    "fit_nelson_siegel",
    "nelson_siegel_zero_rate",
]
