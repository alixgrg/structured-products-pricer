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
    """Interpolated zero-rate / discount-factor curve.

    Design choices for pricing stability:
    - discount_factor(0) is always exactly 1.0;
    - curves built from discount factors are anchored with t=0, DF=1;
    - bootstrapped curves interpolate linearly on log discount factors;
    - short-end extrapolation uses the first zero rate, not a constant first DF;
    - long-end extrapolation uses the last zero rate, not a constant last DF.
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
        if not np.any(maturities > 0.0):
            raise ValueError("At least one strictly positive maturity is required.")

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
        maturities = clean[maturity_column].to_numpy(dtype=float)
        rates = clean[rate_column].to_numpy(dtype=float)

        return cls(
            maturities=maturities,
            zero_rates=rates,
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

        The curve is automatically anchored with t=0 and DF=1.0.
        """
        t = np.asarray(maturities, dtype=float)
        df = np.asarray(discount_factors, dtype=float)

        if t.ndim != 1 or df.ndim != 1:
            raise ValueError("maturities and discount_factors must be one-dimensional arrays.")
        if len(t) != len(df):
            raise ValueError("maturities and discount_factors must have the same length.")
        if len(t) == 0:
            raise ValueError("At least one discount-factor point is required.")
        if np.any(~np.isfinite(t)) or np.any(~np.isfinite(df)):
            raise ValueError("maturities and discount_factors must contain finite values.")
        if np.any(t < 0.0):
            raise ValueError("discount-factor curve maturities must be non-negative.")
        if np.any(df <= 0.0) or np.any(df > 2.0):
            raise ValueError("discount factors must be positive and reasonably bounded.")

        order = np.argsort(t)
        t = t[order]
        df = df[order]

        if len(np.unique(t)) != len(t):
            raise ValueError("maturities must be unique.")

        if np.any(t == 0.0):
            df0 = float(df[t == 0.0][0])
            if abs(df0 - 1.0) > 1e-10:
                raise ValueError("discount factor at t=0 must be 1.0.")
        else:
            t = np.insert(t, 0, 0.0)
            df = np.insert(df, 0, 1.0)

        positive = t > 0.0
        if not np.any(positive):
            raise ValueError("At least one strictly positive maturity is required.")

        zero_rates = np.empty_like(t, dtype=float)
        zero_rates[positive] = -np.log(df[positive]) / t[positive]

        first_positive_rate = float(zero_rates[positive][0])
        zero_rates[~positive] = first_positive_rate

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
            first_positive_rate = self._first_positive_zero_rate()
            values = np.where(
                t > 1e-14,
                -np.log(df) / np.maximum(t, 1e-14),
                first_positive_rate,
            )
        elif self.interpolation == "linear":
            values = np.interp(
                t,
                self.maturities,
                self.zero_rates,
                left=self.zero_rates[0],
                right=self.zero_rates[-1],
            )
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

        values = np.where(t <= 1e-14, 1.0, values)

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

    def parallel_shift(self, bump: float, *, name_suffix: str | None = None) -> "YieldCurve":
        """Return a parallel-shifted zero-rate curve."""
        suffix = name_suffix if name_suffix is not None else f"_shift_{float(bump):+.6f}"
        return YieldCurve(
            maturities=self.maturities.copy(),
            zero_rates=self.zero_rates + float(bump),
            interpolation=self.interpolation,
            name=f"{self.name}{suffix}",
            interpolation_on=self.interpolation_on,
        )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "maturity": self.maturities,
                "zero_rate": self.zero_rates,
                "discount_factor": self.discount_factor(self.maturities),
            }
        )

    def check_no_static_arbitrage(self, *, tolerance: float = 1e-10) -> dict[str, bool | float]:
        """Basic ZC curve sanity checks."""
        df = np.asarray(self.discount_factor(self.maturities), dtype=float)
        forwards = []
        for start, end in zip(self.maturities[:-1], self.maturities[1:], strict=False):
            if end > start:
                forwards.append(self.forward_rate(float(start), float(end)))

        forwards_array = np.asarray(forwards, dtype=float) if forwards else np.asarray([], dtype=float)

        return {
            "anchored_at_zero": bool(abs(float(self.discount_factor(0.0)) - 1.0) <= tolerance),
            "positive_discount_factors": bool(np.all(df > 0.0)),
            "non_increasing_discount_factors": bool(np.all(np.diff(df) <= tolerance)),
            "min_discount_factor": float(np.min(df)),
            "max_discount_factor": float(np.max(df)),
            "min_zero_rate": float(np.min(self.zero_rates)),
            "max_zero_rate": float(np.max(self.zero_rates)),
            "min_forward_rate": float(np.min(forwards_array)) if forwards_array.size else np.nan,
            "max_forward_rate": float(np.max(forwards_array)) if forwards_array.size else np.nan,
        }

    def _interpolated_discount_factors(self, maturity: np.ndarray) -> np.ndarray:
        t = np.asarray(maturity, dtype=float)

        xp = self.maturities
        log_df_nodes = self._log_discount_factor_nodes()

        if xp[0] > 0.0:
            xp = np.insert(xp, 0, 0.0)
            log_df_nodes = np.insert(log_df_nodes, 0, 0.0)

        if self.interpolation == "linear":
            log_df = np.interp(t, xp, log_df_nodes)
        else:
            log_df = self._cubic_interpolate_generic(t, xp, log_df_nodes)

        # Stable extrapolation:
        # - before first positive point: flat first zero rate;
        # - after last point: flat last zero rate.
        positive = self.maturities > 0.0
        first_t = float(self.maturities[positive][0])
        last_t = float(self.maturities[-1])

        first_zero_rate = self._first_positive_zero_rate()
        last_zero_rate = float(self.zero_rates[-1])

        below = (t > 1e-14) & (t < first_t)
        above = t > last_t

        if np.any(below):
            log_df = np.asarray(log_df, dtype=float)
            log_df[below] = -first_zero_rate * t[below]

        if np.any(above):
            log_df = np.asarray(log_df, dtype=float)
            log_df[above] = -last_zero_rate * t[above]

        values = np.exp(log_df)
        values = np.where(t <= 1e-14, 1.0, values)
        return values

    def _log_discount_factor_nodes(self) -> np.ndarray:
        log_df = np.empty_like(self.maturities, dtype=float)
        is_zero = self.maturities <= 1e-14
        log_df[is_zero] = 0.0
        log_df[~is_zero] = -self.zero_rates[~is_zero] * self.maturities[~is_zero]
        return log_df

    def _first_positive_zero_rate(self) -> float:
        positive = self.maturities > 0.0
        return float(self.zero_rates[positive][0])

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
    """Parameters of a Nelson-Siegel zero-rate curve."""

    beta0: float
    beta1: float
    beta2: float
    lambda_: float


def nelson_siegel_zero_rate(maturity: float | np.ndarray, params: NelsonSiegelParameters) -> float | np.ndarray:
    """Evaluate the Nelson-Siegel zero rate at one maturity or many maturities."""
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
    """Fit Nelson-Siegel parameters to observed zero rates."""
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
    """Convenience wrapper exposing a fitted Nelson-Siegel curve API."""

    params: NelsonSiegelParameters
    name: str = "nelson_siegel_curve"

    @classmethod
    def fit(cls, maturities: np.ndarray, zero_rates: np.ndarray, *, name: str = "nelson_siegel_curve") -> "NelsonSiegelCurve":
        """Fit a Nelson-Siegel curve from market zero rates."""
        return cls(params=fit_nelson_siegel(maturities, zero_rates), name=name)

    def zero_rate(self, maturity: float | np.ndarray) -> float | np.ndarray:
        """Return the fitted zero rate at a given maturity."""
        return nelson_siegel_zero_rate(maturity, self.params)

    def discount_factor(self, maturity: float | np.ndarray, *, compounding: CompoundingMethod = "continuous") -> float | np.ndarray:
        """Return the discount factor implied by the fitted curve."""
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
        """Return the forward rate implied between two maturities."""
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
        """Return the zero-coupon bond price implied by the fitted curve."""
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
