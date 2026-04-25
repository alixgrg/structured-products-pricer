"""Yield curve construction and basic interest-rate calculations.

This module provides:
- zero-rate interpolation,
- discount factors,
- forward rates,
- zero-coupon pricing,
- optional Nelson-Siegel fitting.

Rates are expected in decimal format:
2.5% -> 0.025
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


InterpolationMethod = Literal["linear", "cubic"]
CompoundingMethod = Literal["continuous", "annual"]


@dataclass(frozen=True, slots=True)
class YieldCurve:
    """Interpolated zero-rate curve.

    Parameters
    ----------
    maturities:
        Maturities in years.
    zero_rates:
        Zero rates in decimal format.
    interpolation:
        Interpolation method. Use "linear" as the robust default.
    name:
        Optional curve name, for example "France 2026-02-27".
    """

    maturities: np.ndarray
    zero_rates: np.ndarray
    interpolation: InterpolationMethod = "linear"
    name: str = "yield_curve"

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

        order = np.argsort(maturities)
        maturities = maturities[order]
        zero_rates = zero_rates[order]

        unique_maturities, unique_indices = np.unique(maturities, return_index=True)
        if len(unique_maturities) != len(maturities):
            raise ValueError("maturities must be unique.")

        if self.interpolation not in ("linear", "cubic"):
            raise ValueError("interpolation must be either 'linear' or 'cubic'.")

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
        """Build a curve from a normalized rate-curve dataframe."""
        required = {maturity_column, rate_column}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        clean = frame[[maturity_column, rate_column]].dropna().copy()
        clean = clean.sort_values(maturity_column)

        return cls(
            maturities=clean[maturity_column].to_numpy(dtype=float),
            zero_rates=clean[rate_column].to_numpy(dtype=float),
            interpolation=interpolation,
            name=name,
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
        """Build a curve from the normalized output of load_rate_curves.

        If country or observation_date are omitted, the function uses the latest
        available observation date and the first available country in that date.
        """
        required = {
            "country",
            "observation_date",
            "curve_tenor_years",
            "rate_decimal",
        }
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        data = frame.copy()
        data["observation_date"] = pd.to_datetime(data["observation_date"])

        if observation_date is None:
            selected_date = data["observation_date"].max()
        else:
            selected_date = pd.to_datetime(observation_date)

        data = data[data["observation_date"] == selected_date]

        if country is None:
            if data.empty:
                raise ValueError(f"No rate curve data found for date {selected_date}.")
            selected_country = str(data["country"].iloc[0])
        else:
            selected_country = country

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

    def zero_rate(self, maturity: float | np.ndarray) -> float | np.ndarray:
        """Return the interpolated zero rate for one or several maturities."""
        t = np.asarray(maturity, dtype=float)

        if np.any(t < 0.0):
            raise ValueError("maturity must be non-negative.")

        if self.interpolation == "linear":
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
        """Return the forward rate between start and end.

        For continuous compounding:
            f(t1,t2) = -ln(P(0,t2) / P(0,t1)) / (t2 - t1)

        For annual compounding:
            f(t1,t2) = (P(0,t1) / P(0,t2)) ** (1 / (t2 - t1)) - 1
        """
        if start < 0.0:
            raise ValueError("start must be non-negative.")

        if end <= start:
            raise ValueError("end must be strictly greater than start.")

        df_start = self.discount_factor(start, compounding=compounding)
        df_end = self.discount_factor(end, compounding=compounding)
        year_fraction = end - start

        if compounding == "continuous":
            return float(-np.log(df_end / df_start) / year_fraction)

        if compounding == "annual":
            return float((df_start / df_end) ** (1.0 / year_fraction) - 1.0)

        raise ValueError("compounding must be either 'continuous' or 'annual'.")

    def zero_coupon_price(
        self,
        maturity: float,
        *,
        notional: float = 100.0,
        compounding: CompoundingMethod = "continuous",
    ) -> float:
        """Price a zero-coupon bond."""
        return float(notional * self.discount_factor(maturity, compounding=compounding))

    def to_frame(self) -> pd.DataFrame:
        """Return the pillar curve as a dataframe."""
        return pd.DataFrame(
            {
                "maturity": self.maturities,
                "zero_rate": self.zero_rates,
                "discount_factor": self.discount_factor(self.maturities),
            }
        )

    def _cubic_interpolate(self, maturity: np.ndarray) -> np.ndarray:
        """Cubic spline interpolation with scipy, with a clear error if unavailable."""
        try:
            from scipy.interpolate import CubicSpline
        except ImportError as exc:
            raise ImportError(
                "Cubic interpolation requires scipy. Install it or use interpolation='linear'."
            ) from exc

        spline = CubicSpline(
            self.maturities,
            self.zero_rates,
            bc_type="natural",
            extrapolate=True,
        )

        values = spline(maturity)

        below = maturity < self.maturities[0]
        above = maturity > self.maturities[-1]

        if np.any(below):
            values = np.asarray(values)
            values[below] = self.zero_rates[0]

        if np.any(above):
            values = np.asarray(values)
            values[above] = self.zero_rates[-1]

        return values


@dataclass(frozen=True, slots=True)
class NelsonSiegelParameters:
    """Nelson-Siegel parameter container."""

    beta0: float
    beta1: float
    beta2: float
    lambda_: float


def nelson_siegel_zero_rate(
    maturity: float | np.ndarray,
    params: NelsonSiegelParameters,
) -> float | np.ndarray:
    """Evaluate the Nelson-Siegel zero-rate function."""
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
    """Fit Nelson-Siegel parameters by least squares."""
    try:
        from scipy.optimize import minimize
    except ImportError as exc:
        raise ImportError("Nelson-Siegel fitting requires scipy.") from exc

    maturities = np.asarray(maturities, dtype=float)
    zero_rates = np.asarray(zero_rates, dtype=float)

    if initial_guess is None:
        initial_guess = NelsonSiegelParameters(
            beta0=float(zero_rates[-1]),
            beta1=float(zero_rates[0] - zero_rates[-1]),
            beta2=0.0,
            lambda_=1.0,
        )

    def objective(x: np.ndarray) -> float:
        params = NelsonSiegelParameters(
            beta0=float(x[0]),
            beta1=float(x[1]),
            beta2=float(x[2]),
            lambda_=float(x[3]),
        )
        fitted = nelson_siegel_zero_rate(maturities, params)
        errors = fitted - zero_rates
        return float(np.mean(errors**2))

    result = minimize(
        objective,
        x0=np.array(
            [
                initial_guess.beta0,
                initial_guess.beta1,
                initial_guess.beta2,
                initial_guess.lambda_,
            ],
            dtype=float,
        ),
        bounds=[
            (-0.20, 0.20),
            (-0.50, 0.50),
            (-0.50, 0.50),
            (1e-4, 10.0),
        ],
        method="L-BFGS-B",
    )

    if not result.success:
        raise RuntimeError(f"Nelson-Siegel calibration failed: {result.message}")

    return NelsonSiegelParameters(
        beta0=float(result.x[0]),
        beta1=float(result.x[1]),
        beta2=float(result.x[2]),
        lambda_=float(result.x[3]),
    )


@dataclass(frozen=True, slots=True)
class NelsonSiegelCurve:
    """Yield curve backed by a fitted Nelson-Siegel function."""

    params: NelsonSiegelParameters
    name: str = "nelson_siegel_curve"

    @classmethod
    def fit(
        cls,
        maturities: np.ndarray,
        zero_rates: np.ndarray,
        *,
        name: str = "nelson_siegel_curve",
    ) -> "NelsonSiegelCurve":
        params = fit_nelson_siegel(maturities, zero_rates)
        return cls(params=params, name=name)

    def zero_rate(self, maturity: float | np.ndarray) -> float | np.ndarray:
        return nelson_siegel_zero_rate(maturity, self.params)

    def discount_factor(
        self,
        maturity: float | np.ndarray,
        *,
        compounding: CompoundingMethod = "continuous",
    ) -> float | np.ndarray:
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

    def forward_rate(
        self,
        start: float,
        end: float,
        *,
        compounding: CompoundingMethod = "continuous",
    ) -> float:
        if end <= start:
            raise ValueError("end must be strictly greater than start.")

        df_start = self.discount_factor(start, compounding=compounding)
        df_end = self.discount_factor(end, compounding=compounding)
        year_fraction = end - start

        if compounding == "continuous":
            return float(-np.log(df_end / df_start) / year_fraction)

        if compounding == "annual":
            return float((df_start / df_end) ** (1.0 / year_fraction) - 1.0)

        raise ValueError("compounding must be either 'continuous' or 'annual'.")

    def zero_coupon_price(
        self,
        maturity: float,
        *,
        notional: float = 100.0,
        compounding: CompoundingMethod = "continuous",
    ) -> float:
        return float(notional * self.discount_factor(maturity, compounding=compounding))