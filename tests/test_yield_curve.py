from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.rates.yield_curve import (
    NelsonSiegelCurve,
    NelsonSiegelParameters,
    YieldCurve,
    nelson_siegel_zero_rate,
)


def test_discount_factor_at_zero_is_one() -> None:
    curve = YieldCurve(
        maturities=np.array([0.25, 1.0, 2.0, 5.0]),
        zero_rates=np.array([0.02, 0.025, 0.03, 0.035]),
    )

    assert curve.discount_factor(0.0) == pytest.approx(1.0)


def test_discount_factors_are_reasonably_decreasing_for_positive_rates() -> None:
    curve = YieldCurve(
        maturities=np.array([0.25, 1.0, 2.0, 5.0]),
        zero_rates=np.array([0.02, 0.025, 0.03, 0.035]),
    )

    maturities = np.array([0.0, 0.25, 1.0, 2.0, 5.0])
    discount_factors = curve.discount_factor(maturities)

    assert np.all(discount_factors > 0.0)
    assert np.all(np.diff(discount_factors) <= 0.0)


def test_linear_interpolation_returns_expected_mid_rate() -> None:
    curve = YieldCurve(
        maturities=np.array([1.0, 2.0]),
        zero_rates=np.array([0.02, 0.04]),
        interpolation="linear",
    )

    assert curve.zero_rate(1.5) == pytest.approx(0.03)


def test_zero_coupon_repricing_with_continuous_rate() -> None:
    curve = YieldCurve(
        maturities=np.array([1.0, 2.0, 3.0]),
        zero_rates=np.array([0.03, 0.03, 0.03]),
    )

    price = curve.zero_coupon_price(2.0, notional=100.0)

    assert price == pytest.approx(100.0 * np.exp(-0.03 * 2.0))


def test_forward_rate_for_flat_curve_equals_flat_rate() -> None:
    curve = YieldCurve(
        maturities=np.array([1.0, 2.0, 5.0]),
        zero_rates=np.array([0.025, 0.025, 0.025]),
    )

    assert curve.forward_rate(1.0, 5.0) == pytest.approx(0.025)


def test_curve_can_be_built_from_normalized_rate_frame() -> None:
    frame = pd.DataFrame(
        {
            "country": ["France", "France", "France"],
            "observation_date": pd.to_datetime(
                ["2026-02-27", "2026-02-27", "2026-02-27"]
            ),
            "curve_tenor_years": [1.0, 2.0, 5.0],
            "rate_decimal": [0.02, 0.025, 0.03],
        }
    )

    curve = YieldCurve.from_rate_curves(
        frame,
        country="France",
        observation_date="2026-02-27",
    )

    assert curve.name == "France 2026-02-27"
    assert curve.zero_rate(2.0) == pytest.approx(0.025)


def test_curve_uses_latest_date_after_country_filter() -> None:
    frame = pd.DataFrame(
        {
            "country": ["France", "France", "United States", "United States"],
            "observation_date": pd.to_datetime(
                ["2026-02-24", "2026-02-24", "2026-02-25", "2026-02-25"]
            ),
            "curve_tenor_years": [1.0, 2.0, 1.0, 2.0],
            "rate_decimal": [0.02, 0.025, 0.03, 0.035],
        }
    )

    curve = YieldCurve.from_rate_curves(frame, country="France")

    assert curve.name == "France 2026-02-24"
    assert curve.zero_rate(2.0) == pytest.approx(0.025)


def test_cubic_interpolation_smoke() -> None:
    pytest.importorskip("scipy")

    curve = YieldCurve(
        maturities=np.array([0.25, 1.0, 2.0, 5.0]),
        zero_rates=np.array([0.02, 0.025, 0.03, 0.035]),
        interpolation="cubic",
    )

    rate = curve.zero_rate(1.5)

    assert np.isfinite(rate)


def test_nelson_siegel_formula_returns_finite_rate() -> None:
    params = NelsonSiegelParameters(
        beta0=0.03,
        beta1=-0.01,
        beta2=0.02,
        lambda_=1.0,
    )

    rate = nelson_siegel_zero_rate(2.0, params)

    assert np.isfinite(rate)


def test_nelson_siegel_curve_fits_simple_curve() -> None:
    pytest.importorskip("scipy")

    maturities = np.array([0.25, 1.0, 2.0, 5.0, 10.0])
    zero_rates = np.array([0.02, 0.023, 0.026, 0.03, 0.032])

    curve = NelsonSiegelCurve.fit(maturities, zero_rates)

    assert np.isfinite(curve.zero_rate(3.0))
    assert curve.discount_factor(0.0) == pytest.approx(1.0)
