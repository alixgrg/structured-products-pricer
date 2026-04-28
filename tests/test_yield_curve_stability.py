import math

import numpy as np
import pandas as pd
import pytest

from src.market.market_data import MarketData
from src.models.discounting_model import DiscountingModel
from src.products.zero_coupon_bond import ZeroCouponBond
from src.rates.bootstrap import bootstrap_yield_curve
from src.rates.market_instruments import BootstrapMarket, DepositQuote
from src.rates.yield_curve import YieldCurve


def test_discount_curve_is_anchored_at_zero() -> None:
    curve = YieldCurve.from_discount_factors(
        maturities=[0.5, 1.0, 2.0],
        discount_factors=[0.99, 0.975, 0.94],
        name="test_curve",
    )

    assert curve.discount_factor(0.0) == pytest.approx(1.0, abs=1e-15)
    assert curve.maturities[0] == pytest.approx(0.0)
    assert curve.to_frame().iloc[0]["discount_factor"] == pytest.approx(1.0)


def test_short_end_discount_factor_is_not_flat_first_df() -> None:
    curve = YieldCurve.from_discount_factors(
        maturities=[0.5, 1.0],
        discount_factors=[0.99, 0.975],
        name="test_curve",
    )

    df_0 = curve.discount_factor(0.0)
    df_3m = curve.discount_factor(0.25)
    df_6m = curve.discount_factor(0.5)

    assert df_0 == pytest.approx(1.0)
    assert 1.0 > df_3m > df_6m
    assert df_3m != pytest.approx(df_6m)


def test_log_discount_interpolation_matches_node_values() -> None:
    maturities = [0.5, 1.0, 2.0]
    dfs = [0.99, 0.975, 0.94]

    curve = YieldCurve.from_discount_factors(maturities, dfs)

    for maturity, df in zip(maturities, dfs, strict=True):
        assert curve.discount_factor(maturity) == pytest.approx(df, rel=1e-12)


def test_forward_rate_is_finite_between_pillars() -> None:
    curve = YieldCurve.from_discount_factors(
        maturities=[0.5, 1.0, 2.0],
        discount_factors=[0.99, 0.975, 0.94],
    )

    forward = curve.forward_rate(0.5, 1.0)

    assert np.isfinite(forward)


def test_bootstrap_output_contains_zero_anchor() -> None:
    market = BootstrapMarket(
        valuation_date=pd.Timestamp("2026-01-02"),
        deposits=(
            DepositQuote("3M", 0.03),
            DepositQuote("6M", 0.032),
        ),
        name="test_bootstrap",
    )

    result = bootstrap_yield_curve(market)

    first = result.points.iloc[0]

    assert first["maturity_years"] == pytest.approx(0.0)
    assert first["discount_factor"] == pytest.approx(1.0)
    assert first["source"] == "anchor"
    assert result.diagnostics["anchored_at_zero"] is True


def test_discounting_model_zcb_uses_curve_when_market_rate_absent() -> None:
    curve = YieldCurve.from_discount_factors(
        maturities=[1.0, 5.0],
        discount_factors=[math.exp(-0.03 * 1.0), math.exp(-0.03 * 5.0)],
    )

    model = DiscountingModel(yield_curve=curve)
    product = ZeroCouponBond("ZCB-5Y", notional=100.0, maturity=5.0)

    price = model.price(product)

    assert price == pytest.approx(100.0 * math.exp(-0.03 * 5.0), rel=1e-12)


def test_discounting_model_market_rate_overrides_curve() -> None:
    curve = YieldCurve.from_discount_factors(
        maturities=[1.0, 5.0],
        discount_factors=[math.exp(-0.03 * 1.0), math.exp(-0.03 * 5.0)],
    )

    model = DiscountingModel(yield_curve=curve)
    product = ZeroCouponBond("ZCB-5Y", notional=100.0, maturity=5.0)
    market = MarketData(rate=0.10)

    price = model.price(product, market)

    assert price == pytest.approx(100.0 * math.exp(-0.10 * 5.0), rel=1e-12)


def test_discounting_model_dv01_is_price_change_for_plus_one_bp() -> None:
    curve = YieldCurve.from_discount_factors(
        maturities=[1.0, 5.0],
        discount_factors=[math.exp(-0.03 * 1.0), math.exp(-0.03 * 5.0)],
    )

    model = DiscountingModel(yield_curve=curve)
    product = ZeroCouponBond("ZCB-5Y", notional=100.0, maturity=5.0)

    dv01 = model.dv01(product)

    assert dv01 < 0.0


def test_discounting_model_rho_is_derivative_to_rate() -> None:
    model = DiscountingModel(rate=0.03)
    product = ZeroCouponBond("ZCB-5Y", notional=100.0, maturity=5.0)

    rho = model.rho(product)

    expected = -5.0 * 100.0 * math.exp(-0.03 * 5.0)

    assert rho == pytest.approx(expected, rel=1e-4)