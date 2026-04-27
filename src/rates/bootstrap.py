"""Bootstrapping of zero-coupon curves from deposits, FRAs and swaps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from src.conventions.day_count import year_fraction
from src.rates.market_instruments import BootstrapMarket, DepositQuote, FRAQuote, SwapQuote
from src.rates.yield_curve import InterpolationMethod, YieldCurve


@dataclass(frozen=True, slots=True)
class BootstrapPoint:
    maturity_date: pd.Timestamp
    maturity_years: float
    discount_factor: float
    zero_rate: float
    source: str
    quote_rate: float


@dataclass(frozen=True, slots=True)
class BootstrapInstrumentCheck:
    instrument_type: str
    tenor: str
    maturity_date: pd.Timestamp
    maturity_years: float
    quote_rate: float
    model_rate: float
    rate_error: float
    rate_error_bps: float
    quote_price: float
    model_price: float
    price_error: float
    abs_price_error: float
    pillar_dv01: float


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    curve: YieldCurve
    points: pd.DataFrame
    instrument_checks: pd.DataFrame
    pillar_dv01: pd.DataFrame
    diagnostics: dict[str, float | bool]


def bootstrap_yield_curve(
    market: BootstrapMarket,
    *,
    curve_day_count: str = "ACT/365F",
    interpolation: InterpolationMethod = "linear",
) -> BootstrapResult:
    """Bootstrap a discount curve from deposits, FRAs and swaps.

    Parameters
    ----------
    market:
        Market quotes and valuation date.
    curve_day_count:
        Day-count used to convert instrument dates to curve maturities.
    interpolation:
        Interpolation method for the final curve.
    """
    valuation_date = pd.Timestamp(market.valuation_date).normalize()
    raw_points: list[BootstrapPoint] = []
    discount_factors: dict[pd.Timestamp, float] = {}

    for quote in sorted(market.deposits, key=lambda q: _sort_key_from_tenor(q.tenor)):
        point = _bootstrap_deposit(quote, valuation_date, market, curve_day_count)
        _store_point(point, raw_points, discount_factors)

    for quote in sorted(market.fras, key=lambda q: _sort_key_from_tenor(q.end_tenor)):
        point = _bootstrap_fra(quote, valuation_date, market, discount_factors, curve_day_count, interpolation)
        _store_point(point, raw_points, discount_factors)

    for quote in sorted(market.swaps, key=lambda q: _sort_key_from_tenor(q.maturity_tenor)):
        point = _bootstrap_swap(quote, valuation_date, market, discount_factors, curve_day_count, interpolation)
        _store_point(point, raw_points, discount_factors)

    if len(raw_points) < 2:
        raise ValueError("At least two curve points are required after bootstrapping.")

    # Last quote wins if two instruments map to the same maturity.
    points_by_maturity: dict[float, BootstrapPoint] = {p.maturity_years: p for p in raw_points}
    points = sorted(points_by_maturity.values(), key=lambda p: p.maturity_years)

    maturities = np.array([p.maturity_years for p in points], dtype=float)
    dfs = np.array([p.discount_factor for p in points], dtype=float)

    curve = YieldCurve.from_discount_factors(
        maturities,
        dfs,
        interpolation=interpolation,
        name=market.name,
        interpolation_on="discount_factors",
    )

    frame = pd.DataFrame(
        {
            "maturity_date": [p.maturity_date for p in points],
            "maturity_years": [p.maturity_years for p in points],
            "discount_factor": [p.discount_factor for p in points],
            "zero_rate": [p.zero_rate for p in points],
            "source": [p.source for p in points],
            "quote_rate": [p.quote_rate for p in points],
            "pillar_dv01": [p.maturity_years * p.discount_factor * 1e-4 for p in points],
        }
    )

    instrument_checks = _build_instrument_checks(market, valuation_date, curve, curve_day_count)
    pillar_dv01 = frame[["maturity_date", "maturity_years", "discount_factor", "pillar_dv01", "source"]].copy()

    diagnostics = curve.check_no_static_arbitrage()
    diagnostics.update(
        {
            "point_count": float(len(frame)),
            "min_maturity": float(frame["maturity_years"].min()),
            "max_maturity": float(frame["maturity_years"].max()),
            "max_abs_rate_error_bps": float(instrument_checks["rate_error_bps"].abs().max()) if not instrument_checks.empty else 0.0,
            "max_abs_price_error": float(instrument_checks["abs_price_error"].max()) if not instrument_checks.empty else 0.0,
            "rmse_rate_error_bps": float(np.sqrt(np.mean(np.square(instrument_checks["rate_error_bps"])))) if not instrument_checks.empty else 0.0,
        }
    )

    return BootstrapResult(
        curve=curve,
        points=frame,
        instrument_checks=instrument_checks,
        pillar_dv01=pillar_dv01,
        diagnostics=diagnostics,
    )


def _bootstrap_deposit(
    quote: DepositQuote,
    valuation_date: pd.Timestamp,
    market: BootstrapMarket,
    curve_day_count: str,
) -> BootstrapPoint:
    maturity_date = quote.maturity_date(valuation_date, market.calendar)
    accrual = year_fraction(valuation_date, maturity_date, quote.day_count)
    if accrual <= 0.0:
        raise ValueError(f"Invalid deposit accrual for tenor {quote.tenor}.")

    df = 1.0 / (1.0 + quote.rate * accrual)
    return _make_point(maturity_date, valuation_date, df, f"deposit_{quote.tenor}", quote.rate, curve_day_count)


def _bootstrap_fra(
    quote: FRAQuote,
    valuation_date: pd.Timestamp,
    market: BootstrapMarket,
    discount_factors: dict[pd.Timestamp, float],
    curve_day_count: str,
    interpolation: InterpolationMethod,
) -> BootstrapPoint:
    start_date = quote.start_date(valuation_date, market.calendar)
    end_date = quote.end_date(valuation_date, market.calendar)
    accrual = year_fraction(start_date, end_date, quote.day_count)
    if accrual <= 0.0:
        raise ValueError(f"Invalid FRA accrual for {quote.start_tenor}x{quote.end_tenor}.")

    df_start = _discount_factor_at(start_date, valuation_date, discount_factors, curve_day_count, interpolation)
    df_end = df_start / (1.0 + quote.rate * accrual)
    return _make_point(end_date, valuation_date, df_end, f"fra_{quote.start_tenor}_{quote.end_tenor}", quote.rate, curve_day_count)


def _bootstrap_swap(
    quote: SwapQuote,
    valuation_date: pd.Timestamp,
    market: BootstrapMarket,
    discount_factors: dict[pd.Timestamp, float],
    curve_day_count: str,
    interpolation: InterpolationMethod,
) -> BootstrapPoint:
    schedule = quote.fixed_leg_schedule(valuation_date, market.calendar)
    accruals = quote.accruals(valuation_date, market.calendar)

    if len(schedule) != len(accruals):
        raise RuntimeError("Swap schedule/accrual mismatch.")
    if not schedule:
        raise ValueError(f"Empty swap schedule for maturity {quote.maturity_tenor}.")

    final_date = schedule[-1]

    # If every intermediate coupon date is already a known pillar, the swap
    # equation is linear in P(0, T) and we can use the closed form.
    all_intermediate_known = all(pd.Timestamp(pay_date).normalize() in discount_factors for pay_date in schedule[:-1])

    if all_intermediate_known:
        annuity_known = 0.0
        for pay_date, accrual in zip(schedule[:-1], accruals[:-1], strict=True):
            df = _discount_factor_at(pay_date, valuation_date, discount_factors, curve_day_count, interpolation)
            annuity_known += accrual * df

        final_accrual = accruals[-1]
        denominator = 1.0 + quote.fixed_rate * final_accrual
        numerator = 1.0 - quote.fixed_rate * annuity_known
        if denominator <= 0.0 or numerator <= 0.0:
            raise ValueError(f"Invalid par-swap bootstrap equation for {quote.maturity_tenor}.")

        df_final = numerator / denominator
    else:
        # When some coupon dates are not yet pillars (e.g. sparse long-end
        # quotes), annuity terms depend on interpolation with the final point.
        # Solve the par equation numerically for self-consistency.
        def objective(df_final_guess: float) -> float:
            tmp_discount_factors = dict(discount_factors)
            tmp_discount_factors[final_date] = float(df_final_guess)

            annuity = 0.0
            for pay_date, accrual in zip(schedule, accruals, strict=True):
                pay_df = _discount_factor_at(
                    pay_date,
                    valuation_date,
                    tmp_discount_factors,
                    curve_day_count,
                    interpolation,
                )
                annuity += float(accrual) * float(pay_df)

            return quote.fixed_rate * annuity + float(df_final_guess) - 1.0

        lower = 1e-8
        upper = 2.0
        f_lower = objective(lower)
        f_upper = objective(upper)

        if f_lower * f_upper > 0.0:
            raise ValueError(
                f"Could not bracket bootstrap root for swap {quote.maturity_tenor}. "
                f"f({lower})={f_lower:.6g}, f({upper})={f_upper:.6g}"
            )

        df_final = float(brentq(objective, lower, upper, xtol=1e-12, maxiter=200))

    if df_final <= 0.0:
        raise ValueError(f"Invalid final swap discount factor for {quote.maturity_tenor}: {df_final}")

    return _make_point(schedule[-1], valuation_date, df_final, f"swap_{quote.maturity_tenor}", quote.fixed_rate, curve_day_count)


def _discount_factor_at(
    date: pd.Timestamp,
    valuation_date: pd.Timestamp,
    discount_factors: dict[pd.Timestamp, float],
    curve_day_count: str,
    interpolation: InterpolationMethod,
) -> float:
    date = pd.Timestamp(date).normalize()
    if date <= valuation_date:
        return 1.0
    if date in discount_factors:
        return float(discount_factors[date])
    if not discount_factors:
        raise ValueError(f"Cannot infer discount factor at {date.date()} before any bootstrapped point.")

    target_t = year_fraction(valuation_date, date, curve_day_count)  # type: ignore[arg-type]
    known = sorted(discount_factors.items(), key=lambda item: item[0])
    maturities = np.array([year_fraction(valuation_date, d, curve_day_count) for d, _ in known], dtype=float)  # type: ignore[arg-type]
    dfs = np.array([df for _, df in known], dtype=float)

    if len(known) == 1:
        z = -np.log(dfs[0]) / maturities[0]
        return float(np.exp(-z * target_t))

    curve = YieldCurve.from_discount_factors(
        maturities,
        dfs,
        interpolation=interpolation,
        name="temporary_bootstrap_curve",
        interpolation_on="discount_factors",
    )
    return float(curve.discount_factor(target_t))


def _make_point(
    maturity_date: pd.Timestamp,
    valuation_date: pd.Timestamp,
    discount_factor: float,
    source: str,
    quote_rate: float,
    curve_day_count: str,
) -> BootstrapPoint:
    maturity_years = year_fraction(valuation_date, maturity_date, curve_day_count)  # type: ignore[arg-type]
    if maturity_years <= 0.0:
        raise ValueError(f"Invalid maturity for point {source}.")
    if discount_factor <= 0.0:
        raise ValueError(f"Invalid discount factor for point {source}: {discount_factor}")

    zero_rate = -np.log(discount_factor) / maturity_years
    return BootstrapPoint(
        maturity_date=pd.Timestamp(maturity_date).normalize(),
        maturity_years=float(maturity_years),
        discount_factor=float(discount_factor),
        zero_rate=float(zero_rate),
        source=source,
        quote_rate=float(quote_rate),
    )


def _store_point(
    point: BootstrapPoint,
    raw_points: list[BootstrapPoint],
    discount_factors: dict[pd.Timestamp, float],
) -> None:
    raw_points.append(point)
    discount_factors[point.maturity_date] = point.discount_factor


def _build_instrument_checks(
    market: BootstrapMarket,
    valuation_date: pd.Timestamp,
    curve: YieldCurve,
    curve_day_count: str,
) -> pd.DataFrame:
    checks: list[BootstrapInstrumentCheck] = []

    for quote in sorted(market.deposits, key=lambda q: _sort_key_from_tenor(q.tenor)):
        maturity_date = quote.maturity_date(valuation_date, market.calendar)
        maturity_years = float(year_fraction(valuation_date, maturity_date, curve_day_count))  # type: ignore[arg-type]
        accrual = float(year_fraction(valuation_date, maturity_date, quote.day_count))  # type: ignore[arg-type]
        model_df = float(curve.discount_factor(maturity_years))
        quote_price = 1.0 / (1.0 + quote.rate * accrual)
        model_rate = (1.0 / model_df - 1.0) / accrual
        checks.append(
            BootstrapInstrumentCheck(
                instrument_type="deposit",
                tenor=quote.tenor,
                maturity_date=maturity_date,
                maturity_years=maturity_years,
                quote_rate=float(quote.rate),
                model_rate=float(model_rate),
                rate_error=float(model_rate - quote.rate),
                rate_error_bps=float((model_rate - quote.rate) * 1e4),
                quote_price=float(quote_price),
                model_price=model_df,
                price_error=float(model_df - quote_price),
                abs_price_error=float(abs(model_df - quote_price)),
                pillar_dv01=float(maturity_years * model_df * 1e-4),
            )
        )

    for quote in sorted(market.fras, key=lambda q: _sort_key_from_tenor(q.end_tenor)):
        start_date = quote.start_date(valuation_date, market.calendar)
        end_date = quote.end_date(valuation_date, market.calendar)
        maturity_years = float(year_fraction(valuation_date, end_date, curve_day_count))  # type: ignore[arg-type]
        accrual = float(year_fraction(start_date, end_date, quote.day_count))  # type: ignore[arg-type]
        start_years = float(year_fraction(valuation_date, start_date, curve_day_count))  # type: ignore[arg-type]
        end_df = float(curve.discount_factor(maturity_years))
        start_df = float(curve.discount_factor(start_years))
        quote_implied_end_df = start_df / (1.0 + quote.rate * accrual)
        model_rate = (start_df / end_df - 1.0) / accrual
        checks.append(
            BootstrapInstrumentCheck(
                instrument_type="fra",
                tenor=f"{quote.start_tenor}x{quote.end_tenor}",
                maturity_date=end_date,
                maturity_years=maturity_years,
                quote_rate=float(quote.rate),
                model_rate=float(model_rate),
                rate_error=float(model_rate - quote.rate),
                rate_error_bps=float((model_rate - quote.rate) * 1e4),
                quote_price=float(quote_implied_end_df),
                model_price=end_df,
                price_error=float(end_df - quote_implied_end_df),
                abs_price_error=float(abs(end_df - quote_implied_end_df)),
                pillar_dv01=float(accrual * end_df * 1e-4),
            )
        )

    for quote in sorted(market.swaps, key=lambda q: _sort_key_from_tenor(q.maturity_tenor)):
        schedule = quote.fixed_leg_schedule(valuation_date, market.calendar)
        accruals = quote.accruals(valuation_date, market.calendar)
        maturity_date = schedule[-1]
        maturity_years = float(year_fraction(valuation_date, maturity_date, curve_day_count))  # type: ignore[arg-type]

        discount_end = float(curve.discount_factor(maturity_years))
        annuity_known = 0.0
        for pay_date, accrual in zip(schedule[:-1], accruals[:-1], strict=True):
            pay_years = float(year_fraction(valuation_date, pay_date, curve_day_count))  # type: ignore[arg-type]
            annuity_known += float(accrual) * float(curve.discount_factor(pay_years))

        final_accrual = float(accruals[-1])
        annuity = annuity_known + final_accrual * discount_end

        par_rate = (1.0 - discount_end) / annuity
        model_price = quote.fixed_rate * annuity + discount_end - 1.0
        checks.append(
            BootstrapInstrumentCheck(
                instrument_type="swap",
                tenor=quote.maturity_tenor,
                maturity_date=maturity_date,
                maturity_years=maturity_years,
                quote_rate=float(quote.fixed_rate),
                model_rate=float(par_rate),
                rate_error=float(par_rate - quote.fixed_rate),
                rate_error_bps=float((par_rate - quote.fixed_rate) * 1e4),
                quote_price=0.0,
                model_price=float(model_price),
                price_error=float(model_price),
                abs_price_error=float(abs(model_price)),
                pillar_dv01=float(annuity * 1e-4),
            )
        )

    if not checks:
        return pd.DataFrame(
            columns=[
                "instrument_type",
                "tenor",
                "maturity_date",
                "maturity_years",
                "quote_rate",
                "model_rate",
                "rate_error",
                "rate_error_bps",
                "quote_price",
                "model_price",
                "price_error",
                "abs_price_error",
                "pillar_dv01",
            ]
        )

    return pd.DataFrame(
        {
            "instrument_type": [item.instrument_type for item in checks],
            "tenor": [item.tenor for item in checks],
            "maturity_date": [item.maturity_date for item in checks],
            "maturity_years": [item.maturity_years for item in checks],
            "quote_rate": [item.quote_rate for item in checks],
            "model_rate": [item.model_rate for item in checks],
            "rate_error": [item.rate_error for item in checks],
            "rate_error_bps": [item.rate_error_bps for item in checks],
            "quote_price": [item.quote_price for item in checks],
            "model_price": [item.model_price for item in checks],
            "price_error": [item.price_error for item in checks],
            "abs_price_error": [item.abs_price_error for item in checks],
            "pillar_dv01": [item.pillar_dv01 for item in checks],
        }
    )


def _sort_key_from_tenor(tenor: str) -> float:
    from src.conventions.day_count import Tenor

    return Tenor.parse(tenor).years_approx()


__all__ = [
    "BootstrapPoint",
    "BootstrapInstrumentCheck",
    "BootstrapResult",
    "bootstrap_yield_curve",
]
