"""Rates package public API."""

from src.rates.bootstrap import (
    BootstrapInstrumentCheck,
    BootstrapPoint,
    BootstrapResult,
    bootstrap_yield_curve,
)
from src.rates.market_instruments import (
    BootstrapMarket,
    DepositQuote,
    FRAQuote,
    SwapQuote,
)
from src.rates.yield_curve import (
    CompoundingMethod,
    InterpolationMethod,
    NelsonSiegelCurve,
    NelsonSiegelParameters,
    YieldCurve,
    fit_nelson_siegel,
    nelson_siegel_zero_rate,
)

__all__ = [
    "BootstrapInstrumentCheck",
    "BootstrapMarket",
    "BootstrapPoint",
    "BootstrapResult",
    "CompoundingMethod",
    "DepositQuote",
    "FRAQuote",
    "InterpolationMethod",
    "NelsonSiegelCurve",
    "NelsonSiegelParameters",
    "SwapQuote",
    "YieldCurve",
    "bootstrap_yield_curve",
    "fit_nelson_siegel",
    "nelson_siegel_zero_rate",
]
