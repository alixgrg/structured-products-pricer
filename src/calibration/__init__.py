"""Calibration package public API."""

from src.calibration.base import CalibrationResult
from src.calibration.implied_vol import (
    ImpliedVolSurface,
    build_smile_slice,
    build_surface_grid,
    calibrate_implied_vol_panel,
    clean_option_panel,
    export_options_normalized,
    implied_volatility_from_price,
    normalize_option_surface_quotes,
)
from src.calibration.market_validation import (
    MarketErrorThresholds,
    RepricingValidationResult,
    VolatilitySurfaceLike,
    reprice_vanilla_market_quotes,
)
from src.calibration.svi import (
    SSVIParameters,
    SSVIVolSurface,
    SVIParameters,
    SVISlice,
    SVIVolSurface,
    check_butterfly_arbitrage_slice,
    check_calendar_arbitrage_surface,
    fit_svi_slice,
    ssvi_theta,
    ssvi_total_variance,
    svi_total_variance,
)
from src.calibration.vol_surface_registry import (
    VolSurfaceKey,
    VolSurfaceRecord,
    VolSurfaceRegistry,
    choose_default_underlying,
)

__all__ = [
    "CalibrationResult",
    "ImpliedVolSurface",
    "MarketErrorThresholds",
    "RepricingValidationResult",
    "SSVIParameters",
    "SSVIVolSurface",
    "SVIParameters",
    "SVISlice",
    "SVIVolSurface",
    "VolatilitySurfaceLike",
    "VolSurfaceKey",
    "VolSurfaceRecord",
    "VolSurfaceRegistry",
    "build_smile_slice",
    "build_surface_grid",
    "calibrate_implied_vol_panel",
    "check_butterfly_arbitrage_slice",
    "check_calendar_arbitrage_surface",
    "choose_default_underlying",
    "clean_option_panel",
    "export_options_normalized",
    "fit_svi_slice",
    "implied_volatility_from_price",
    "normalize_option_surface_quotes",
    "reprice_vanilla_market_quotes",
    "ssvi_theta",
    "ssvi_total_variance",
    "svi_total_variance",
]
