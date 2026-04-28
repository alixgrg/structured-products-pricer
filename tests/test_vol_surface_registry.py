import numpy as np
import pandas as pd
import pytest

from src.calibration.implied_vol import (
    ImpliedVolSurface,
    normalize_option_surface_quotes,
)
from src.calibration.vol_surface_registry import (
    VolSurfaceRegistry,
    choose_default_underlying,
)


def _synthetic_options() -> pd.DataFrame:
    rows = []
    valuation_date = pd.Timestamp("2026-04-28")

    for underlying, spot, base_iv in [
        ("MSFT", 100.0, 0.22),
        ("AAPL", 180.0, 0.25),
    ]:
        for maturity in [0.25, 0.50, 1.00]:
            for moneyness in [0.80, 0.90, 1.00, 1.10, 1.20]:
                strike = spot * moneyness
                iv = base_iv + 0.04 * abs(np.log(moneyness)) + 0.01 * maturity
                rows.append(
                    {
                        "underlying": underlying,
                        "valuation_date": valuation_date,
                        "maturity": maturity,
                        "strike": strike,
                        "underlying_price": spot,
                        "iv": iv,
                        "option_type": "call",
                    }
                )

    return pd.DataFrame(rows)


def test_options_normalized_keeps_required_surface_columns() -> None:
    normalized = normalize_option_surface_quotes(_synthetic_options())

    required = {
        "underlying",
        "valuation_date",
        "maturity",
        "strike",
        "iv",
        "option_type",
        "log_moneyness",
        "time_to_maturity_years",
        "implied_vol",
    }

    assert required.issubset(normalized.columns)
    assert set(normalized["underlying"]) == {"MSFT", "AAPL"}
    assert normalized["valuation_date"].nunique() == 1


def test_implied_vol_surface_rejects_mixed_underlyings() -> None:
    normalized = normalize_option_surface_quotes(_synthetic_options())

    with pytest.raises(ValueError, match="multiple underlying"):
        ImpliedVolSurface.from_quotes(
            normalized.rename(columns={"maturity": "time_to_maturity_years", "iv": "implied_vol"}),
            maturity_column="time_to_maturity_years",
            log_moneyness_column="log_moneyness",
            iv_column="implied_vol",
        )


def test_vol_surface_registry_builds_one_surface_per_underlying_date() -> None:
    normalized = normalize_option_surface_quotes(_synthetic_options())

    registry = VolSurfaceRegistry.from_option_quotes(
        normalized,
        min_quotes_per_surface=8,
        fit_interpolated=True,
        fit_svi=False,
        fit_ssvi=False,
        preferred_underlyings=("MSFT", "AAPL"),
    )

    assert set(registry.available_underlyings()) == {"MSFT", "AAPL"}
    assert registry.default_underlying == "MSFT"

    msft_surface = registry.get("MSFT", "2026-04-28", model="interpolated")
    aapl_surface = registry.get("AAPL", "2026-04-28", model="interpolated")

    assert msft_surface.underlying == "MSFT"
    assert aapl_surface.underlying == "AAPL"

    msft_vol = msft_surface.volatility(0.5, 0.0)
    aapl_vol = aapl_surface.volatility(0.5, 0.0)

    assert msft_vol > 0.0
    assert aapl_vol > 0.0
    assert msft_vol != pytest.approx(aapl_vol)


def test_choose_default_underlying_prefers_msft_then_aapl() -> None:
    normalized = normalize_option_surface_quotes(_synthetic_options())

    selected = choose_default_underlying(
        normalized,
        preferred=("MSFT", "AAPL"),
        min_quotes=8,
    )

    assert selected == "MSFT"


def test_registry_summary_contains_model_flags() -> None:
    normalized = normalize_option_surface_quotes(_synthetic_options())

    registry = VolSurfaceRegistry.from_option_quotes(
        normalized,
        min_quotes_per_surface=8,
        fit_interpolated=True,
        fit_svi=False,
        fit_ssvi=False,
    )

    summary = registry.summary()

    assert {"underlying", "valuation_date", "quote_count", "has_interpolated"}.issubset(summary.columns)
    assert summary["has_interpolated"].all()